/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/lite/delegates/webnn/webnn_delegate.h"

#include <algorithm>
#include <array>
#include <cinttypes>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#include <emscripten/html5.h>
#include <emscripten/val.h>
#endif

#include <fp16/fp16.h>
#include "tensorflow/lite/builtin_ops.h"
#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/minimal_logging.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/internal/utils/sparsity_format_converter.h"
#include "tensorflow/lite/kernels/kernel_util.h"

namespace tflite {
namespace webnn {
namespace {

void CopyTensorDataInt32OrInt64(int64_t* dst, const TfLiteTensor& tensor,
                                size_t n) {
  if (tensor.type == kTfLiteInt32) {
    const int32_t* data = GetTensorData<int32_t>(&tensor);
    std::copy(data, data + n, dst);
  } else if (tensor.type == kTfLiteInt64) {
    const int64_t* data = GetTensorData<int64_t>(&tensor);
    std::copy(data, data + n, dst);
  }
}

// Forward declaration.
TfLiteStatus DelegatePrepare(TfLiteContext* context, TfLiteDelegate* delegate);

class Delegate {
  friend class Subgraph;

 public:
  explicit Delegate(const emscripten::val wnn_context)
      : wnn_context_(wnn_context) {}

  TfLiteIntArray* PrepareOpsToDelegate(TfLiteContext* context);
  TfLiteDelegate* tflite_delegate() { return &delegate_; }

 private:
  TfLiteDelegate delegate_ = {
      reinterpret_cast<void*>(this),  // .data_
      DelegatePrepare,                // .Prepare
      nullptr,                        // .CopyFromBufferHandle
      nullptr,                        // .CopyToBufferHandle
      nullptr,                        // .FreeBufferHandle
      kTfLiteDelegateFlagsNone,       // .flags
  };

  // Unpacked data for quasi-static tensors, i.e. tensors produced by
  // dequantizing or unpacking static buffers.
  std::vector<char> static_unpacked_data_;
  // Mapping from a tensor index for a quasi-static tensor to the offset to
  // its unpacked data within static_unpacked_data_.
  std::unordered_map<int, size_t> static_unpacked_data_map_;
  // Set of indices of nodes which unpack static data, e.g. Dequantize
  // operators which convert FP16 static weights to FP32. These nodes are simply
  // ignored in the delegate implementation, because their outputs are
  // pre-unpacked in DelegatePrepare.
  std::unordered_set<int> static_unpack_nodes_;
  // Set of indices of tensors with unpacked static sparse weights.
  std::unordered_set<int> static_sparse_weights_;
  emscripten::val wnn_context_;
};

class Subgraph {
 public:
  static Subgraph* Create(TfLiteContext* context,
                          const TfLiteDelegateParams* params,
                          const Delegate* delegate) {
    // Convert subgraph inputs and outputs to hash sets for faster lookup.
    const std::unordered_set<int> inputs(
        &params->input_tensors->data[0],
        &params->input_tensors->data[params->input_tensors->size]);
    std::unordered_set<int> outputs;
    for (int o = 0; o < params->output_tensors->size; o++) {
      const int output_tensor_idx = params->output_tensors->data[o];
      // Exclude quasi-static tensors which may have become subgraph outputs
      // after partitioning.
      if (delegate->static_unpacked_data_map_.count(output_tensor_idx) == 0) {
        outputs.insert(output_tensor_idx);
      }
    }

    TfLiteIntArray* execution_plan;
    if (context->GetExecutionPlan(context, &execution_plan) != kTfLiteOk) {
      return nullptr;
    }

    // Create WebNN graph builder
    emscripten::val wnn_builder =
        emscripten::val::global("MLGraphBuilder").new_(delegate->wnn_context_);
    if (!wnn_builder.as<bool>()) {
      TF_LITE_KERNEL_LOG(context, "Failed to create WebNN graph builder.");
      return nullptr;
    }

    bool has_sparse_weights = false;
    // Detect which tensors are used as inputs or outputs of any subgraph nodes.
    // -1 denotes tensor not used in the subgraph. These indexes will be
    // filtered out and removed later.
    std::vector<int> tensors(context->tensors_size, -1);
    for (int i = 0; i < params->nodes_to_replace->size; i++) {
      const int node_index = params->nodes_to_replace->data[i];

      TfLiteNode* node = nullptr;
      TfLiteRegistration* registration = nullptr;
      if (context->GetNodeAndRegistration(context, node_index, &node,
                                          &registration) != kTfLiteOk) {
        return nullptr;
      }

      // Detect if any of the node's inputs are sparse weights.
      if (!has_sparse_weights) {
        for (int i = 0; i < node->inputs->size; i++) {
          if (delegate->static_sparse_weights_.count(node->inputs->data[i]) !=
              0) {
            has_sparse_weights = true;
          }
        }
      }

      if (delegate->static_unpack_nodes_.count(node_index) != 0) {
        // The node unpacks static input and can be skipped because its input
        // was pre-unpacked in DelegatePrepare.
        continue;
      }

      switch (registration->builtin_code) {
        case kTfLiteBuiltinMean:
        case kTfLiteBuiltinPad:
        case kTfLiteBuiltinReshape:
        case kTfLiteBuiltinResizeBilinear:
        case kTfLiteBuiltinStridedSlice:
        case kTfLiteBuiltinSlice:
          // Ignore all but the first input (axes, static padding, new shape,
          // begins/offsets, sizes), because other inputs are represented as
          // parameters of the WebNN operator rather than extra input.
          {
            const int t = node->inputs->data[0];
            tensors[t] = t;
          }
          break;
        case kTfLiteBuiltinSplit:
          // Ignore the first input (axis),
          // because it is represented as parameters of the WebNN operator
          // rather than extra input.
          {
            const int t = node->inputs->data[1];
            tensors[t] = t;
          }
          break;
        case kTfLiteBuiltinTranspose:
          // Ignore the second input (perm), as it is represented as
          // parameters of the WebNN operator rather than extra input.
          {
            const int t = node->inputs->data[0];
            tensors[t] = t;
            break;
          }
        default:
          // All other operators: process all inputs
          for (int k = 0; k < node->inputs->size; k++) {
            if (registration->builtin_code == kTfLiteBuiltinTransposeConv &&
                k == 0) {
              // Ignore the output size parameter (see above).
              continue;
            }
            const int t = node->inputs->data[k];
            if (t >= 0) {
              tensors[t] = t;
            }
          }
      }
      for (int k = 0; k < node->outputs->size; k++) {
        const int t = node->outputs->data[k];
        if (t >= 0) {
          tensors[t] = t;
        }
      }
    }
    // Filter out and remove -1 (unused) indexes.
    tensors.erase(std::remove_if(tensors.begin(), tensors.end(),
                                 [](int i) { return i < 0; }),
                  tensors.end());
    std::sort(tensors.begin(), tensors.end());

    // Create a set of quasi-static tensors for VisitNode function
    std::unordered_set<int> quasi_static_tensors;
    for (const std::pair<const int, size_t>& entry :
         delegate->static_unpacked_data_map_) {
      quasi_static_tensors.insert(entry.first);
    }

    // WebNN operands for TFLite tensors
    std::unordered_map<int, emscripten::val> webnn_operands;
    std::unordered_set<int> compute_inputs;
    for (int t : tensors) {
      std::string datatype;
      switch (context->tensors[t].type) {
        case kTfLiteFloat32:
          datatype = "float32";
          break;
        default:
          TF_LITE_KERNEL_LOG(
              context,
              "unsupported datatype (%s) of tensor %d in WebNN delegate",
              TfLiteTypeGetName(context->tensors[t].type), t);
          return nullptr;
      }

      const void* data = nullptr;
      if (context->tensors[t].allocation_type == kTfLiteMmapRo) {
        data = context->tensors[t].data.raw_const;
      } else {
        // Check for quasi-static data.
        const auto it = delegate->static_unpacked_data_map_.find(t);
        if (it != delegate->static_unpacked_data_map_.end()) {
          data = delegate->static_unpacked_data_.data() + it->second;
        }
      }

      std::vector<int32_t> dims(
          &context->tensors[t].dims->data[0],
          &context->tensors[t].dims->data[context->tensors[t].dims->size]);

      if (inputs.count(t) != 0 || quasi_static_tensors.count(t) != 0) {
        emscripten::val desc = emscripten::val::object();
        desc.set("type", emscripten::val(datatype));
        desc.set("dataType", emscripten::val(datatype));
        // Workaround for single-value operand as which is not supported in Chromium yet
        // TODO: remove this workaround once it is supported
        if (dims.size() == 0) {
          dims = {1};
        }
        desc.set("dimensions", emscripten::val::array(dims));

        emscripten::val operand = emscripten::val::object();
        if (data == nullptr) {
          compute_inputs.insert(t);
          std::string name = std::to_string(t);
          operand = wnn_builder.call<emscripten::val>("input", name, desc);
        } else {
          if (dims.size() == 0) {
            // Create a single-value operand from the specified number of the specified type.
            operand = wnn_builder.call<emscripten::val>(
                "constant", static_cast<const float*>(data)[0],
                emscripten::val(datatype));
          } else {
            // Create an operand for a graph constant.
            auto data_size = context->tensors[t].bytes / 4;
            emscripten::val view{emscripten::typed_memory_view(
                data_size, static_cast<const float*>(data))};
            operand = wnn_builder.call<emscripten::val>("constant", desc, view);
          }
        }
        webnn_operands.insert(std::make_pair(t, operand));
      }
    }

    // Create WebNN nodes for TFLite delegate nodes
    for (int i = 0; i < params->nodes_to_replace->size; i++) {
      const int node_index = params->nodes_to_replace->data[i];
      if (delegate->static_unpack_nodes_.count(node_index)) {
        // The node unpacks static input and can be skipped because its input
        // was pre-unpacked in DelegatePrepare.
        continue;
      }

      TfLiteNode* node = nullptr;
      TfLiteRegistration* registration = nullptr;
      if (context->GetNodeAndRegistration(context, node_index, &node,
                                          &registration) != kTfLiteOk) {
        return nullptr;
      }

      if (VisitNode(wnn_builder, context, registration, node, node_index,
                    quasi_static_tensors, webnn_operands, false) != kTfLiteOk) {
        return nullptr;
      }
    }

    emscripten::val named_operands = emscripten::val::object();
    for (auto o : outputs) {
      std::string name = std::to_string(o);
      if (!webnn_operands.at(o).as<bool>()) {
        TF_LITE_KERNEL_LOG(context, "Invalid operand");
        return nullptr;
      }
      named_operands.set(name, webnn_operands.at(o));
    }

    emscripten::val wnn_graph = wnn_builder.call<emscripten::val>("buildSync", named_operands);
    if (!wnn_graph.as<bool>()) {
      TF_LITE_KERNEL_LOG(context, "failed to build WebNN graph");
      return nullptr;
    }
    return new Subgraph(delegate->wnn_context_, wnn_graph, std::move(compute_inputs), std::move(outputs));
  }

  TfLiteStatus Prepare(TfLiteContext* context) { return kTfLiteOk; }

  TfLiteStatus Invoke(TfLiteContext* context) {
    bool any_pointers_changed = false;
    for (std::pair<int, void*> io_info : externals_) {
      const TfLiteTensor& tensor = context->tensors[io_info.first];
      void* data_pointer = &dummy_data_;
      if (tensor.data.raw != nullptr) {
        data_pointer = tensor.data.raw;
      } else {
        if (tensor.bytes != 0) {
          TF_LITE_KERNEL_LOG(
              context, "unexpected null data pointer in external tensor %d",
              io_info.first);
          return kTfLiteError;
        }
      }
      if (data_pointer != io_info.second) {
        any_pointers_changed = true;
        externals_[io_info.first] = data_pointer;
      }
    }

    if (any_pointers_changed) {
      graph_inputs_ = emscripten::val::object();
      for (int t : inputs_) {
        std::string name = std::to_string(t);
        auto input_size = context->tensors[t].bytes / 4;
        auto input_data = context->tensors[t].data.f;
        emscripten::val view{ emscripten::typed_memory_view(input_size, input_data) };
        graph_inputs_.set(name, view);
      }

      graph_outputs_ = emscripten::val::object();
      for (int t : outputs_) {
        std::string name = std::to_string(t);
        auto output_size = context->tensors[t].bytes / 4;
        auto output_data = context->tensors[t].data.f;
        emscripten::val view{emscripten::typed_memory_view(output_size, output_data)};
        graph_outputs_.set(name, view);
      }
    }

    wnn_context_.call<void>("computeSync", wnn_graph_, graph_inputs_, graph_outputs_);

    return kTfLiteOk;
  }

  static TfLiteStatus CalculatePadding(TfLiteContext* context,
                                       TfLitePadding padding, std::string& auto_pad,
                                       int node_index) {
    switch (padding) {
      case kTfLitePaddingSame: {
        auto_pad = "same-upper";
        return kTfLiteOk;
      }
      case kTfLitePaddingValid:
        auto_pad = "explicit";
        return kTfLiteOk;
      default:
        TF_LITE_MAYBE_KERNEL_LOG(context,
                                 "invalid padding mode (%d) in node #%d",
                                 static_cast<int>(padding), node_index);
        return kTfLiteError;
    }
  }

  static TfLiteStatus CheckConvolutionParams(TfLiteContext* context,
                                             const TfLiteConvParams* params,
                                             int node_index) {
    if (params->stride_width <= 0) {
      TF_LITE_MAYBE_KERNEL_LOG(context, "invalid stride width %d in node #%d",
                               params->stride_width, node_index);
      return kTfLiteError;
    }
    if (params->stride_height <= 0) {
      TF_LITE_MAYBE_KERNEL_LOG(context, "invalid stride height %d in node #%d",
                               params->stride_height, node_index);
      return kTfLiteError;
    }

    if (params->dilation_width_factor <= 0) {
      TF_LITE_MAYBE_KERNEL_LOG(context,
                               "invalid dilation width factor %d in node #%d",
                               params->dilation_width_factor, node_index);
      return kTfLiteError;
    }
    if (params->dilation_height_factor <= 0) {
      TF_LITE_MAYBE_KERNEL_LOG(context,
                               "invalid dilation height factor %d in node #%d",
                               params->dilation_height_factor, node_index);
      return kTfLiteError;
    }

    return kTfLiteOk;
  }

  static TfLiteStatus CheckDepthwiseConvolutionParams(
      TfLiteContext* context, const TfLiteDepthwiseConvParams* params,
      int output_channels, int node_index) {
    if (params->stride_width <= 0) {
      TF_LITE_MAYBE_KERNEL_LOG(context, "invalid stride width %d in node #%d",
                               params->stride_width, node_index);
      return kTfLiteError;
    }
    if (params->stride_height <= 0) {
      TF_LITE_MAYBE_KERNEL_LOG(context, "invalid stride height %d in node #%d",
                               params->stride_height, node_index);
      return kTfLiteError;
    }

    if (params->depth_multiplier <= 0) {
      TF_LITE_MAYBE_KERNEL_LOG(context,
                               "invalid depth multiplier %d in node #%d",
                               params->depth_multiplier, node_index);
      return kTfLiteError;
    }
    if (output_channels % params->depth_multiplier != 0) {
      TF_LITE_MAYBE_KERNEL_LOG(context,
                               "depth multiplier %d is incompatible with "
                               "number of output channels %d in node #%d",
                               params->depth_multiplier, output_channels,
                               node_index);
      return kTfLiteError;
    }

    if (params->dilation_width_factor <= 0) {
      TF_LITE_MAYBE_KERNEL_LOG(context,
                               "invalid dilation width factor %d in node #%d",
                               params->dilation_width_factor, node_index);
      return kTfLiteError;
    }
    if (params->dilation_height_factor <= 0) {
      TF_LITE_MAYBE_KERNEL_LOG(context,
                               "invalid dilation height factor %d in node #%d",
                               params->dilation_height_factor, node_index);
      return kTfLiteError;
    }

    return kTfLiteOk;
  }

  static TfLiteStatus CheckTransposedConvolutionParams(
      TfLiteContext* context, const TfLiteTransposeConvParams* params,
      int node_index) {
    if (params->stride_width <= 0) {
      TF_LITE_MAYBE_KERNEL_LOG(context, "invalid stride width %d in node #%d",
                               params->stride_width, node_index);
      return kTfLiteError;
    }
    if (params->stride_height <= 0) {
      TF_LITE_MAYBE_KERNEL_LOG(context, "invalid stride height %d in node #%d",
                               params->stride_height, node_index);
      return kTfLiteError;
    }

    return kTfLiteOk;
  }

  static TfLiteStatus CheckFullyConnectedParams(
      TfLiteContext* context, const TfLiteFullyConnectedParams* params,
      int node_index) {
    if (params->weights_format != kTfLiteFullyConnectedWeightsFormatDefault) {
      TF_LITE_MAYBE_KERNEL_LOG(
          context, "unsupported non-default weights format in node #%d",
          node_index);
      return kTfLiteError;
    }

    if (params->asymmetric_quantize_inputs) {
      TF_LITE_MAYBE_KERNEL_LOG(
          context, "unsupported asymmetric quantize inputs in node #%d",
          node_index);
      return kTfLiteError;
    }

    return kTfLiteOk;
  }

  static TfLiteStatus CheckPoolingParams(TfLiteContext* context,
                                         const TfLitePoolParams* params,
                                         int node_index) {
    if (params->stride_width <= 0) {
      TF_LITE_MAYBE_KERNEL_LOG(context, "invalid stride width %d in node #%d",
                               params->stride_width, node_index);
      return kTfLiteError;
    }
    if (params->stride_height <= 0) {
      TF_LITE_MAYBE_KERNEL_LOG(context, "invalid stride height %d in node #%d",
                               params->stride_height, node_index);
      return kTfLiteError;
    }

    if (params->filter_width <= 0) {
      TF_LITE_MAYBE_KERNEL_LOG(context, "invalid filter width %d in node #%d",
                               params->filter_width, node_index);
      return kTfLiteError;
    }
    if (params->filter_height <= 0) {
      TF_LITE_MAYBE_KERNEL_LOG(context, "invalid filter height %d in node #%d",
                               params->filter_height, node_index);
      return kTfLiteError;
    }

    if (params->stride_width > params->filter_width) {
      TF_LITE_MAYBE_KERNEL_LOG(
          context,
          "unsupported width stride %d exceeding filter width %d in node #%d",
          params->stride_width, params->filter_width, node_index);
      return kTfLiteError;
    }

    if (params->stride_height > params->filter_height) {
      TF_LITE_MAYBE_KERNEL_LOG(
          context,
          "unsupported height stride %d exceeding filter height %d in node #%d",
          params->stride_height, params->filter_height, node_index);
      return kTfLiteError;
    }

    if (params->filter_width == 1 && params->filter_height == 1 &&
        std::max(params->stride_width, params->stride_height) > 1) {
      TF_LITE_MAYBE_KERNEL_LOG(context,
                               "unsupported pooling with 1x1 filter "
                               "and %dx%d stride in node #%d",
                               params->stride_width, params->stride_height,
                               node_index);
      return kTfLiteError;
    }

    return kTfLiteOk;
  }

  static TfLiteStatus CheckNumInputsAndOutputs(
      TfLiteContext* context, TfLiteNode* node, int min_num_inputs,
      int max_num_inputs, int expected_num_outputs, int node_index) {
    if (node->inputs->size < min_num_inputs ||
        node->inputs->size > max_num_inputs) {
      TF_LITE_MAYBE_KERNEL_LOG(context,
                               "unexpected number of inputs (%d) in node #%d",
                               node->inputs->size, node_index);
      return kTfLiteError;
    }
    if (node->outputs->size != expected_num_outputs) {
      TF_LITE_MAYBE_KERNEL_LOG(
          context, "unexpected number of outputs (%d != %d) in node #%d",
          node->outputs->size, expected_num_outputs, node_index);
      return kTfLiteError;
    }
    return kTfLiteOk;
  }

  static TfLiteStatus CheckNumInputsAndOutputs(TfLiteContext* context,
                                               TfLiteNode* node,
                                               int expected_num_inputs,
                                               int expected_num_outputs,
                                               int node_index) {
    if (node->inputs->size != expected_num_inputs) {
      TF_LITE_MAYBE_KERNEL_LOG(
          context, "unexpected number of inputs (%d != %d) in node #%d",
          node->inputs->size, expected_num_inputs, node_index);
      return kTfLiteError;
    }
    if (node->outputs->size != expected_num_outputs) {
      TF_LITE_MAYBE_KERNEL_LOG(
          context, "unexpected number of outputs (%d != %d) in node #%d",
          node->outputs->size, expected_num_outputs, node_index);
      return kTfLiteError;
    }
    return kTfLiteOk;
  }

  static TfLiteStatus CheckTensorType(TfLiteContext* context,
                                      const TfLiteTensor& tensor,
                                      TfLiteType expected_type,
                                      int tensor_index, int node_index) {
    if (tensor.type != expected_type) {
      TF_LITE_MAYBE_KERNEL_LOG(
          context, "unsupported type %s in tensor #%d in node #%d",
          TfLiteTypeGetName(tensor.type), tensor_index, node_index);
      return kTfLiteError;
    }
    return kTfLiteOk;
  }

  static TfLiteStatus CheckTensorFloat32Type(TfLiteContext* context,
                                             const TfLiteTensor& tensor,
                                             int tensor_index, int node_index) {
    return CheckTensorType(context, tensor, kTfLiteFloat32, tensor_index,
                           node_index);
  }

  static TfLiteStatus CheckTensorFloat32OrQInt8Type(TfLiteContext* context,
                                                    const TfLiteTensor& tensor,
                                                    int tensor_index,
                                                    int node_index) {
    switch (tensor.type) {
      case kTfLiteFloat32:
        break;
      default:
        TF_LITE_MAYBE_KERNEL_LOG(
            context, "unsupported type %s in tensor #%d in node #%d",
            TfLiteTypeGetName(tensor.type), tensor_index, node_index);
        return kTfLiteError;
    }
    return kTfLiteOk;
  }

  static TfLiteStatus CheckTensorFloat32OrQInt32Type(TfLiteContext* context,
                                                     const TfLiteTensor& tensor,
                                                     int tensor_index,
                                                     int node_index) {
    switch (tensor.type) {
      case kTfLiteFloat32:
        break;
      default:
        TF_LITE_MAYBE_KERNEL_LOG(
            context, "unsupported type %s in tensor #%d in node #%d",
            TfLiteTypeGetName(tensor.type), tensor_index, node_index);
        return kTfLiteError;
    }
    return kTfLiteOk;
  }

  static TfLiteStatus CheckTensorInt32Type(TfLiteContext* context,
                                           const TfLiteTensor& tensor,
                                           int tensor_index, int node_index) {
    switch (tensor.type) {
      case kTfLiteInt32:
        return kTfLiteOk;
      default:
        TF_LITE_MAYBE_KERNEL_LOG(
            context, "unsupported type %s in tensor #%d in node #%d",
            TfLiteTypeGetName(tensor.type), tensor_index, node_index);
    }
    return kTfLiteError;
  }

  static TfLiteStatus CheckTensorInt32OrInt64Type(TfLiteContext* context,
                                                  const TfLiteTensor& tensor,
                                                  int tensor_index,
                                                  int node_index) {
    switch (tensor.type) {
      case kTfLiteInt32:
      case kTfLiteInt64:
        return kTfLiteOk;
      default:
        TF_LITE_MAYBE_KERNEL_LOG(
            context, "unsupported type %s in tensor #%d in node #%d",
            TfLiteTypeGetName(tensor.type), tensor_index, node_index);
    }
    return kTfLiteError;
  }

  static TfLiteStatus CheckTensorShape(TfLiteContext* context,
                                       const TfLiteTensor& tensor,
                                       int min_num_dims, int max_num_dims,
                                       int tensor_index) {
    if (min_num_dims == max_num_dims) {
      if (tensor.dims->size != min_num_dims) {
        TF_LITE_MAYBE_KERNEL_LOG(
            context,
            "unsupported number of shape dimensions (%d) in tensor #%d: "
            "%d dimensions expected",
            tensor.dims->size, tensor_index, min_num_dims);
        return kTfLiteError;
      }
    } else {
      if (tensor.dims->size < min_num_dims) {
        TF_LITE_MAYBE_KERNEL_LOG(
            context,
            "unsupported number of shape dimensions (%d) in tensor #%d: "
            "at least %d dimensions expected",
            tensor.dims->size, tensor_index, min_num_dims);
        return kTfLiteError;
      }
      if (tensor.dims->size > max_num_dims) {
        TF_LITE_MAYBE_KERNEL_LOG(
            context,
            "unsupported number of shape dimensions (%d) in tensor #%d: "
            "at most %d dimensions expected",
            tensor.dims->size, tensor_index, max_num_dims);
        return kTfLiteError;
      }
    }
    for (int i = 0; i < tensor.dims->size; i++) {
      if (tensor.dims->data[i] <= 0) {
        TF_LITE_MAYBE_KERNEL_LOG(context,
                                 "invalid num of elements (%d) in "
                                 "dimension #%d in tensor #%d",
                                 tensor.dims->data[i], i, tensor_index);
        return kTfLiteError;
      }
    }
    return kTfLiteOk;
  }

  static TfLiteStatus CheckTensorShape(TfLiteContext* context,
                                       const TfLiteTensor& tensor,
                                       int expected_num_dims,
                                       int tensor_index) {
    return CheckTensorShape(context, tensor, expected_num_dims,
                            expected_num_dims, tensor_index);
  }

  static TfLiteStatus CheckSlopeTensorShape(TfLiteContext* context,
                                            const TfLiteTensor& tensor,
                                            int tensor_index,
                                            int node_index,
                                            int input_last_dim) {
    const auto slope_rank = NumDimensions(&tensor);
    if (slope_rank < 1) {
      TF_LITE_MAYBE_KERNEL_LOG(context,
                               "unexpected number of shape dimensions (%d) in "
                               "slope tensor #%d in Prelu node #%d: "
                               "expected at least a 1D tensor",
                               slope_rank, tensor_index, node_index);
      return kTfLiteError;
    }
    // Validate that all non-channel dimensions (if any) are exactly 1.
    for (int i = 0; i < slope_rank - 1; i++) {
      if (SizeOfDimension(&tensor, i) != 1) {
        TF_LITE_MAYBE_KERNEL_LOG(
            context,
            "unexpected value %d of shape dimension #%d in "
            "slope tensor #%d in Prelu node #%d: "
            "expected 1 for non-channel dimensions",
            tensor.dims->data[i], i, tensor_index, node_index);
        return kTfLiteError;
      }
    }
    // Validate that the input and slope have the same last dimension.
    if (tensor.dims->data[slope_rank - 1] != input_last_dim) {
      TF_LITE_MAYBE_KERNEL_LOG(
          context,
          "unexpected value %d of last dimension in "
          "slope tensor #%d in Prelu node #%d: "
          "expected %d to be same as input's last dimension",
          tensor.dims->data[slope_rank - 1], tensor_index, node_index,
          input_last_dim);
      return kTfLiteError;
    }

    return kTfLiteOk;
  }

  static TfLiteStatus CheckPaddingsTensorShape(TfLiteContext* context,
                                               const TfLiteTensor& tensor,
                                               int expected_rows,
                                               int tensor_index,
                                               int node_index) {
    if (tensor.dims->size != 2) {
      TF_LITE_MAYBE_KERNEL_LOG(context,
                               "unexpected number of shape dimensions (%d) in "
                               "padding tensor #%d in node #%d: "
                               "expected a 2D tensor",
                               tensor.dims->size, tensor_index, node_index);
      return kTfLiteError;
    }
    if (tensor.dims->data[0] != expected_rows) {
      TF_LITE_MAYBE_KERNEL_LOG(context,
                               "unexpected number of rows (%d) in "
                               "padding tensor #%d in node #%d: "
                               "%d rows expected",
                               tensor.dims->size, tensor_index, node_index,
                               expected_rows);
      return kTfLiteError;
    }
    if (tensor.dims->data[1] != 2) {
      TF_LITE_MAYBE_KERNEL_LOG(context,
                               "unexpected number of columns (%d) in "
                               "padding tensor #%d in node #%d: "
                               "2 columns expected",
                               tensor.dims->size, tensor_index, node_index);
      return kTfLiteError;
    }
    return kTfLiteOk;
  }

  static TfLiteStatus CheckAxesTensorShape(TfLiteContext* context,
                                           const TfLiteTensor& tensor,
                                           int tensor_index, int node_index) {
    if (tensor.dims->size != 1) {
      TF_LITE_MAYBE_KERNEL_LOG(context,
                               "unexpected number of shape dimensions (%d) in "
                               "axes tensor #%d in node #%d: "
                               "expected a 1D tensor",
                               tensor.dims->size, tensor_index, node_index);
      return kTfLiteError;
    }
    return kTfLiteOk;
  }

  static TfLiteStatus CheckShapeTensorShape(TfLiteContext* context,
                                            const TfLiteTensor& tensor,
                                            int tensor_index, int node_index) {
    if (tensor.dims->size != 1) {
      TF_LITE_MAYBE_KERNEL_LOG(context,
                               "unexpected number of shape dimensions (%d) in "
                               "shape tensor #%d in node #%d: "
                               "expected a 1D tensor",
                               tensor.dims->size, tensor_index, node_index);
      return kTfLiteError;
    }
    return kTfLiteOk;
  }

  static TfLiteStatus CheckTensorNonDynamicAllocation(
      TfLiteContext* context, const TfLiteTensor& tensor, int tensor_index,
      int node_index) {
    // TODO: remove checks once dynamic tensors are supported
    if (tensor.allocation_type == kTfLiteDynamic) {
      TF_LITE_MAYBE_KERNEL_LOG(
          context,
          "invalid allocation type in tensor #%d in node #%d: "
          "expected non-dynamic tensor",
          tensor_index, node_index);
      return kTfLiteError;
    }
    return kTfLiteOk;
  }

  static TfLiteStatus CheckTensorStaticAllocation(TfLiteContext* context,
                                                  const TfLiteTensor& tensor,
                                                  int tensor_index,
                                                  int node_index) {
    if (tensor.allocation_type != kTfLiteMmapRo ||
        tensor.data.raw_const == nullptr) {
      TF_LITE_MAYBE_KERNEL_LOG(
          context,
          "invalid allocation type in tensor #%d in node #%d: "
          "expected static read-only tensor",
          tensor_index, node_index);
      return kTfLiteError;
    }
    return kTfLiteOk;
  }

  static TfLiteStatus CheckTensorsDimensionMatch(
      TfLiteContext* context, const TfLiteTensor& input_tensor,
      const TfLiteTensor& output_tensor, int dimension_index, int node_index,
      const char* op_name) {
    if (SizeOfDimension(&input_tensor, dimension_index) !=
        SizeOfDimension(&output_tensor, dimension_index)) {
      TF_LITE_MAYBE_KERNEL_LOG(
          context,
          "mismatch in shape dimension %d (%d != %d) in input and output "
          "tensors of %s operator #%d",
          dimension_index, SizeOfDimension(&input_tensor, dimension_index),
          SizeOfDimension(&output_tensor, dimension_index), op_name,
          node_index);
      return kTfLiteError;
    }
    return kTfLiteOk;
  }

  // Check if WebNN op is supported on current platform
  static TfLiteStatus CheckWebNNOpSupport(
      const emscripten::val builder, const std::string op_name) {
    if (!builder[op_name].as<bool>()) {
      TFLITE_LOG_PROD(tflite::TFLITE_LOG_WARNING,
                      "WebNN's %s op is not supported on current platform, "
                      "will fallback to default delegate.",
                      op_name.c_str());
      return kTfLiteError;
    }
    return kTfLiteOk;
  }

  static emscripten::val BuildClamp(const emscripten::val& builder,
                                    const emscripten::val& input,
                                    float min_value, float max_value) {
    emscripten::val options = emscripten::val::object();
    options.set("minValue", min_value);
    options.set("maxValue", max_value);
    return builder.call<emscripten::val>("clamp", input, options);
  }

  static emscripten::val GetClampOperator(
      const emscripten::val& builder, float min_value, float max_value) {
    emscripten::val options = emscripten::val::object();
    options.set("minValue", min_value);
    options.set("maxValue", max_value);
    return builder.call<emscripten::val>("clamp", options);
  }

  static TfLiteStatus CheckActivation(const emscripten::val builder,
                                      int node_index,
                                      TfLiteFusedActivation activation) {
    switch (activation) {
      case kTfLiteActNone:
        return kTfLiteOk;
      case kTfLiteActRelu:
        return CheckWebNNOpSupport(builder, "relu");
      case kTfLiteActReluN1To1:
      case kTfLiteActRelu6:
        return CheckWebNNOpSupport(builder, "clamp");
      case kTfLiteActTanh:
        return CheckWebNNOpSupport(builder, "tanh");
      case kTfLiteActSignBit:
        TFLITE_LOG_PROD(tflite::TFLITE_LOG_WARNING,
            "unsupported fused activation (Sign) in node #%d",
            node_index);
        return kTfLiteError;
      case kTfLiteActSigmoid:
        return CheckWebNNOpSupport(builder, "sigmoid");
      default:
        TFLITE_LOG_PROD(tflite::TFLITE_LOG_WARNING,
                        "invalid fused activation (%d) in node #%d",
                        static_cast<int>(activation), node_index);
        return kTfLiteError;
    }
  }
  static emscripten::val GetActivation(
      const emscripten::val& builder, TfLiteFusedActivation activation) {
    switch (activation) {
      case kTfLiteActRelu:
        return builder.call<emscripten::val>("relu");
      case kTfLiteActReluN1To1:
        return GetClampOperator(builder, -1.0f, +1.0f);
      case kTfLiteActRelu6:
        return GetClampOperator(builder, 0.0f, 6.0f);
      case kTfLiteActTanh:
        return builder.call<emscripten::val>("tanh");
      case kTfLiteActSigmoid:
        return builder.call<emscripten::val>("sigmoid");
      default:
        return emscripten::val::object();
    }
  }

  static TfLiteStatus VisitActivation(
      const emscripten::val& builder, int input_tensor_id, int output_tensor_id,
      TfLiteFusedActivation activation,
      std::unordered_map<int, emscripten::val>& webnn_operands) {
    switch (activation) {
      case kTfLiteActNone:
        return kTfLiteOk;
      case kTfLiteActRelu:
        webnn_operands.at(output_tensor_id) = builder.call<emscripten::val>(
            "relu", webnn_operands.at(input_tensor_id));
        return kTfLiteOk;
      case kTfLiteActReluN1To1:
        webnn_operands.at(output_tensor_id) = BuildClamp(
            builder, webnn_operands.at(input_tensor_id), -1.0f, +1.0f);
        return kTfLiteOk;
      case kTfLiteActRelu6:
        webnn_operands.at(output_tensor_id) =
            BuildClamp(builder, webnn_operands.at(input_tensor_id), 0.0f, 6.0f);
        return kTfLiteOk;
      case kTfLiteActTanh:
        webnn_operands.at(output_tensor_id) = builder.call<emscripten::val>(
            "tanh", webnn_operands.at(input_tensor_id));
        return kTfLiteOk;
      case kTfLiteActSigmoid:
        webnn_operands.at(output_tensor_id) = builder.call<emscripten::val>(
            "sigmoid", webnn_operands.at(input_tensor_id));
        return kTfLiteOk;
      default:
        return kTfLiteOk;
    }
  }

  static TfLiteStatus VisitNode(
      const emscripten::val& builder, TfLiteContext* context,
      TfLiteRegistration* registration, TfLiteNode* node, int node_index,
      const std::unordered_set<int>& quasi_static_tensors,
      std::unordered_map<int, emscripten::val>& webnn_operands,
      const bool detect_supported_op) {
    // TFLite context used for logging purposes. When we create a new node
    // (detect_supported_op is false), logging context is the same as context,
    // and error messages are passed to TFLite. When we detect supported
    // operations (detect_supported_op is true), logging context is null, and
    // error messages are supressed.
    TfLiteContext* logging_context = detect_supported_op ? nullptr : context;
    switch (registration->builtin_code) {
      case kTfLiteBuiltinAbs:
        return VisitUnaryNode(builder, logging_context, node_index, node,
                              context->tensors, webnn_operands,
                              detect_supported_op, "abs");
      case kTfLiteBuiltinAdd: {
        const TfLiteAddParams* add_params =
            static_cast<const TfLiteAddParams*>(node->builtin_data);

        return VisitAddNode(builder, logging_context, node_index, node,
                            context->tensors, add_params, webnn_operands,
                            detect_supported_op);
      }
      case kTfLiteBuiltinSub: {
        const TfLiteSubParams* sub_params =
            static_cast<const TfLiteSubParams*>(node->builtin_data);

        return VisitSubNode(builder, logging_context, node_index, node,
                            context->tensors, sub_params, webnn_operands,
                            detect_supported_op);
      }
      case kTfLiteBuiltinMul: {
        const TfLiteMulParams* mul_params =
            static_cast<const TfLiteMulParams*>(node->builtin_data);

        return VisitMulNode(builder, logging_context, node_index, node,
                            context->tensors, mul_params, webnn_operands,
                            detect_supported_op);
      }
      case kTfLiteBuiltinNeg:
        return VisitUnaryNode(builder, logging_context, node_index, node,
                              context->tensors, webnn_operands,
                              detect_supported_op, "neg");
      case kTfLiteBuiltinPad:
        return VisitPadNode(builder, logging_context, node_index, node,
                            context->tensors, webnn_operands,
                            detect_supported_op);
      case kTfLiteBuiltinPrelu:
        return VisitPreluNode(builder, logging_context, node_index, node,
                              context->tensors, webnn_operands,
                              detect_supported_op);
      case kTfLiteBuiltinAveragePool2d: {
        const TfLitePoolParams* pool_params =
            static_cast<const TfLitePoolParams*>(node->builtin_data);

        return VisitAveragePool2DNode(builder, logging_context, node_index,
                                      node, context->tensors, pool_params,
                                      webnn_operands, detect_supported_op);
      }
      case kTfLiteBuiltinCeil:
        return VisitUnaryNode(builder, logging_context, node_index, node,
                              context->tensors, webnn_operands,
                              detect_supported_op, "ceil");
      case kTfLiteBuiltinMaxPool2d: {
        const TfLitePoolParams* pool_params =
            static_cast<const TfLitePoolParams*>(node->builtin_data);

        return VisitMaxPool2DNode(builder, logging_context, node_index, node,
                                  context->tensors, pool_params, webnn_operands,
                                  detect_supported_op);
      }
      case kTfLiteBuiltinMaximum:
        return VisitMaximumNode(builder, logging_context, node_index, node,
                                context->tensors, webnn_operands,
                                detect_supported_op);
      case kTfLiteBuiltinMean: {
        const TfLiteReducerParams* reducer_params =
            static_cast<const TfLiteReducerParams*>(node->builtin_data);

        return VisitMeanNode(builder, logging_context, node_index, node,
                             context->tensors, reducer_params, webnn_operands,
                             detect_supported_op);
      }
      case kTfLiteBuiltinMinimum:
        return VisitMinimumNode(builder, logging_context, node_index, node,
                                context->tensors, webnn_operands,
                                detect_supported_op);
      case kTfLiteBuiltinConcatenation: {
        const TfLiteConcatenationParams* concat_params =
            static_cast<const TfLiteConcatenationParams*>(node->builtin_data);

        return VisitConcatenationNode(builder, logging_context, node_index,
                                      node, context->tensors, concat_params,
                                      webnn_operands, detect_supported_op);
      }
      case kTfLiteBuiltinConv2d: {
        const TfLiteConvParams* conv_params =
            static_cast<const TfLiteConvParams*>(node->builtin_data);

        return VisitConv2DNode(builder, logging_context, node_index, node,
                               context->tensors, conv_params,
                               quasi_static_tensors, webnn_operands,
                               detect_supported_op);
      }
      case kTfLiteBuiltinDepthwiseConv2d: {
        const TfLiteDepthwiseConvParams* dwconv_params =
            static_cast<const TfLiteDepthwiseConvParams*>(node->builtin_data);

        return VisitDepthwiseConv2DNode(builder, logging_context, node_index,
                                        node, context->tensors, dwconv_params,
                                        quasi_static_tensors, webnn_operands,
                                        detect_supported_op);
      }
      case kTfLiteBuiltinDiv: {
        const TfLiteDivParams* div_params =
            static_cast<const TfLiteDivParams*>(node->builtin_data);

        return VisitDivNode(builder, logging_context, node_index, node,
                            context->tensors, div_params, webnn_operands,
                            detect_supported_op);
      }
      case kTfLiteBuiltinElu:
        return VisitEluNode(builder, logging_context, node_index, node,
                            context->tensors, webnn_operands,
                            detect_supported_op);
      case kTfLiteBuiltinFullyConnected: {
        const TfLiteFullyConnectedParams* fc_params =
            static_cast<const TfLiteFullyConnectedParams*>(node->builtin_data);

        return VisitFullyConnectedNode(builder, logging_context, node_index,
                                       node, context->tensors, fc_params,
                                       quasi_static_tensors, webnn_operands,
                                       detect_supported_op);
      }
      case kTfLiteBuiltinFloor:
        return VisitUnaryNode(builder, logging_context, node_index, node,
                              context->tensors, webnn_operands,
                              detect_supported_op, "floor");
      case kTfLiteBuiltinHardSwish:
        return VisitUnaryNode(builder, logging_context, node_index, node,
                              context->tensors, webnn_operands,
                              detect_supported_op, "hardSwish");
      case kTfLiteBuiltinLeakyRelu: {
        const TfLiteLeakyReluParams* leaky_relu_params =
            static_cast<const TfLiteLeakyReluParams*>(node->builtin_data);

        return VisitLeakyReluNode(builder, logging_context, node_index, node,
                                  context->tensors, leaky_relu_params,
                                  webnn_operands, detect_supported_op);
      }
      case kTfLiteBuiltinLogistic:
        return VisitUnaryNode(builder, logging_context, node_index, node,
                              context->tensors, webnn_operands,
                              detect_supported_op, "sigmoid");
      case kTfLiteBuiltinRelu:
        return VisitUnaryNode(builder, logging_context, node_index, node,
                              context->tensors, webnn_operands,
                              detect_supported_op, "relu");
      case kTfLiteBuiltinReluN1To1:
        return VisitClampNode(builder, logging_context, node_index, node,
                              context->tensors, webnn_operands,
                              detect_supported_op, -1.0f, +1.0f);
      case kTfLiteBuiltinRelu6:
        return VisitClampNode(builder, logging_context, node_index, node,
                              context->tensors, webnn_operands,
                              detect_supported_op, 0.0f, 6.0f);
      case kTfLiteBuiltinReshape: {
        const TfLiteReshapeParams* reshape_params =
            static_cast<const TfLiteReshapeParams*>(node->builtin_data);

        return VisitReshapeNode(builder, logging_context, node_index, node,
                                context->tensors, reshape_params,
                                webnn_operands, detect_supported_op);
      }
      case kTfLiteBuiltinResizeBilinear: {
        const TfLiteResizeBilinearParams* resize_params =
            static_cast<const TfLiteResizeBilinearParams*>(node->builtin_data);

        return VisitResizeBilinearNode(builder, logging_context, node_index,
                                       node, context->tensors, resize_params,
                                       webnn_operands, detect_supported_op);
      }
      case kTfLiteBuiltinSlice:
        return VisitSliceNode(builder, logging_context, node_index, node,
                              context->tensors, webnn_operands,
                              detect_supported_op);
      case kTfLiteBuiltinSoftmax: {
        const TfLiteSoftmaxParams* softmax_params =
            static_cast<const TfLiteSoftmaxParams*>(node->builtin_data);

        return VisitSoftmaxNode(builder, logging_context, node_index, node,
                                context->tensors, softmax_params,
                                webnn_operands, detect_supported_op);
      }
      case kTfLiteBuiltinSplit: {
        const TfLiteSplitParams* split_params =
            static_cast<const TfLiteSplitParams*>(node->builtin_data);

        return VisitSplitNode(builder, logging_context, node_index, node,
                              context->tensors, split_params, webnn_operands,
                              detect_supported_op);
      }
      case kTfLiteBuiltinStridedSlice: {
        const auto* params =
            static_cast<const TfLiteStridedSliceParams*>(node->builtin_data);
        return VisitStridedSliceNode(builder, logging_context, node_index, node,
                                     context->tensors, params, webnn_operands,
                                     detect_supported_op);
      }
      case kTfLiteBuiltinTanh:
        return VisitUnaryNode(builder, logging_context, node_index, node,
                              context->tensors, webnn_operands,
                              detect_supported_op, "tanh");
      case kTfLiteBuiltinTranspose:
        return VisitTransposeNode(builder, logging_context, node_index, node,
                                  context->tensors, webnn_operands,
                                  detect_supported_op);
      case kTfLiteBuiltinTransposeConv: {
        const TfLiteTransposeConvParams* deconv_params =
            static_cast<const TfLiteTransposeConvParams*>(node->builtin_data);

        return VisitTransposeConvNode(builder, logging_context, node_index,
                                      node, context->tensors, deconv_params,
                                      quasi_static_tensors, webnn_operands,
                                      detect_supported_op);
      }
      case kTfLiteBuiltinUnpack: {
        const TfLiteUnpackParams* unpack_params =
            static_cast<const TfLiteUnpackParams*>(node->builtin_data);

        return VisitUnpackNode(builder, logging_context, node_index, node,
                               context->tensors, unpack_params, webnn_operands,
                               detect_supported_op);
      }
      case kTfLiteBuiltinCustom: {
        if (strcmp(registration->custom_name, "Convolution2DTransposeBias") ==
            0) {
          TfLiteTransposeConvParams deconv_params = {kTfLitePaddingUnknown};
          std::memcpy(&deconv_params, node->custom_initial_data,
                      node->custom_initial_data_size);

          return VisitMediaPipeDeconvolutionNode(
              builder, logging_context, node_index, node, context->tensors,
              &deconv_params, quasi_static_tensors, webnn_operands,
              detect_supported_op);
        }
        return kTfLiteError;
      }
      default:
        return kTfLiteError;
    }
  }

  static TfLiteStatus VisitAddNode(
      const emscripten::val& builder, TfLiteContext* logging_context,
      int node_index, TfLiteNode* node, const TfLiteTensor* tensors,
      const TfLiteAddParams* add_params,
      std::unordered_map<int, emscripten::val>& webnn_operands,
      const bool detect_supported_op) {
    TF_LITE_ENSURE_STATUS(
        CheckNumInputsAndOutputs(logging_context, node, 2, 1, node_index));

    const int input1_tensor_id = node->inputs->data[0];
    const TfLiteTensor& input1_tensor = tensors[input1_tensor_id];
    TF_LITE_ENSURE_STATUS(CheckTensorFloat32OrQInt8Type(
        logging_context, input1_tensor, input1_tensor_id, node_index));
    TF_LITE_ENSURE_STATUS(CheckTensorNonDynamicAllocation(
        logging_context, input1_tensor, input1_tensor_id, node_index));

    const int input2_tensor_id = node->inputs->data[1];
    const TfLiteTensor& input2_tensor = tensors[input2_tensor_id];
    TF_LITE_ENSURE_STATUS(CheckTensorFloat32OrQInt8Type(
        logging_context, input2_tensor, input2_tensor_id, node_index));
    TF_LITE_ENSURE_STATUS(CheckTensorNonDynamicAllocation(
        logging_context, input2_tensor, input2_tensor_id, node_index));

    const int output_tensor_id = node->outputs->data[0];
    const TfLiteTensor& output_tensor = tensors[output_tensor_id];
    TF_LITE_ENSURE_STATUS(CheckTensorFloat32OrQInt8Type(
        logging_context, output_tensor, output_tensor_id, node_index));
    TF_LITE_ENSURE_STATUS(CheckTensorNonDynamicAllocation(
        logging_context, output_tensor, output_tensor_id, node_index));

    if (detect_supported_op) {
      TF_LITE_ENSURE_STATUS(CheckWebNNOpSupport(builder, "add"));
      if (add_params != nullptr) {
        TF_LITE_ENSURE_STATUS(
            CheckActivation(builder, node_index, add_params->activation));
      }
    } else {
      TF_LITE_ENSURE(logging_context, webnn_operands.at(input1_tensor_id).as<bool>());
      TF_LITE_ENSURE(logging_context, webnn_operands.at(input2_tensor_id).as<bool>());
      webnn_operands.insert(std::make_pair(output_tensor_id,
          builder.call<emscripten::val>("add",
                                        webnn_operands.at(input1_tensor_id),
                                        webnn_operands.at(input2_tensor_id))));
      TF_LITE_ENSURE(logging_context, webnn_operands.at(output_tensor_id).as<bool>());

      if (add_params != nullptr) {
        TF_LITE_ENSURE_STATUS(
            VisitActivation(builder, output_tensor_id, output_tensor_id,
                            add_params->activation, webnn_operands));
      }
    }

    return kTfLiteOk;
  }

  static TfLiteStatus VisitSubNode(
      const emscripten::val& builder, TfLiteContext* logging_context,
      int node_index, TfLiteNode* node, const TfLiteTensor* tensors,
      const TfLiteSubParams* sub_params,
      std::unordered_map<int, emscripten::val>& webnn_operands,
      const bool detect_supported_op) {
    TF_LITE_ENSURE_STATUS(
        CheckNumInputsAndOutputs(logging_context, node, 2, 1, node_index));

    const int input1_tensor_id = node->inputs->data[0];
    const TfLiteTensor& input1_tensor = tensors[input1_tensor_id];
    TF_LITE_ENSURE_STATUS(CheckTensorFloat32OrQInt8Type(
        logging_context, input1_tensor, input1_tensor_id, node_index));
    TF_LITE_ENSURE_STATUS(CheckTensorNonDynamicAllocation(
        logging_context, input1_tensor, input1_tensor_id, node_index));

    const int input2_tensor_id = node->inputs->data[1];
    const TfLiteTensor& input2_tensor = tensors[input2_tensor_id];
    TF_LITE_ENSURE_STATUS(CheckTensorFloat32OrQInt8Type(
        logging_context, input2_tensor, input2_tensor_id, node_index));
    TF_LITE_ENSURE_STATUS(CheckTensorNonDynamicAllocation(
        logging_context, input2_tensor, input2_tensor_id, node_index));

    const int output_tensor_id = node->outputs->data[0];
    const TfLiteTensor& output_tensor = tensors[output_tensor_id];
    TF_LITE_ENSURE_STATUS(CheckTensorFloat32OrQInt8Type(
        logging_context, output_tensor, output_tensor_id, node_index));
    TF_LITE_ENSURE_STATUS(CheckTensorNonDynamicAllocation(
        logging_context, output_tensor, output_tensor_id, node_index));

    if (detect_supported_op) {
      TF_LITE_ENSURE_STATUS(CheckWebNNOpSupport(builder, "sub"));
      if (sub_params != nullptr) {
        TF_LITE_ENSURE_STATUS(
            CheckActivation(builder, node_index, sub_params->activation));
      }
    } else {
      TF_LITE_ENSURE(logging_context, webnn_operands.at(input1_tensor_id).as<bool>());
      TF_LITE_ENSURE(logging_context, webnn_operands.at(input2_tensor_id).as<bool>());
      webnn_operands.insert(std::make_pair(output_tensor_id,
          builder.call<emscripten::val>("sub",
                                        webnn_operands.at(input1_tensor_id),
                                        webnn_operands.at(input2_tensor_id))));
      TF_LITE_ENSURE(logging_context, webnn_operands.at(output_tensor_id).as<bool>());

      if (sub_params != nullptr) {
        TF_LITE_ENSURE_STATUS(
            VisitActivation(builder, output_tensor_id, output_tensor_id,
                            sub_params->activation, webnn_operands));
      }
    }

    return kTfLiteOk;
  }

  static TfLiteStatus VisitMulNode(
      const emscripten::val& builder, TfLiteContext* logging_context,
      int node_index, TfLiteNode* node, const TfLiteTensor* tensors,
      const TfLiteMulParams* mul_params,
      std::unordered_map<int, emscripten::val>& webnn_operands,
      const bool detect_supported_op) {
    TF_LITE_ENSURE_STATUS(
        CheckNumInputsAndOutputs(logging_context, node, 2, 1, node_index));

    const int input1_tensor_id = node->inputs->data[0];
    const TfLiteTensor& input1_tensor = tensors[input1_tensor_id];
    TF_LITE_ENSURE_STATUS(CheckTensorFloat32OrQInt8Type(
        logging_context, input1_tensor, input1_tensor_id, node_index));
    TF_LITE_ENSURE_STATUS(CheckTensorNonDynamicAllocation(
        logging_context, input1_tensor, input1_tensor_id, node_index));

    const int input2_tensor_id = node->inputs->data[1];
    const TfLiteTensor& input2_tensor = tensors[input2_tensor_id];
    TF_LITE_ENSURE_STATUS(CheckTensorFloat32OrQInt8Type(
        logging_context, input2_tensor, input2_tensor_id, node_index));
    TF_LITE_ENSURE_STATUS(CheckTensorNonDynamicAllocation(
        logging_context, input2_tensor, input2_tensor_id, node_index));

    const int output_tensor_id = node->outputs->data[0];
    const TfLiteTensor& output_tensor = tensors[output_tensor_id];
    TF_LITE_ENSURE_STATUS(CheckTensorFloat32OrQInt8Type(
        logging_context, output_tensor, output_tensor_id, node_index));
    TF_LITE_ENSURE_STATUS(CheckTensorNonDynamicAllocation(
        logging_context, output_tensor, output_tensor_id, node_index));

    if (detect_supported_op) {
      TF_LITE_ENSURE_STATUS(CheckWebNNOpSupport(builder, "mul"));
      if (mul_params != nullptr) {
        TF_LITE_ENSURE_STATUS(
            CheckActivation(builder, node_index, mul_params->activation));
      }
    } else {
      TF_LITE_ENSURE(logging_context, webnn_operands.at(input1_tensor_id).as<bool>());
      TF_LITE_ENSURE(logging_context, webnn_operands.at(input2_tensor_id).as<bool>());
      webnn_operands.insert(std::make_pair(output_tensor_id,
          builder.call<emscripten::val>("mul",
                                        webnn_operands.at(input1_tensor_id),
                                        webnn_operands.at(input2_tensor_id))));
      TF_LITE_ENSURE(logging_context, webnn_operands.at(output_tensor_id).as<bool>());
      if (mul_params != nullptr) {
        TF_LITE_ENSURE_STATUS(
            VisitActivation(builder, output_tensor_id, output_tensor_id,
                            mul_params->activation, webnn_operands));
      }
    }

    return kTfLiteOk;
  }

  static TfLiteStatus VisitLeakyReluNode(
      const emscripten::val& builder, TfLiteContext* logging_context,
      int node_index, TfLiteNode* node, const TfLiteTensor* tensors,
      const TfLiteLeakyReluParams* leaky_relu_params,
      std::unordered_map<int, emscripten::val>& webnn_operands,
      const bool detect_supported_op) {
    TF_LITE_ENSURE_STATUS(CheckNumInputsAndOutputs(
        logging_context, node, 1, 1, node_index));

    const int input_tensor_id = node->inputs->data[0];
    const TfLiteTensor& input_tensor = tensors[input_tensor_id];
    TF_LITE_ENSURE_STATUS(CheckTensorFloat32Type(
        logging_context, input_tensor, input_tensor_id, node_index));
    TF_LITE_ENSURE_STATUS(CheckTensorNonDynamicAllocation(
        logging_context, input_tensor, input_tensor_id, node_index));

    const int output_tensor_id = node->outputs->data[0];
    const TfLiteTensor& output_tensor = tensors[output_tensor_id];
    TF_LITE_ENSURE_STATUS(CheckTensorFloat32Type(
        logging_context, output_tensor, output_tensor_id, node_index));
    TF_LITE_ENSURE_STATUS(CheckTensorNonDynamicAllocation(
        logging_context, output_tensor, output_tensor_id, node_index));

    if (detect_supported_op) {
      TF_LITE_ENSURE_STATUS(CheckWebNNOpSupport(builder, "leakyRelu"));
    } else {
      TF_LITE_ENSURE(logging_context,
                     webnn_operands.at(input_tensor_id).as<bool>());
      emscripten::val options = emscripten::val::object();
      options.set("alpha", leaky_relu_params->alpha);
      webnn_operands.insert(std::make_pair(
          output_tensor_id,
          builder.call<emscripten::val>(
              "leakyRelu", webnn_operands.at(input_tensor_id), options)));
      TF_LITE_ENSURE(logging_context,
                     webnn_operands.at(output_tensor_id).as<bool>());
    }

    return kTfLiteOk;
  }

  static TfLiteStatus VisitPadNode(
      const emscripten::val& builder, TfLiteContext* logging_context,
      int node_index, TfLiteNode* node, const TfLiteTensor* tensors,
      std::unordered_map<int, emscripten::val>& webnn_operands,
      const bool detect_supported_op) {
    TF_LITE_ENSURE_STATUS(
        CheckNumInputsAndOutputs(logging_context, node, 2, 1, node_index));

    const int input_tensor_id = node->inputs->data[0];
    const TfLiteTensor& input_tensor = tensors[input_tensor_id];
    TF_LITE_ENSURE_STATUS(CheckTensorFloat32Type(
        logging_context, input_tensor, input_tensor_id, node_index));
    TF_LITE_ENSURE_STATUS(CheckTensorNonDynamicAllocation(
        logging_context, input_tensor, input_tensor_id, node_index));

    const int padding_tensor_id = node->inputs->data[1];
    const TfLiteTensor& paddings_tensor = tensors[padding_tensor_id];
    TF_LITE_ENSURE_STATUS(CheckTensorType(logging_context, paddings_tensor,
                                          kTfLiteInt32, padding_tensor_id,
                                          node_index));
    TF_LITE_ENSURE_STATUS(CheckPaddingsTensorShape(
        logging_context, paddings_tensor, input_tensor.dims->size,
        padding_tensor_id, node_index));
    TF_LITE_ENSURE_STATUS(CheckTensorStaticAllocation(
        logging_context, paddings_tensor, padding_tensor_id, node_index));

    const int output_tensor_id = node->outputs->data[0];
    const TfLiteTensor& output_tensor = tensors[output_tensor_id];
    TF_LITE_ENSURE_STATUS(CheckTensorFloat32Type(
        logging_context, output_tensor, output_tensor_id, node_index));
    TF_LITE_ENSURE_STATUS(CheckTensorNonDynamicAllocation(
        logging_context, output_tensor, output_tensor_id, node_index));

    const int32_t* paddings_data =
        reinterpret_cast<const int32_t*>(paddings_tensor.data.data);
    for (int i = 0; i < paddings_tensor.dims->data[0]; i++) {
      const int32_t pre_padding = paddings_data[i * 2 + 0];
      if (pre_padding < 0) {
        TF_LITE_MAYBE_KERNEL_LOG(
            logging_context,
            "invalid pre-padding %d for dimension #%d in node %d", pre_padding,
            i, node_index);
        return kTfLiteError;
      }

      const int32_t post_padding = paddings_data[i * 2 + 1];
      if (post_padding < 0) {
        TF_LITE_MAYBE_KERNEL_LOG(
            logging_context,
            "invalid post-padding %d for dimension #%d in node %d", pre_padding,
            i, node_index);
        return kTfLiteError;
      }
    }

    if (detect_supported_op) {
      TF_LITE_ENSURE_STATUS(CheckWebNNOpSupport(builder, "pad"));
    } else {
      TF_LITE_ENSURE(logging_context,
                     webnn_operands.at(input_tensor_id).as<bool>());
      emscripten::val beginning_padding = emscripten::val::array();
      emscripten::val ending_padding = emscripten::val::array();
      for (int i = 0; i < paddings_tensor.dims->data[0]; i++) {
        beginning_padding.call<void>(
            "push", static_cast<uint32_t>(paddings_data[i * 2 + 0]));
        ending_padding.call<void>(
            "push", static_cast<uint32_t>(paddings_data[i * 2 + 1]));
      }
      webnn_operands.insert(std::make_pair(
          output_tensor_id, builder.call<emscripten::val>(
                                "pad", webnn_operands.at(input_tensor_id),
                                beginning_padding, ending_padding)));
      TF_LITE_ENSURE(logging_context, webnn_operands.at(output_tensor_id).as<bool>());
    }

    return kTfLiteOk;
  }

  static TfLiteStatus VisitPreluNode(
      const emscripten::val& builder, TfLiteContext* logging_context,
      int node_index, TfLiteNode* node, const TfLiteTensor* tensors,
      std::unordered_map<int, emscripten::val>& webnn_operands,
      const bool detect_supported_op) {
    TF_LITE_ENSURE_STATUS(
        CheckNumInputsAndOutputs(logging_context, node, 2, 1, node_index));

    const int input_tensor_id = node->inputs->data[0];
    const TfLiteTensor& input_tensor = tensors[input_tensor_id];
    TF_LITE_ENSURE_STATUS(CheckTensorFloat32Type(logging_context, input_tensor,
                                                 input_tensor_id, node_index));
    TF_LITE_ENSURE_STATUS(CheckTensorNonDynamicAllocation(
        logging_context, input_tensor, input_tensor_id, node_index));

    const int input_last_dim =
        input_tensor.dims->data[input_tensor.dims->size - 1];

    const int slope_tensor_id = node->inputs->data[1];
    const TfLiteTensor& slope_tensor = tensors[slope_tensor_id];
    TF_LITE_ENSURE_STATUS(CheckTensorFloat32Type(logging_context, slope_tensor,
                                                 slope_tensor_id, node_index));
    TF_LITE_ENSURE_STATUS(CheckSlopeTensorShape(logging_context, slope_tensor,
                                                slope_tensor_id, node_index,
                                                input_last_dim));

    TF_LITE_ENSURE_STATUS(CheckTensorStaticAllocation(
        logging_context, slope_tensor, slope_tensor_id, node_index));

    const int output_tensor_id = node->outputs->data[0];
    const TfLiteTensor& output_tensor = tensors[output_tensor_id];
    TF_LITE_ENSURE_STATUS(CheckTensorFloat32Type(logging_context, output_tensor,
                                                 output_tensor_id, node_index));
    TF_LITE_ENSURE_STATUS(CheckTensorNonDynamicAllocation(
        logging_context, output_tensor, output_tensor_id, node_index));

    if (detect_supported_op) {
      TF_LITE_ENSURE_STATUS(CheckWebNNOpSupport(builder, "prelu"));
    } else {
      TF_LITE_ENSURE(logging_context,
                     webnn_operands.at(input_tensor_id).as<bool>());
      TF_LITE_ENSURE(logging_context,
                     webnn_operands.at(slope_tensor_id).as<bool>());

      webnn_operands.insert(std::make_pair(
          output_tensor_id, builder.call<emscripten::val>(
                                "prelu", webnn_operands.at(input_tensor_id),
                                webnn_operands.at(slope_tensor_id))));
      TF_LITE_ENSURE(logging_context,
                     webnn_operands.at(output_tensor_id).as<bool>());
    }

    return kTfLiteOk;
  }

  static TfLiteStatus VisitAveragePool2DNode(
      const emscripten::val& builder, TfLiteContext* logging_context,
      int node_index, TfLiteNode* node, const TfLiteTensor* tensors,
      const TfLitePoolParams* pool_params,
      std::unordered_map<int, emscripten::val>& webnn_operands,
      const bool detect_supported_op) {
    TF_LITE_ENSURE_STATUS(
        CheckNumInputsAndOutputs(logging_context, node, 1, 1, node_index));

    const int input_tensor_id = node->inputs->data[0];
    const TfLiteTensor& input_tensor = tensors[input_tensor_id];
    TF_LITE_ENSURE_STATUS(CheckTensorFloat32Type(
        logging_context, input_tensor, input_tensor_id, node_index));
    TF_LITE_ENSURE_STATUS(CheckTensorNonDynamicAllocation(
        logging_context, input_tensor, input_tensor_id, node_index));

    const int output_tensor_id = node->outputs->data[0];
    const TfLiteTensor& output_tensor = tensors[output_tensor_id];
    TF_LITE_ENSURE_STATUS(CheckTensorFloat32Type(
        logging_context, output_tensor, output_tensor_id, node_index));
    TF_LITE_ENSURE_STATUS(CheckTensorNonDynamicAllocation(
        logging_context, output_tensor, output_tensor_id, node_index));

    TF_LITE_ENSURE_STATUS(
        CheckPoolingParams(logging_context, pool_params, node_index));

    std::string auto_pad;
    TF_LITE_ENSURE_STATUS(CalculatePadding(
        logging_context, pool_params->padding, auto_pad, node_index));

    if (detect_supported_op) {
      TF_LITE_ENSURE_STATUS(
          CheckActivation(builder, node_index, pool_params->activation));
      if (pool_params->filter_height != 1 || pool_params->filter_width != 1) {
        TF_LITE_ENSURE_STATUS(CheckWebNNOpSupport(builder, "averagePool2d"));
      }
    } else {
      TF_LITE_ENSURE(logging_context, webnn_operands.at(input_tensor_id).as<bool>());
      if (pool_params->filter_height == 1 && pool_params->filter_width == 1) {
        // Only do activation.
        webnn_operands.insert(std::make_pair(output_tensor_id, webnn_operands.at(input_tensor_id)));
      } else {
        std::vector<int32_t> strides = {
            pool_params->stride_height, pool_params->stride_width};
        std::vector<int32_t> windowDimensions = {
            pool_params->filter_height, pool_params->filter_width};

        emscripten::val options = emscripten::val::object();
        options.set("autoPad", emscripten::val(auto_pad));
        options.set("strides", emscripten::val::array(strides));
        options.set("windowDimensions", emscripten::val::array(windowDimensions));
        options.set("layout", emscripten::val("nhwc"));
        webnn_operands.insert(std::make_pair(
            output_tensor_id,
            builder.call<emscripten::val>("averagePool2d",
                                          webnn_operands.at(input_tensor_id),
                                          options)));
      }
      TF_LITE_ENSURE(logging_context, webnn_operands.at(output_tensor_id).as<bool>());

      TF_LITE_ENSURE_STATUS(
          VisitActivation(builder, output_tensor_id, output_tensor_id,
                          pool_params->activation, webnn_operands));
    }

    return kTfLiteOk;
  }

  static TfLiteStatus VisitMaxPool2DNode(
      const emscripten::val& builder, TfLiteContext* logging_context,
      int node_index, TfLiteNode* node, const TfLiteTensor* tensors,
      const TfLitePoolParams* pool_params,
      std::unordered_map<int, emscripten::val>& webnn_operands,
      const bool detect_supported_op) {
    TF_LITE_ENSURE_STATUS(
        CheckNumInputsAndOutputs(logging_context, node, 1, 1, node_index));

    const int input_tensor_id = node->inputs->data[0];
    const TfLiteTensor& input_tensor = tensors[input_tensor_id];
    TF_LITE_ENSURE_STATUS(CheckTensorFloat32Type(
        logging_context, input_tensor, input_tensor_id, node_index));
    TF_LITE_ENSURE_STATUS(CheckTensorNonDynamicAllocation(
        logging_context, input_tensor, input_tensor_id, node_index));

    const int output_tensor_id = node->outputs->data[0];
    const TfLiteTensor& output_tensor = tensors[output_tensor_id];
    TF_LITE_ENSURE_STATUS(CheckTensorFloat32Type(
        logging_context, output_tensor, output_tensor_id, node_index));
    TF_LITE_ENSURE_STATUS(CheckTensorNonDynamicAllocation(
        logging_context, output_tensor, output_tensor_id, node_index));

    TF_LITE_ENSURE_STATUS(
        CheckPoolingParams(logging_context, pool_params, node_index));

    std::string auto_pad;
    TF_LITE_ENSURE_STATUS(CalculatePadding(
        logging_context, pool_params->padding, auto_pad, node_index));

    if (detect_supported_op) {
      TF_LITE_ENSURE_STATUS(CheckWebNNOpSupport(builder, "maxPool2d"));
      TF_LITE_ENSURE_STATUS(
          CheckActivation(builder, node_index, pool_params->activation));
    } else {
      std::vector<int32_t> strides = {
          pool_params->stride_height, pool_params->stride_width};
      std::vector<int32_t> windowDimensions = {
          pool_params->filter_height, pool_params->filter_width};

      emscripten::val options = emscripten::val::object();
      options.set("autoPad", emscripten::val(auto_pad));
      options.set("strides", emscripten::val::array(strides));
      options.set("windowDimensions", emscripten::val::array(windowDimensions));
      options.set("layout", emscripten::val("nhwc"));

      TF_LITE_ENSURE(logging_context, webnn_operands.at(input_tensor_id).as<bool>());
      webnn_operands.insert(std::make_pair(
          output_tensor_id,
          builder.call<emscripten::val>("maxPool2d",
                                        webnn_operands.at(input_tensor_id),
                                        options)));
      TF_LITE_ENSURE(logging_context, webnn_operands.at(output_tensor_id).as<bool>());

      TF_LITE_ENSURE_STATUS(
          VisitActivation(builder, output_tensor_id, output_tensor_id,
                          pool_params->activation, webnn_operands));
    }

    return kTfLiteOk;
  }

  static TfLiteStatus VisitMaximumNode(
      const emscripten::val& builder, TfLiteContext* logging_context,
      int node_index, TfLiteNode* node, const TfLiteTensor* tensors,
      std::unordered_map<int, emscripten::val>& webnn_operands,
      const bool detect_supported_op) {
    TF_LITE_ENSURE_STATUS(
        CheckNumInputsAndOutputs(logging_context, node, 2, 1, node_index));

    const int input1_tensor_id = node->inputs->data[0];
    const TfLiteTensor& input1_tensor = tensors[input1_tensor_id];
    TF_LITE_ENSURE_STATUS(CheckTensorFloat32OrQInt8Type(
        logging_context, input1_tensor, input1_tensor_id, node_index));
    TF_LITE_ENSURE_STATUS(CheckTensorNonDynamicAllocation(
        logging_context, input1_tensor, input1_tensor_id, node_index));

    const int input2_tensor_id = node->inputs->data[1];
    const TfLiteTensor& input2_tensor = tensors[input2_tensor_id];
    TF_LITE_ENSURE_STATUS(CheckTensorFloat32OrQInt8Type(
        logging_context, input2_tensor, input2_tensor_id, node_index));
    TF_LITE_ENSURE_STATUS(CheckTensorNonDynamicAllocation(
        logging_context, input2_tensor, input2_tensor_id, node_index));

    const int output_tensor_id = node->outputs->data[0];
    const TfLiteTensor& output_tensor = tensors[output_tensor_id];
    TF_LITE_ENSURE_STATUS(CheckTensorFloat32OrQInt8Type(
        logging_context, output_tensor, output_tensor_id, node_index));
    TF_LITE_ENSURE_STATUS(CheckTensorNonDynamicAllocation(
        logging_context, output_tensor, output_tensor_id, node_index));

    if (detect_supported_op) {
      TF_LITE_ENSURE_STATUS(CheckWebNNOpSupport(builder, "max"));
    } else {
      TF_LITE_ENSURE(logging_context, webnn_operands.at(input1_tensor_id).as<bool>());
      TF_LITE_ENSURE(logging_context, webnn_operands.at(input2_tensor_id).as<bool>());
      webnn_operands.insert(std::make_pair(output_tensor_id,
          builder.call<emscripten::val>("max",
                                        webnn_operands.at(input1_tensor_id),
                                        webnn_operands.at(input2_tensor_id))));
      TF_LITE_ENSURE(logging_context, webnn_operands.at(output_tensor_id).as<bool>());
    }

    return kTfLiteOk;
  }

  static TfLiteStatus VisitMeanNode(
      const emscripten::val& builder, TfLiteContext* logging_context,
      int node_index, TfLiteNode* node, const TfLiteTensor* tensors,
      const TfLiteReducerParams* reducer_params,
      std::unordered_map<int, emscripten::val>& webnn_operands,
      const bool detect_supported_op) {
    TF_LITE_ENSURE_STATUS(
        CheckNumInputsAndOutputs(logging_context, node, 2, 1, node_index));

    const int input_tensor_id = node->inputs->data[0];
    const TfLiteTensor& input_tensor = tensors[input_tensor_id];
    TF_LITE_ENSURE_STATUS(CheckTensorFloat32Type(
        logging_context, input_tensor, input_tensor_id, node_index));
    TF_LITE_ENSURE_STATUS(CheckTensorShape(logging_context, input_tensor, 4,
                                           input_tensor_id));
    TF_LITE_ENSURE_STATUS(CheckTensorNonDynamicAllocation(
        logging_context, input_tensor, input_tensor_id, node_index));

    const int axes_tensor_id = node->inputs->data[1];
    const TfLiteTensor& axes_tensor = tensors[axes_tensor_id];
    TF_LITE_ENSURE_STATUS(CheckTensorType(logging_context, axes_tensor,
                                          kTfLiteInt32, axes_tensor_id,
                                          node_index));
    TF_LITE_ENSURE_STATUS(CheckAxesTensorShape(
        logging_context, axes_tensor, axes_tensor_id, node_index));
    TF_LITE_ENSURE_STATUS(CheckTensorStaticAllocation(
        logging_context, axes_tensor, axes_tensor_id, node_index));

    const int32_t* axes_data =
        reinterpret_cast<const int32_t*>(axes_tensor.data.data);
    const int num_reduction_axes = NumElements(&axes_tensor);
    switch (num_reduction_axes) {
      case 1:
        if (axes_data[0] != 2) {
          TF_LITE_MAYBE_KERNEL_LOG(
              logging_context,
              "unsupported MEAN reduction along non-spatial "
              "axis %d in node %d",
              axes_data[0], node_index);
          return kTfLiteError;
        }
        break;
      case 2:
        if (std::min(axes_data[0], axes_data[1]) != 1 ||
            std::max(axes_data[0], axes_data[1]) != 2) {
          TF_LITE_MAYBE_KERNEL_LOG(
              logging_context,
              "unsupported MEAN reduction along non-spatial "
              "axes %d and %d in node %d",
              std::min(axes_data[0], axes_data[1]),
              std::max(axes_data[0], axes_data[1]), node_index);
          return kTfLiteError;
        }
        break;
      default:
        TF_LITE_MAYBE_KERNEL_LOG(
            logging_context,
            "unsupported MEAN reduction along %d axes in node %d",
            SizeOfDimension(&axes_tensor, 0), node_index);
        return kTfLiteError;
    }

    const int output_tensor_id = node->outputs->data[0];
    const TfLiteTensor& output_tensor = tensors[output_tensor_id];
    TF_LITE_ENSURE_STATUS(CheckTensorFloat32Type(
        logging_context, output_tensor, output_tensor_id, node_index));
    int expected_output_dims = 4;
    if (!reducer_params->keep_dims) {
      expected_output_dims -= num_reduction_axes;
    }
    TF_LITE_ENSURE_STATUS(CheckTensorShape(logging_context, output_tensor,
                                           expected_output_dims,
                                           output_tensor_id));
    TF_LITE_ENSURE_STATUS(CheckTensorNonDynamicAllocation(
        logging_context, output_tensor, output_tensor_id, node_index));

    if (detect_supported_op) {
      TF_LITE_ENSURE_STATUS(CheckWebNNOpSupport(builder, "averagePool2d"));
      if (num_reduction_axes != 2) {
        TFLITE_LOG_PROD(
            tflite::TFLITE_LOG_WARNING,
            "WebNN only support TFLite's mean op with axis = [1, 2], "
            "will fallback to default delegate.");
        return kTfLiteError;
      }
    } else {
      emscripten::val pool2dOptions = emscripten::val::object();
      pool2dOptions.set("layout", emscripten::val("nhwc"));
      TF_LITE_ENSURE(logging_context,
                     webnn_operands.at(input_tensor_id).as<bool>());
      emscripten::val output = builder.call<emscripten::val>(
          "averagePool2d", webnn_operands.at(input_tensor_id), pool2dOptions);
      if (!reducer_params->keep_dims) {
        // Reshape output to 2D tensor
        emscripten::val new_output_shape = emscripten::val::array();
        new_output_shape.call<void>("push", output_tensor.dims->data[0]);
        new_output_shape.call<void>("push", output_tensor.dims->data[1]);
        output =
            builder.call<emscripten::val>("reshape", output, new_output_shape);
      }
      webnn_operands.insert(std::make_pair(output_tensor_id, output));

      TF_LITE_ENSURE(logging_context,
                     webnn_operands.at(output_tensor_id).as<bool>());
    }

    return kTfLiteOk;
  }

  static TfLiteStatus VisitMinimumNode(
      const emscripten::val& builder, TfLiteContext* logging_context,
      int node_index, TfLiteNode* node, const TfLiteTensor* tensors,
      std::unordered_map<int, emscripten::val>& webnn_operands,
      const bool detect_supported_op) {
    TF_LITE_ENSURE_STATUS(
        CheckNumInputsAndOutputs(logging_context, node, 2, 1, node_index));

    const int input1_tensor_id = node->inputs->data[0];
    const TfLiteTensor& input1_tensor = tensors[input1_tensor_id];
    TF_LITE_ENSURE_STATUS(CheckTensorFloat32OrQInt8Type(
        logging_context, input1_tensor, input1_tensor_id, node_index));
    TF_LITE_ENSURE_STATUS(CheckTensorNonDynamicAllocation(
        logging_context, input1_tensor, input1_tensor_id, node_index));

    const int input2_tensor_id = node->inputs->data[1];
    const TfLiteTensor& input2_tensor = tensors[input2_tensor_id];
    TF_LITE_ENSURE_STATUS(CheckTensorFloat32OrQInt8Type(
        logging_context, input2_tensor, input2_tensor_id, node_index));
    TF_LITE_ENSURE_STATUS(CheckTensorNonDynamicAllocation(
        logging_context, input2_tensor, input2_tensor_id, node_index));

    const int output_tensor_id = node->outputs->data[0];
    const TfLiteTensor& output_tensor = tensors[output_tensor_id];
    TF_LITE_ENSURE_STATUS(CheckTensorFloat32OrQInt8Type(
        logging_context, output_tensor, output_tensor_id, node_index));
    TF_LITE_ENSURE_STATUS(CheckTensorNonDynamicAllocation(
        logging_context, output_tensor, output_tensor_id, node_index));

    if (detect_supported_op) {
      TF_LITE_ENSURE_STATUS(CheckWebNNOpSupport(builder, "min"));
    } else {
      TF_LITE_ENSURE(logging_context, webnn_operands.at(input1_tensor_id).as<bool>());
      TF_LITE_ENSURE(logging_context, webnn_operands.at(input2_tensor_id).as<bool>());
      webnn_operands.insert(std::make_pair(output_tensor_id,
          builder.call<emscripten::val>("min",
                                        webnn_operands.at(input1_tensor_id),
                                        webnn_operands.at(input2_tensor_id))));
      TF_LITE_ENSURE(logging_context, webnn_operands.at(output_tensor_id).as<bool>());
    }

    return kTfLiteOk;
  }

  static TfLiteStatus VisitConcatenationNode(
      const emscripten::val& builder, TfLiteContext* logging_context,
      int node_index, TfLiteNode* node, const TfLiteTensor* tensors,
      const TfLiteConcatenationParams* concat_params,
      std::unordered_map<int, emscripten::val>& webnn_operands,
      const bool detect_supported_op) {
    TF_LITE_ENSURE_STATUS(
        CheckNumInputsAndOutputs(logging_context, node, 1, 4, 1, node_index));
    const int num_inputs = NumInputs(node);

    const int output_tensor_id = node->outputs->data[0];
    const TfLiteTensor& output_tensor = tensors[output_tensor_id];
    TF_LITE_ENSURE_STATUS(CheckTensorFloat32OrQInt8Type(
        logging_context, output_tensor, output_tensor_id, node_index));
    TF_LITE_ENSURE_STATUS(CheckTensorNonDynamicAllocation(
        logging_context, output_tensor, output_tensor_id, node_index));

    // Check dimensions
    int axis = concat_params->axis;
    if (axis < 0) axis += NumDimensions(&output_tensor);
    int sum_axis = 0;

    for (int i = 0; i < num_inputs; i++) {
      const TfLiteTensor& input_tensor = tensors[node->inputs->data[i]];
      TF_LITE_ENSURE_STATUS(CheckTensorFloat32OrQInt8Type(
          logging_context, input_tensor, node->inputs->data[i], node_index));
      TF_LITE_ENSURE_STATUS(CheckTensorNonDynamicAllocation(
          logging_context, input_tensor, node->inputs->data[i], node_index));

      TF_LITE_ENSURE_EQ(logging_context, NumDimensions(&input_tensor),
                        NumDimensions(&output_tensor));

      for (int d = 0; d < NumDimensions(&output_tensor); d++) {
        // All dimensions must match except the 'axis'.
        if (d == axis) {
          continue;
        }
        const TfLiteTensor& input_tensor = tensors[node->inputs->data[i]];
        TF_LITE_ENSURE_STATUS(CheckTensorsDimensionMatch(
            logging_context, input_tensor, output_tensor, d, node_index,
            "CONCATENATE"));
      }
      sum_axis += SizeOfDimension(&input_tensor, axis);
    }

    if (SizeOfDimension(&output_tensor, axis) != sum_axis) {
      TF_LITE_MAYBE_KERNEL_LOG(
          logging_context,
          "mismatch in axis dimension %d (%d != %d) in output and input"
          "tensors of CONCATENATE operator #%d",
          axis, SizeOfDimension(&output_tensor, axis), sum_axis, node_index);
      return kTfLiteError;
    }

    if (detect_supported_op) {
      TF_LITE_ENSURE_STATUS(CheckWebNNOpSupport(builder, "concat"));
    } else {
      emscripten::val input_operands = emscripten::val::array();
      for (size_t i = 0; i < num_inputs; ++i) {
        TF_LITE_ENSURE(logging_context,
                       webnn_operands.at(node->inputs->data[i]).as<bool>());
        input_operands.call<void>("push",
                                  webnn_operands.at(node->inputs->data[i]));
      }
      webnn_operands.insert(
          std::make_pair(node->outputs->data[0],
                         builder.call<emscripten::val>("concat", input_operands,
                                                       static_cast<uint32_t>(axis))));
      TF_LITE_ENSURE(logging_context,
                     webnn_operands.at(node->outputs->data[0]).as<bool>());
    }
    return kTfLiteOk;
  }

  static TfLiteStatus VisitConv2DNode(
      const emscripten::val& builder, TfLiteContext* logging_context,
      int node_index, TfLiteNode* node, const TfLiteTensor* tensors,
      const TfLiteConvParams* conv_params,
      const std::unordered_set<int>& quasi_static_tensors,
      std::unordered_map<int, emscripten::val>& webnn_operands,
      const bool detect_supported_op) {
    TF_LITE_ENSURE_STATUS(
        CheckConvolutionParams(logging_context, conv_params, node_index));

    TF_LITE_ENSURE_STATUS(
        CheckNumInputsAndOutputs(logging_context, node, 3, 1, node_index));

    const int input_tensor_id = node->inputs->data[0];
    const TfLiteTensor& input_tensor = tensors[input_tensor_id];
    TF_LITE_ENSURE_STATUS(CheckTensorFloat32OrQInt8Type(
        logging_context, input_tensor, input_tensor_id, node_index));
    TF_LITE_ENSURE_STATUS(CheckTensorShape(logging_context, input_tensor, 4,
                                           input_tensor_id));
    TF_LITE_ENSURE_STATUS(CheckTensorNonDynamicAllocation(
        logging_context, input_tensor, input_tensor_id, node_index));

    const int filter_tensor_id = node->inputs->data[1];
    const TfLiteTensor& filter_tensor = tensors[filter_tensor_id];
    TF_LITE_ENSURE_STATUS(CheckTensorFloat32OrQInt8Type(
        logging_context, filter_tensor, filter_tensor_id, node_index));
    TF_LITE_ENSURE_STATUS(CheckTensorShape(logging_context, filter_tensor, 4,
                                           filter_tensor_id));
    if (quasi_static_tensors.count(filter_tensor_id) == 0) {
      TF_LITE_ENSURE_STATUS(CheckTensorStaticAllocation(
          logging_context, filter_tensor, filter_tensor_id, node_index));
    }

    const int bias_tensor_id = node->inputs->data[2];
    // bias_tensor_id < 0 means without bias.
    if (bias_tensor_id >= 0) {
      const TfLiteTensor& bias_tensor = tensors[bias_tensor_id];
      TF_LITE_ENSURE_STATUS(CheckTensorFloat32OrQInt32Type(
          logging_context, bias_tensor, node->inputs->data[2], node_index));
      TF_LITE_ENSURE_STATUS(CheckTensorShape(logging_context, bias_tensor, 1,
                                            node->inputs->data[2]));
      if (quasi_static_tensors.count(node->inputs->data[2]) == 0) {
        TF_LITE_ENSURE_STATUS(CheckTensorStaticAllocation(
            logging_context, bias_tensor, node->inputs->data[2], node_index));
      }
    }

    const int output_tensor_id = node->outputs->data[0];
    const TfLiteTensor& output_tensor = tensors[output_tensor_id];
    TF_LITE_ENSURE_STATUS(CheckTensorFloat32OrQInt8Type(
        logging_context, output_tensor, output_tensor_id, node_index));
    TF_LITE_ENSURE_STATUS(CheckTensorShape(logging_context, output_tensor, 4,
                                           output_tensor_id));
    TF_LITE_ENSURE_STATUS(CheckTensorNonDynamicAllocation(
        logging_context, output_tensor, output_tensor_id, node_index));

    std::string auto_pad;
    TF_LITE_ENSURE_STATUS(CalculatePadding(
        logging_context, conv_params->padding, auto_pad, node_index));

    if (detect_supported_op) {
      TF_LITE_ENSURE_STATUS(CheckWebNNOpSupport(builder, "conv2d"));
      TF_LITE_ENSURE_STATUS(
          CheckActivation(builder, node_index, conv_params->activation));
    } else {
      std::vector<int32_t> strides = {
          conv_params->stride_height, conv_params->stride_width};
      std::vector<int32_t> dilations = {
          conv_params->dilation_height_factor, conv_params->dilation_width_factor};

      emscripten::val options = emscripten::val::object();
      options.set("autoPad", emscripten::val(auto_pad));
      options.set("strides", emscripten::val::array(strides));
      options.set("dilations", emscripten::val::array(dilations));
      options.set("inputLayout", emscripten::val("nhwc"));
      options.set("filterLayout", emscripten::val("ohwi"));
      TF_LITE_ENSURE(logging_context, webnn_operands.at(input_tensor_id).as<bool>());
      TF_LITE_ENSURE(logging_context, webnn_operands.at(filter_tensor_id).as<bool>());
      if (bias_tensor_id >= 0) {
        TF_LITE_ENSURE(logging_context, webnn_operands.at(bias_tensor_id).as<bool>());
        options.set("bias", webnn_operands.at(bias_tensor_id));
      }

      if (conv_params->activation != kTfLiteActNone) {
        emscripten::val activation_operator = GetActivation(
            builder, conv_params->activation);
        options.set("activation", activation_operator);
      }
      emscripten::val output = builder.call<emscripten::val>(
          "conv2d", webnn_operands.at(input_tensor_id),
          webnn_operands.at(filter_tensor_id), options);
      webnn_operands.insert(std::make_pair(output_tensor_id, output));
      TF_LITE_ENSURE(logging_context, webnn_operands.at(output_tensor_id).as<bool>());
    }

    return kTfLiteOk;
  }

  static TfLiteStatus VisitMediaPipeDeconvolutionNode(
      const emscripten::val& builder, TfLiteContext* logging_context,
      int node_index, TfLiteNode* node, const TfLiteTensor* tensors,
      const TfLiteTransposeConvParams* deconv_params,
      const std::unordered_set<int>& quasi_static_tensors,
      std::unordered_map<int, emscripten::val>& webnn_operands,
      const bool detect_supported_op) {
    TF_LITE_ENSURE_STATUS(
        CheckNumInputsAndOutputs(logging_context, node, 3, 1, node_index));

    const int input_tensor_id = node->inputs->data[0];
    const TfLiteTensor& input_tensor = tensors[input_tensor_id];
    TF_LITE_ENSURE_STATUS(CheckTensorFloat32Type(
        logging_context, input_tensor, input_tensor_id, node_index));
    TF_LITE_ENSURE_STATUS(CheckTensorShape(logging_context, input_tensor, 4,
                                           input_tensor_id));
    TF_LITE_ENSURE_STATUS(CheckTensorNonDynamicAllocation(
        logging_context, input_tensor, input_tensor_id, node_index));

    const int filter_tensor_id = node->inputs->data[1];
    const TfLiteTensor& filter_tensor = tensors[filter_tensor_id];
    TF_LITE_ENSURE_STATUS(CheckTensorFloat32Type(
        logging_context, filter_tensor, filter_tensor_id, node_index));
    TF_LITE_ENSURE_STATUS(CheckTensorShape(logging_context, filter_tensor, 4,
                                           filter_tensor_id));
    if (quasi_static_tensors.count(filter_tensor_id) == 0) {
      TF_LITE_ENSURE_STATUS(CheckTensorStaticAllocation(
          logging_context, filter_tensor, filter_tensor_id, node_index));
    }

    const int bias_tensor_id = node->inputs->data[2];
    const TfLiteTensor& bias_tensor = tensors[bias_tensor_id];
    TF_LITE_ENSURE_STATUS(CheckTensorFloat32Type(
        logging_context, bias_tensor, bias_tensor_id, node_index));
    TF_LITE_ENSURE_STATUS(CheckTensorShape(logging_context, bias_tensor, 1,
                                           bias_tensor_id));
    if (quasi_static_tensors.count(bias_tensor_id) == 0) {
      TF_LITE_ENSURE_STATUS(CheckTensorStaticAllocation(
          logging_context, bias_tensor, bias_tensor_id, node_index));
    }

    const int output_tensor_id = node->outputs->data[0];
    const TfLiteTensor& output_tensor = tensors[output_tensor_id];
    TF_LITE_ENSURE_STATUS(CheckTensorFloat32Type(
        logging_context, output_tensor, output_tensor_id, node_index));
    TF_LITE_ENSURE_STATUS(CheckTensorShape(logging_context, output_tensor, 4,
                                           output_tensor_id));
    TF_LITE_ENSURE_STATUS(CheckTensorNonDynamicAllocation(
        logging_context, output_tensor, output_tensor_id, node_index));

    const int output_channels = filter_tensor.dims->data[0];
    const int kernel_height = filter_tensor.dims->data[1];
    const int kernel_width = filter_tensor.dims->data[2];
    const int input_channels = filter_tensor.dims->data[3];

    TF_LITE_ENSURE_STATUS(CheckTransposedConvolutionParams(
        logging_context, deconv_params, node_index));

    std::string auto_pad;
    TF_LITE_ENSURE_STATUS(CalculatePadding(
        logging_context, deconv_params->padding, auto_pad, node_index));

    if (detect_supported_op) {
      TF_LITE_ENSURE_STATUS(CheckWebNNOpSupport(builder, "convTranspose2d"));
    } else {
      emscripten::val options = emscripten::val::object();
      options.set("autoPad", emscripten::val(auto_pad));
      std::vector<int32_t> strides = {
          deconv_params->stride_height, deconv_params->stride_width};
      options.set("strides", emscripten::val::array(strides));
      options.set("inputLayout", emscripten::val("nhwc"));
      options.set("filterLayout", emscripten::val("ohwi"));
      TF_LITE_ENSURE(logging_context, webnn_operands.at(input_tensor_id).as<bool>());
      TF_LITE_ENSURE(logging_context, webnn_operands.at(filter_tensor_id).as<bool>());
      if (bias_tensor_id >= 0) {
        TF_LITE_ENSURE(logging_context, webnn_operands.at(bias_tensor_id).as<bool>());
        options.set("bias", webnn_operands.at(bias_tensor_id));
      }
      emscripten::val output = builder.call<emscripten::val>(
          "convTranspose2d", webnn_operands.at(input_tensor_id),
          webnn_operands.at(filter_tensor_id), options);
      webnn_operands.insert(std::make_pair(output_tensor_id, output));
      TF_LITE_ENSURE(logging_context, webnn_operands.at(output_tensor_id).as<bool>());
    }

    return kTfLiteOk;
  }

  static TfLiteStatus VisitDepthwiseConv2DNode(
      const emscripten::val& builder, TfLiteContext* logging_context,
      int node_index, TfLiteNode* node, const TfLiteTensor* tensors,
      const TfLiteDepthwiseConvParams* dwconv_params,
      const std::unordered_set<int>& quasi_static_tensors,
      std::unordered_map<int, emscripten::val>& webnn_operands,
      const bool detect_supported_op) {
    TF_LITE_ENSURE_STATUS(
        CheckNumInputsAndOutputs(logging_context, node, 3, 1, node_index));

    const int input_tensor_id = node->inputs->data[0];
    const TfLiteTensor& input_tensor = tensors[input_tensor_id];
    TF_LITE_ENSURE_STATUS(CheckTensorFloat32OrQInt8Type(
        logging_context, input_tensor, input_tensor_id, node_index));
    TF_LITE_ENSURE_STATUS(CheckTensorShape(logging_context, input_tensor, 4,
                                           input_tensor_id));
    TF_LITE_ENSURE_STATUS(CheckTensorNonDynamicAllocation(
        logging_context, input_tensor, input_tensor_id, node_index));

    const int filter_tensor_id = node->inputs->data[1];
    const TfLiteTensor& filter_tensor = tensors[filter_tensor_id];
    TF_LITE_ENSURE_STATUS(CheckTensorFloat32OrQInt8Type(
        logging_context, filter_tensor, filter_tensor_id, node_index));
    TF_LITE_ENSURE_STATUS(CheckTensorShape(logging_context, filter_tensor, 4,
                                           filter_tensor_id));
    if (quasi_static_tensors.count(filter_tensor_id) == 0) {
      TF_LITE_ENSURE_STATUS(CheckTensorStaticAllocation(
          logging_context, filter_tensor, filter_tensor_id, node_index));
    }

    const int bias_tensor_id = node->inputs->data[2];
    // bias_tensor_id < 0 means without bias.
    if (bias_tensor_id >= 0) {
      const TfLiteTensor& bias_tensor = tensors[bias_tensor_id];
      TF_LITE_ENSURE_STATUS(CheckTensorFloat32OrQInt32Type(
          logging_context, bias_tensor, node->inputs->data[2], node_index));
      TF_LITE_ENSURE_STATUS(CheckTensorShape(logging_context, bias_tensor, 1,
                                            node->inputs->data[2]));
      if (quasi_static_tensors.count(node->inputs->data[2]) == 0) {
        TF_LITE_ENSURE_STATUS(CheckTensorStaticAllocation(
            logging_context, bias_tensor, node->inputs->data[2], node_index));
      }
    }

    const int output_tensor_id = node->outputs->data[0];
    const TfLiteTensor& output_tensor = tensors[output_tensor_id];
    TF_LITE_ENSURE_STATUS(CheckTensorFloat32OrQInt8Type(
        logging_context, output_tensor, output_tensor_id, node_index));
    TF_LITE_ENSURE_STATUS(CheckTensorShape(logging_context, output_tensor, 4,
                                           output_tensor_id));
    TF_LITE_ENSURE_STATUS(CheckTensorNonDynamicAllocation(
        logging_context, output_tensor, output_tensor_id, node_index));

    const int output_channels = filter_tensor.dims->data[3];
    TF_LITE_ENSURE_STATUS(CheckDepthwiseConvolutionParams(
        logging_context, dwconv_params, output_channels, node_index));

    std::string auto_pad;
    TF_LITE_ENSURE_STATUS(CalculatePadding(
        logging_context, dwconv_params->padding, auto_pad, node_index));

    if (detect_supported_op) {
      TF_LITE_ENSURE_STATUS(CheckWebNNOpSupport(builder, "conv2d"));
      TF_LITE_ENSURE_STATUS(CheckActivation(
          builder, node_index, dwconv_params->activation));
    } else {
      std::vector<int32_t> strides = {
          dwconv_params->stride_height, dwconv_params->stride_width};
      std::vector<int32_t> dilations = {
          dwconv_params->dilation_height_factor, dwconv_params->dilation_width_factor};

      emscripten::val options = emscripten::val::object();
      options.set("autoPad", emscripten::val(auto_pad));
      options.set("strides", emscripten::val::array(strides));
      options.set("dilations", emscripten::val::array(dilations));
      options.set("inputLayout", emscripten::val("nhwc"));
      const auto groups = output_channels / dwconv_params->depth_multiplier;
      options.set("groups", groups);
      if (groups != 1) {
        options.set("filterLayout", emscripten::val("ihwo"));
      } else {
        // groups == 1 is a normal conv2d in WebNN.
        // TODO: Consider depth_multipiler != 1 once the spec issue:
        // https://github.com/webmachinelearning/webnn/issues/353 fixed.
        options.set("filterLayout", emscripten::val("ohwi"));
      }
      TF_LITE_ENSURE(logging_context, webnn_operands.at(input_tensor_id).as<bool>());
      TF_LITE_ENSURE(logging_context, webnn_operands.at(filter_tensor_id).as<bool>());
      if (bias_tensor_id >= 0) {
        TF_LITE_ENSURE(logging_context, webnn_operands.at(bias_tensor_id).as<bool>());
        options.set("bias", webnn_operands.at(bias_tensor_id));
      }

      if (dwconv_params->activation != kTfLiteActNone) {
        emscripten::val activation_operator = GetActivation(
            builder, dwconv_params->activation);
        options.set("activation", activation_operator);
      }
      emscripten::val output = builder.call<emscripten::val>(
          "conv2d", webnn_operands.at(input_tensor_id),
          webnn_operands.at(filter_tensor_id), options);
      webnn_operands.insert(std::make_pair(output_tensor_id, output));
      TF_LITE_ENSURE(logging_context, webnn_operands.at(output_tensor_id).as<bool>());
    }

    return kTfLiteOk;
  }

  static TfLiteStatus VisitDivNode(
      const emscripten::val& builder, TfLiteContext* logging_context,
      int node_index, TfLiteNode* node, const TfLiteTensor* tensors,
      const TfLiteDivParams* div_params,
      std::unordered_map<int, emscripten::val>& webnn_operands,
      const bool detect_supported_op) {
    TF_LITE_ENSURE_STATUS(
        CheckNumInputsAndOutputs(logging_context, node, 2, 1, node_index));

    const int input1_tensor_id = node->inputs->data[0];
    const TfLiteTensor& input1_tensor = tensors[input1_tensor_id];
    TF_LITE_ENSURE_STATUS(CheckTensorFloat32OrQInt8Type(
        logging_context, input1_tensor, input1_tensor_id, node_index));
    TF_LITE_ENSURE_STATUS(CheckTensorNonDynamicAllocation(
        logging_context, input1_tensor, input1_tensor_id, node_index));

    const int input2_tensor_id = node->inputs->data[1];
    const TfLiteTensor& input2_tensor = tensors[input2_tensor_id];
    TF_LITE_ENSURE_STATUS(CheckTensorFloat32OrQInt8Type(
        logging_context, input2_tensor, input2_tensor_id, node_index));
    TF_LITE_ENSURE_STATUS(CheckTensorNonDynamicAllocation(
        logging_context, input2_tensor, input2_tensor_id, node_index));

    const int output_tensor_id = node->outputs->data[0];
    const TfLiteTensor& output_tensor = tensors[output_tensor_id];
    TF_LITE_ENSURE_STATUS(CheckTensorFloat32OrQInt8Type(
        logging_context, output_tensor, output_tensor_id, node_index));
    TF_LITE_ENSURE_STATUS(CheckTensorNonDynamicAllocation(
        logging_context, output_tensor, output_tensor_id, node_index));

    if (detect_supported_op) {
      TF_LITE_ENSURE_STATUS(CheckWebNNOpSupport(builder, "div"));
      if (div_params != nullptr) {
        TF_LITE_ENSURE_STATUS(
            CheckActivation(builder, node_index, div_params->activation));
      }
    } else {
      TF_LITE_ENSURE(logging_context, webnn_operands.at(input1_tensor_id).as<bool>());
      TF_LITE_ENSURE(logging_context, webnn_operands.at(input2_tensor_id).as<bool>());
      webnn_operands.insert(std::make_pair(output_tensor_id,
          builder.call<emscripten::val>("div",
                                        webnn_operands.at(input1_tensor_id),
                                        webnn_operands.at(input2_tensor_id))));
      TF_LITE_ENSURE(logging_context, webnn_operands.at(output_tensor_id).as<bool>());

      if (div_params != nullptr) {
        TF_LITE_ENSURE_STATUS(
            VisitActivation(builder, output_tensor_id, output_tensor_id,
                            div_params->activation, webnn_operands));
      }
    }

    return kTfLiteOk;
  }

  static TfLiteStatus VisitEluNode(
      const emscripten::val& builder, TfLiteContext* logging_context,
      int node_index, TfLiteNode* node, const TfLiteTensor* tensors,
      std::unordered_map<int, emscripten::val>& webnn_operands,
      const bool detect_supported_op) {
    TF_LITE_ENSURE_STATUS(
        CheckNumInputsAndOutputs(logging_context, node, 1, 1, node_index));

    const int input_tensor_id = node->inputs->data[0];
    const TfLiteTensor& input_tensor = tensors[input_tensor_id];
    TF_LITE_ENSURE_STATUS(CheckTensorFloat32OrQInt8Type(
        logging_context, input_tensor, input_tensor_id, node_index));
    TF_LITE_ENSURE_STATUS(CheckTensorNonDynamicAllocation(
        logging_context, input_tensor, input_tensor_id, node_index));

    const int output_tensor_id = node->outputs->data[0];
    const TfLiteTensor& output_tensor = tensors[output_tensor_id];
    TF_LITE_ENSURE_STATUS(CheckTensorFloat32OrQInt8Type(
        logging_context, output_tensor, output_tensor_id, node_index));
    TF_LITE_ENSURE_STATUS(CheckTensorNonDynamicAllocation(
        logging_context, output_tensor, output_tensor_id, node_index));

    if (detect_supported_op) {
      TF_LITE_ENSURE_STATUS(CheckWebNNOpSupport(builder, "elu"));
    } else {
      TF_LITE_ENSURE(logging_context, webnn_operands.at(input_tensor_id).as<bool>());
      // alpha = 1.0 in TFLite Elu node, WebNN's MLEluOptions.alpha is default to 1.0,
      // so no need to set it.
      webnn_operands.insert(std::make_pair(
          output_tensor_id, builder.call<emscripten::val>(
                                "elu", webnn_operands.at(input_tensor_id))));
      TF_LITE_ENSURE(logging_context, webnn_operands.at(output_tensor_id).as<bool>());
    }

    return kTfLiteOk;
  }

  static TfLiteStatus VisitFullyConnectedNode(
      const emscripten::val& builder, TfLiteContext* logging_context,
      int node_index, TfLiteNode* node, const TfLiteTensor* tensors,
      const TfLiteFullyConnectedParams* fc_params,
      const std::unordered_set<int>& quasi_static_tensors,
      std::unordered_map<int, emscripten::val>& webnn_operands,
      const bool detect_supported_op) {
    TF_LITE_ENSURE_STATUS(
        CheckFullyConnectedParams(logging_context, fc_params, node_index));
    TF_LITE_ENSURE_STATUS(
        CheckNumInputsAndOutputs(logging_context, node, 2, 3, 1, node_index));

    const int input_tensor_id = node->inputs->data[0];
    const TfLiteTensor& input_tensor = tensors[input_tensor_id];
    TF_LITE_ENSURE_STATUS(CheckTensorFloat32OrQInt8Type(
        logging_context, input_tensor, input_tensor_id, node_index));
    TF_LITE_ENSURE_STATUS(CheckTensorNonDynamicAllocation(
        logging_context, input_tensor, input_tensor_id, node_index));

    const int filter_tensor_id = node->inputs->data[1];
    const TfLiteTensor& filter_tensor = tensors[filter_tensor_id];
    TF_LITE_ENSURE_STATUS(CheckTensorFloat32OrQInt8Type(
        logging_context, filter_tensor, filter_tensor_id, node_index));
    TF_LITE_ENSURE_STATUS(CheckTensorShape(logging_context, filter_tensor, 2,
                                           filter_tensor_id));
    if (quasi_static_tensors.count(filter_tensor_id) == 0) {
      TF_LITE_ENSURE_STATUS(CheckTensorStaticAllocation(
          logging_context, filter_tensor, filter_tensor_id, node_index));
    }

    int bias_tensor_id = -1;
    if (node->inputs->size >= 3) {
      bias_tensor_id = node->inputs->data[2];
      if (bias_tensor_id >= 0) {
        const TfLiteTensor& bias_tensor = tensors[bias_tensor_id];
        TF_LITE_ENSURE_STATUS(CheckTensorFloat32OrQInt32Type(
            logging_context, bias_tensor, bias_tensor_id, node_index));
        TF_LITE_ENSURE_STATUS(CheckTensorShape(logging_context, bias_tensor, 1,
                                               bias_tensor_id));
        if (quasi_static_tensors.count(bias_tensor_id) == 0) {
          TF_LITE_ENSURE_STATUS(CheckTensorStaticAllocation(
              logging_context, bias_tensor, bias_tensor_id, node_index));
        }
      }
    }

    const int output_tensor_id = node->outputs->data[0];
    const TfLiteTensor& output_tensor = tensors[output_tensor_id];
    TF_LITE_ENSURE_STATUS(CheckTensorFloat32OrQInt8Type(
        logging_context, output_tensor, output_tensor_id, node_index));
    TF_LITE_ENSURE_STATUS(CheckTensorNonDynamicAllocation(
        logging_context, output_tensor, output_tensor_id, node_index));

    const int32_t output_channels = filter_tensor.dims->data[0];
    const int32_t input_channels = filter_tensor.dims->data[1];

    if (input_tensor.dims->size == 0) {
      TF_LITE_MAYBE_KERNEL_LOG(
          logging_context,
          "unexpected number of shape dimensions %d in tensor #%d",
          input_tensor.dims->size, input_tensor_id);
      return kTfLiteError;
    }
    const int32_t * input_dims_data = input_tensor.dims->data;
    int32_t num_input_elements = 1;
    for (int i = 0; i < input_tensor.dims->size; i++) {
      if (input_dims_data[i] <= 0) {
        TF_LITE_MAYBE_KERNEL_LOG(
            logging_context, "invalid dimension #%d (%d) in tensor #%d", i,
            input_dims_data[i], input_tensor_id);
        return kTfLiteError;
      }
      num_input_elements *= input_dims_data[i];
    }

    if (fc_params->keep_num_dims) {
      TF_LITE_ENSURE_STATUS(CheckTensorShape(logging_context, output_tensor,
                                             input_tensor.dims->size,
                                             output_tensor_id));

      for (int i = 0; i < input_tensor.dims->size - 1; i++) {
        if (input_dims_data[i] != output_tensor.dims->data[i]) {
          TF_LITE_MAYBE_KERNEL_LOG(
              logging_context,
              "mismatch in shape dimension %d (%d != %d) in input and output "
              "tensors of FULLY_CONNECTED operator #%d",
              i, input_dims_data[i], output_tensor.dims->data[i],
              node_index);
          return kTfLiteError;
        }
      }
    } else {
      if (num_input_elements % input_channels != 0) {
        TF_LITE_MAYBE_KERNEL_LOG(
            logging_context,
            "number of elements in input tensor #%d in FULLY_CONNECTED "
            "operator is not divisible by input channels (%d)",
            input_tensor_id, input_channels);
        return kTfLiteError;
      }

      TF_LITE_ENSURE_STATUS(CheckTensorShape(logging_context, output_tensor, 2,
                                             output_tensor_id));

      if (output_tensor.dims->data[0] != num_input_elements / input_channels) {
        TF_LITE_MAYBE_KERNEL_LOG(
            logging_context,
            "batch size %d in output tensor #%d in FULLY_CONNECTED operator "
            "does not match batch size %d in reshaped input tensor #%d",
            output_tensor.dims->data[0], output_tensor_id,
            num_input_elements / input_channels, input_tensor_id);
        return kTfLiteError;
      }
    }

    if (output_tensor.dims->data[output_tensor.dims->size - 1] !=
        output_channels) {
      TF_LITE_MAYBE_KERNEL_LOG(
          logging_context,
          "number of channels %d in output tensor #%d does not match output "
          "channels %d in filter tensor #%d",
          output_tensor.dims->data[output_tensor.dims->size - 1],
          output_tensor_id, output_channels, filter_tensor_id);
      return kTfLiteError;
    }

    if (detect_supported_op) {
      TF_LITE_ENSURE_STATUS(CheckWebNNOpSupport(builder, "gemm"));
      TF_LITE_ENSURE_STATUS(
          CheckActivation(builder, node_index, fc_params->activation));
      if (fc_params->keep_num_dims || input_tensor.dims->size != 2) {
        TF_LITE_ENSURE_STATUS(CheckWebNNOpSupport(builder, "reshape"));
      }
    } else {
      emscripten::val options = emscripten::val::object();
      options.set("aTranspose", false);
      options.set("bTranspose",  true);
      if (bias_tensor_id >= 0) {
        TF_LITE_ENSURE(logging_context, webnn_operands.at(bias_tensor_id).as<bool>());
        options.set("c", webnn_operands.at(bias_tensor_id));
      }
      TF_LITE_ENSURE(logging_context, webnn_operands.at(input_tensor_id).as<bool>());
      TF_LITE_ENSURE(logging_context, webnn_operands.at(filter_tensor_id).as<bool>());
      if (fc_params->keep_num_dims || input_tensor.dims->size != 2) {
        // Reshape input to 2D tensor
        emscripten::val new_input_shape = emscripten::val::array();
        new_input_shape.call<void>("push", emscripten::val::null());
        new_input_shape.call<void>("push", input_channels);
        emscripten::val reshaped_input = builder.call<emscripten::val>(
            "reshape", webnn_operands.at(input_tensor_id), new_input_shape);
        emscripten::val gemm = builder.call<emscripten::val>(
            "gemm", reshaped_input, webnn_operands.at(filter_tensor_id),
            options);

        std::vector<int> newShape;
        newShape.assign(
            &output_tensor.dims->data[0],
            &output_tensor.dims->data[0] + output_tensor.dims->size);
        webnn_operands.insert(std::make_pair(
            output_tensor_id,
            builder.call<emscripten::val>("reshape", gemm,
                                          emscripten::val::array(newShape))));
      } else {
        webnn_operands.insert(
            std::make_pair(output_tensor_id,
                           builder.call<emscripten::val>(
                               "gemm", webnn_operands.at(input_tensor_id),
                               webnn_operands.at(filter_tensor_id), options)));
      }
      TF_LITE_ENSURE(logging_context, webnn_operands.at(output_tensor_id).as<bool>());

      TF_LITE_ENSURE_STATUS(
          VisitActivation(builder, output_tensor_id, output_tensor_id,
                          fc_params->activation, webnn_operands));
    }

    return kTfLiteOk;
  }

  static TfLiteStatus VisitClampNode(
      const emscripten::val& builder, TfLiteContext* logging_context,
      int node_index, TfLiteNode* node, const TfLiteTensor* tensors,
      std::unordered_map<int, emscripten::val>& webnn_operands,
      const bool detect_supported_op, float min_value, float max_value) {
    TF_LITE_ENSURE_STATUS(
        CheckNumInputsAndOutputs(logging_context, node, 1, 1, node_index));

    const int input_tensor_id = node->inputs->data[0];
    const TfLiteTensor& input_tensor = tensors[input_tensor_id];
    TF_LITE_ENSURE_STATUS(CheckTensorFloat32OrQInt8Type(
        logging_context, input_tensor, input_tensor_id, node_index));
    TF_LITE_ENSURE_STATUS(CheckTensorNonDynamicAllocation(
        logging_context, input_tensor, input_tensor_id, node_index));

    const int output_tensor_id = node->outputs->data[0];
    const TfLiteTensor& output_tensor = tensors[output_tensor_id];
    TF_LITE_ENSURE_STATUS(CheckTensorFloat32OrQInt8Type(
        logging_context, output_tensor, output_tensor_id, node_index));
    TF_LITE_ENSURE_STATUS(CheckTensorNonDynamicAllocation(
        logging_context, output_tensor, output_tensor_id, node_index));

    if (detect_supported_op) {
      TF_LITE_ENSURE_STATUS(CheckWebNNOpSupport(builder, "clamp"));
    } else {
      TF_LITE_ENSURE(logging_context,
                     webnn_operands.at(input_tensor_id).as<bool>());
      webnn_operands.insert(
          std::make_pair(output_tensor_id,
                         BuildClamp(builder, webnn_operands.at(input_tensor_id),
                                    min_value, max_value)));
      TF_LITE_ENSURE(logging_context,
                     webnn_operands.at(output_tensor_id).as<bool>());
    }

    return kTfLiteOk;
  }

  static TfLiteStatus VisitReshapeNode(
      const emscripten::val& builder, TfLiteContext* logging_context,
      int node_index, TfLiteNode* node, const TfLiteTensor* tensors,
      const TfLiteReshapeParams* reshape_params,
      std::unordered_map<int, emscripten::val>& webnn_operands,
      const bool detect_supported_op) {
    switch (node->inputs->size) {
      case 1:
      case 2:
        break;
      default:
        TF_LITE_MAYBE_KERNEL_LOG(
            logging_context,
            "unexpected number of inputs (%d) in node #%d: "
            "either one or two inputs expected",
            node->inputs->size, node_index);
        return kTfLiteError;
    }
    if (node->outputs->size != 1) {
      TF_LITE_MAYBE_KERNEL_LOG(
          logging_context,
          "unexpected number of outputs (%d) in node #%d: one output expected",
          node->outputs->size, node_index);
      return kTfLiteError;
    }

    const int input_tensor_id = node->inputs->data[0];
    const TfLiteTensor& input_tensor = tensors[input_tensor_id];
    TF_LITE_ENSURE_STATUS(CheckTensorFloat32Type(
        logging_context, input_tensor, input_tensor_id, node_index));
    TF_LITE_ENSURE_STATUS(CheckTensorNonDynamicAllocation(
        logging_context, input_tensor, input_tensor_id, node_index));

    if (node->inputs->size == 2) {
      const int shape_tensor_id = node->inputs->data[1];
      const TfLiteTensor& shape_tensor = tensors[shape_tensor_id];
      TF_LITE_ENSURE_STATUS(CheckTensorType(logging_context, shape_tensor,
                                            kTfLiteInt32, shape_tensor_id,
                                            node_index));
      TF_LITE_ENSURE_STATUS(CheckShapeTensorShape(
          logging_context, shape_tensor, shape_tensor_id, node_index));
      TF_LITE_ENSURE_STATUS(CheckTensorStaticAllocation(
          logging_context, shape_tensor,shape_tensor_id, node_index));
    }

    const int output_tensor_id = node->outputs->data[0];
    const TfLiteTensor& output_tensor = tensors[output_tensor_id];
    TF_LITE_ENSURE_STATUS(CheckTensorFloat32Type(
        logging_context, output_tensor, output_tensor_id, node_index));
    TF_LITE_ENSURE_STATUS(CheckTensorNonDynamicAllocation(
        logging_context, output_tensor, output_tensor_id, node_index));

    if (detect_supported_op) {
      TF_LITE_ENSURE_STATUS(CheckWebNNOpSupport(builder, "reshape"));
    } else {
      TF_LITE_ENSURE(logging_context, webnn_operands.at(input_tensor_id).as<bool>());
      std::vector<int> newShape;
      newShape.assign(&output_tensor.dims->data[0],
                      &output_tensor.dims->data[0] + output_tensor.dims->size);
      webnn_operands.insert(std::make_pair(
          output_tensor_id, builder.call<emscripten::val>(
                                "reshape", webnn_operands.at(input_tensor_id),
                                emscripten::val::array(newShape))));
      TF_LITE_ENSURE(logging_context, webnn_operands.at(output_tensor_id).as<bool>());
    }

    return kTfLiteOk;
  }

  static TfLiteStatus VisitResizeBilinearNode(
      const emscripten::val& builder, TfLiteContext* logging_context,
      int node_index, TfLiteNode* node, const TfLiteTensor* tensors,
      const TfLiteResizeBilinearParams* resize_params,
      std::unordered_map<int, emscripten::val>& webnn_operands,
      const bool detect_supported_op) {
    TF_LITE_ENSURE_STATUS(
        CheckNumInputsAndOutputs(logging_context, node, 2, 1, node_index));

    const int input_tensor_id = node->inputs->data[0];
    const TfLiteTensor& input_tensor = tensors[input_tensor_id];
    TF_LITE_ENSURE_STATUS(CheckTensorFloat32Type(
        logging_context, input_tensor, input_tensor_id, node_index));
    TF_LITE_ENSURE_STATUS(CheckTensorShape(logging_context, input_tensor, 4,
                                           input_tensor_id));
    TF_LITE_ENSURE_STATUS(CheckTensorNonDynamicAllocation(
        logging_context, input_tensor, input_tensor_id, node_index));

    const int shape_tensor_id = node->inputs->data[1];
    const TfLiteTensor& shape_tensor = tensors[shape_tensor_id];
    TF_LITE_ENSURE_STATUS(CheckTensorType(logging_context, shape_tensor,
                                          kTfLiteInt32, shape_tensor_id,
                                          node_index));
    TF_LITE_ENSURE_STATUS(CheckShapeTensorShape(
        logging_context, shape_tensor, shape_tensor_id, node_index));
    if (shape_tensor.dims->data[0] != 2) {
      TF_LITE_MAYBE_KERNEL_LOG(
          logging_context,
          "unexpected number of dimensions %d in the output shape in node %d",
          shape_tensor.dims->data[0], node_index);
    }
    TF_LITE_ENSURE_STATUS(CheckTensorStaticAllocation(
        logging_context, shape_tensor, shape_tensor_id, node_index));

    const int output_tensor_id = node->outputs->data[0];
    const TfLiteTensor& output_tensor = tensors[output_tensor_id];
    TF_LITE_ENSURE_STATUS(CheckTensorFloat32Type(
        logging_context, output_tensor, output_tensor_id, node_index));
    TF_LITE_ENSURE_STATUS(CheckTensorShape(logging_context, output_tensor, 4,
                                           output_tensor_id));
    TF_LITE_ENSURE_STATUS(CheckTensorNonDynamicAllocation(
        logging_context, output_tensor, output_tensor_id, node_index));

    const int32_t* shape_data =
        reinterpret_cast<const int32_t*>(shape_tensor.data.data);
    for (int i = 0; i < shape_tensor.dims->data[0]; i++) {
      const int32_t dim = shape_data[i];
      if (dim <= 0) {
        TF_LITE_MAYBE_KERNEL_LOG(
            logging_context, "invalid output dimension #%d value %d in node %d",
            i, dim, node_index);
        return kTfLiteError;
      }
    }

    if (detect_supported_op) {
      TF_LITE_ENSURE_STATUS(CheckWebNNOpSupport(builder, "resample2d"));
    } else {
      TF_LITE_ENSURE(logging_context, webnn_operands.at(input_tensor_id).as<bool>());
      std::vector<int32_t> sizes = {shape_data[0], shape_data[1]};
      std::vector<int32_t> axes = {1, 2};
      emscripten::val options = emscripten::val::object();
      options.set("mode", emscripten::val("linear"));
      options.set("sizes", emscripten::val::array(sizes));
      options.set("axes", emscripten::val::array(axes));

      webnn_operands.insert(std::make_pair(
          output_tensor_id,
          builder.call<emscripten::val>(
              "resample2d", webnn_operands.at(input_tensor_id), options)));
      TF_LITE_ENSURE(logging_context, webnn_operands.at(output_tensor_id).as<bool>());
    }

    return kTfLiteOk;
  }

  static TfLiteStatus VisitSliceNode(
      const emscripten::val& builder, TfLiteContext* logging_context,
      int node_index, TfLiteNode* node, const TfLiteTensor* tensors,
      std::unordered_map<int, emscripten::val>& webnn_operands,
      const bool detect_supported_op) {
    const int input_tensor_id = node->inputs->data[0];
    const int begin_tensor_id = node->inputs->data[1];
    const int size_tensor_id = node->inputs->data[2];
    const int output_tensor_id = node->outputs->data[0];
    const TfLiteTensor& input_tensor = tensors[input_tensor_id];
    const TfLiteTensor& begin_tensor = tensors[begin_tensor_id];
    const TfLiteTensor& size_tensor = tensors[size_tensor_id];
    const TfLiteTensor& output_tensor = tensors[output_tensor_id];

    TF_LITE_ENSURE_STATUS(CheckShapeTensorShape(logging_context, begin_tensor,
                                                begin_tensor_id, node_index));
    TF_LITE_ENSURE_STATUS(CheckTensorStaticAllocation(
        logging_context, begin_tensor, begin_tensor_id, node_index));
    TF_LITE_ENSURE_STATUS(CheckTensorInt32OrInt64Type(
        logging_context, begin_tensor, begin_tensor_id, node_index));

    TF_LITE_ENSURE_STATUS(CheckShapeTensorShape(logging_context, size_tensor,
                                                size_tensor_id, node_index));
    TF_LITE_ENSURE_STATUS(CheckTensorStaticAllocation(
        logging_context, size_tensor, size_tensor_id, node_index));
    TF_LITE_ENSURE_STATUS(CheckTensorInt32OrInt64Type(
        logging_context, size_tensor, size_tensor_id, node_index));
    TF_LITE_ENSURE_STATUS(CheckTensorsDimensionMatch(
        logging_context, begin_tensor, size_tensor, 0, node_index, "SLICE"));

    const int num_dims = begin_tensor.dims->data[0];

    TF_LITE_ENSURE_STATUS(CheckTensorFloat32Type(logging_context, input_tensor,
                                                 input_tensor_id, node_index));
    TF_LITE_ENSURE_STATUS(CheckTensorShape(logging_context, input_tensor,
                                           num_dims, input_tensor_id));
    TF_LITE_ENSURE_STATUS(CheckTensorNonDynamicAllocation(
        logging_context, input_tensor, input_tensor_id, node_index));

    TF_LITE_ENSURE_STATUS(CheckTensorFloat32Type(logging_context, output_tensor,
                                                 output_tensor_id, node_index));
    TF_LITE_ENSURE_STATUS(CheckTensorShape(logging_context, output_tensor,
                                           num_dims, output_tensor_id));
    TF_LITE_ENSURE_STATUS(CheckTensorNonDynamicAllocation(
        logging_context, output_tensor, output_tensor_id, node_index));

    const auto input_shape = input_tensor.dims;
    const auto output_shape = output_tensor.dims;
    std::vector<int64_t> begin(num_dims);
    std::vector<int64_t> size(num_dims);
    CopyTensorDataInt32OrInt64(begin.data(), begin_tensor, num_dims);
    CopyTensorDataInt32OrInt64(size.data(), size_tensor, num_dims);

    for (size_t i = 0; i < num_dims; i++) {
      if (begin[i] < 0) {
        TF_LITE_MAYBE_KERNEL_LOG(logging_context,
                                 "begin %" PRId64
                                 " must be greater than 0 in SLICE node #%d",
                                 begin[i], node_index);
      }
      if (begin[i] >= input_shape->data[i]) {
        TF_LITE_MAYBE_KERNEL_LOG(
            logging_context,
            "begin %" PRId64
            " must be less than input dimension %d in SLICE node #%d",
            begin[i], input_shape->data[i], node_index);
      }
      if (size[i] <= 0) {
        if (size[i] != -1) {
          TF_LITE_MAYBE_KERNEL_LOG(logging_context,
                                   "size %" PRId64
                                   " must be positive or -1 in SLICE node #%d",
                                   size[i], node_index);
          return kTfLiteError;
        }
        size[i] = input_shape->data[i] - begin[i];
      }
      if (size[i] > input_shape->data[i]) {
        TF_LITE_MAYBE_KERNEL_LOG(logging_context,
                                 "size %" PRId64
                                 " must be less than or equals to input "
                                 "dimension %d in SLICE node #%d",
                                 size[i], input_shape->data[i], node_index);
        return kTfLiteError;
      }
      if (size[i] != output_shape->data[i]) {
        TF_LITE_MAYBE_KERNEL_LOG(logging_context,
                                 "size %" PRId64
                                 " does not match output shape %d at "
                                 "dimension %d in SLICE node #%d",
                                 size[i], output_shape->data[i], i, node_index);
        return kTfLiteError;
      }
      if (begin[i] + size[i] > input_shape->data[i]) {
        TF_LITE_MAYBE_KERNEL_LOG(logging_context,
                                 "begin + size (%" PRId64 " + %" PRId64
                                 ") must not be greater than input "
                                 "dimension %d in SLICE node #%d",
                                 begin[i], size[i], input_shape->data[i],
                                 node_index);
        return kTfLiteError;
      }
    }

    if (detect_supported_op) {
      TF_LITE_ENSURE_STATUS(CheckWebNNOpSupport(builder, "slice"));
    } else {
      TF_LITE_ENSURE(logging_context,
                     webnn_operands.at(input_tensor_id).as<bool>());
      std::vector<int32_t> starts;
      std::vector<int32_t> sizes;
      std::transform(begin.cbegin(), begin.cend(), std::back_inserter(starts),
                     [](int64_t i) { return static_cast<int32_t>(i); });
      std::transform(size.cbegin(), size.cend(), std::back_inserter(sizes),
                     [](int64_t i) { return static_cast<int32_t>(i); });
      webnn_operands.insert(std::make_pair(
          output_tensor_id, builder.call<emscripten::val>(
                                "slice", webnn_operands.at(input_tensor_id),
                                emscripten::val::array(starts),
                                emscripten::val::array(sizes))));
      TF_LITE_ENSURE(logging_context,
                     webnn_operands.at(output_tensor_id).as<bool>());
    }

    return kTfLiteOk;
  }

  static TfLiteStatus VisitSoftmaxNode(
      const emscripten::val& builder, TfLiteContext* logging_context,
      int node_index, TfLiteNode* node, const TfLiteTensor* tensors,
      const TfLiteSoftmaxParams* params,
      std::unordered_map<int, emscripten::val>& webnn_operands,
      const bool detect_supported_op) {
    if (params->beta != 1.0f) {
      if (logging_context != nullptr) {
        TF_LITE_KERNEL_LOG(logging_context,
                           "unsupported beta value %.7f in SOFTMAX node #%d",
                           params->beta, node_index);
      }
      return kTfLiteError;
    }

    TF_LITE_ENSURE_STATUS(
        CheckNumInputsAndOutputs(logging_context, node, 1, 1, node_index));

    const int input_tensor_id = node->inputs->data[0];
    const TfLiteTensor& input_tensor = tensors[input_tensor_id];
    TF_LITE_ENSURE_STATUS(CheckTensorFloat32Type(
        logging_context, input_tensor, input_tensor_id, node_index));
    TF_LITE_ENSURE_STATUS(CheckTensorNonDynamicAllocation(
        logging_context, input_tensor, input_tensor_id, node_index));

    const int output_tensor_id = node->outputs->data[0];
    const TfLiteTensor& output_tensor = tensors[output_tensor_id];
    TF_LITE_ENSURE_STATUS(CheckTensorFloat32Type(
        logging_context, output_tensor, output_tensor_id, node_index));
    TF_LITE_ENSURE_STATUS(CheckTensorNonDynamicAllocation(
        logging_context, output_tensor, output_tensor_id, node_index));

    std::vector<int32_t> input_shape;
    input_shape.assign(&input_tensor.dims->data[0],
                       &input_tensor.dims->data[0] + input_tensor.dims->size);
    if (detect_supported_op) {
      TF_LITE_ENSURE_STATUS(CheckWebNNOpSupport(builder, "softmax"));
    } else {
      TF_LITE_ENSURE(logging_context,
                     webnn_operands.at(input_tensor_id).as<bool>());
      emscripten::val output = emscripten::val::object();
      // WebNN Softmax only support 2d input shape, reshape input to 2d.
      if (input_shape.size() != 2) {
        emscripten::val new_shape = emscripten::val::array();
        // TODO: Remove 'null' support and compute the dimension size in
        // delegate. https://github.com/webmachinelearning/webnn/issues/388
        new_shape.call<void>("push", emscripten::val::null());
        new_shape.call<void>("push", input_shape.back());

        emscripten::val reshape = builder.call<emscripten::val>(
            "reshape", webnn_operands.at(input_tensor_id), new_shape);
        emscripten::val softmax =
            builder.call<emscripten::val>("softmax", reshape);
        // Reshape the output to the same shape of input.
        output = builder.call<emscripten::val>(
            "reshape", softmax, emscripten::val::array(input_shape));
      } else {
        output = builder.call<emscripten::val>(
            "softmax", webnn_operands.at(input_tensor_id));
      }

      webnn_operands.insert(std::make_pair(output_tensor_id, output));
      TF_LITE_ENSURE(logging_context,
                     webnn_operands.at(output_tensor_id).as<bool>());
    }

    return kTfLiteOk;
  }

  static TfLiteStatus VisitSplitNode(
      const emscripten::val& builder, TfLiteContext* logging_context,
      int node_index, TfLiteNode* node, const TfLiteTensor* tensors,
      const TfLiteSplitParams* params,
      std::unordered_map<int, emscripten::val>& webnn_operands,
      const bool detect_supported_op) {
    const int num_splits = params->num_splits;
    if (num_splits == 0) {
      TF_LITE_MAYBE_KERNEL_LOG(
          logging_context,
          "unexpected value of num_splits %d in the split params in node %d",
          num_splits, node_index);
      return kTfLiteError;
    }

    TF_LITE_ENSURE_STATUS(
        CheckNumInputsAndOutputs(logging_context, node, 2, num_splits, node_index));

    const int input_tensor_id = node->inputs->data[1];
    const TfLiteTensor& input_tensor = tensors[input_tensor_id];
    TF_LITE_ENSURE_STATUS(CheckTensorFloat32OrQInt8Type(
        logging_context, input_tensor, input_tensor_id, node_index));
    TF_LITE_ENSURE_STATUS(CheckTensorNonDynamicAllocation(
        logging_context, input_tensor, input_tensor_id, node_index));

    const int axis_tensor_id = node->inputs->data[0];
    const TfLiteTensor& axis_tensor = tensors[axis_tensor_id];
    TF_LITE_ENSURE_STATUS(CheckTensorType(logging_context, axis_tensor,
                                          kTfLiteInt32, axis_tensor_id,
                                          node_index));
    TF_LITE_ENSURE_STATUS(CheckTensorShape(logging_context, axis_tensor, 0,
                                           axis_tensor_id));
    TF_LITE_ENSURE_STATUS(CheckTensorStaticAllocation(
        logging_context, axis_tensor, axis_tensor_id, node_index));

    const int* axis_data =
        reinterpret_cast<const int*>(axis_tensor.data.data);
    int axis_value = axis_data[0];

    const int num_dims = input_tensor.dims->size;
    if (axis_value < 0) {
      axis_value += num_dims;
    }
    if (axis_value < 0 || axis_value > num_dims) {
      TF_LITE_MAYBE_KERNEL_LOG(
          logging_context,
          "unexpected data of axis %d in the axis tensor in node %d",
          axis_data[0], node_index);
      return kTfLiteError;
    }
    const int input_size = input_tensor.dims->data[axis_value];
    if (input_size % num_splits != 0) {
      TF_LITE_MAYBE_KERNEL_LOG(
          logging_context,
          "Not an even split");
      return kTfLiteError;
    }

    const int output_size = node->outputs->size;

    if (detect_supported_op) {
      TF_LITE_ENSURE_STATUS(CheckWebNNOpSupport(builder, "split"));
    } else {
      std::vector<uint32_t> splits = {static_cast<const uint32_t>(num_splits)};
      emscripten::val options = emscripten::val::object();
      options.set("axis", axis_value);
      TF_LITE_ENSURE(logging_context, webnn_operands.at(input_tensor_id).as<bool>());
      emscripten::val split_operand_array = builder.call<emscripten::val>(
          "split", webnn_operands.at(input_tensor_id),
          emscripten::val::array(splits), options);
      TF_LITE_ENSURE(
          logging_context,
          split_operand_array["length"].as<int32_t>() == output_size);

      for (int i = 0; i < output_size; i++) {
        int output_tensor_id = node->outputs->data[i];
        const TfLiteTensor& output_tensor = tensors[output_tensor_id];
        TF_LITE_ENSURE_STATUS(CheckTensorFloat32Type(
            logging_context, output_tensor, output_tensor_id, node_index));
        TF_LITE_ENSURE_STATUS(CheckTensorNonDynamicAllocation(
            logging_context, output_tensor, output_tensor_id, node_index));
        webnn_operands.insert(
            std::make_pair(output_tensor_id, split_operand_array[i]));
        TF_LITE_ENSURE(logging_context, webnn_operands.at(output_tensor_id).as<bool>());
      }
    }

    return kTfLiteOk;
  }

  static TfLiteStatus VisitStridedSliceNode(
      const emscripten::val& builder, TfLiteContext* logging_context,
      int node_index, TfLiteNode* node, const TfLiteTensor* tensors,
      const TfLiteStridedSliceParams* params,
      std::unordered_map<int, emscripten::val>& webnn_operands,
      const bool detect_supported_op) {
    // Only support strided slice with no ellipsis mask, no new axis mask, and
    // no shrink_axis-mask.
    if (params->ellipsis_mask != 0 || params->new_axis_mask != 0 ||
        params->shrink_axis_mask != 0) {
      return kTfLiteError;
    }

    const int stride_tensor_id = node->inputs->data[3];
    const TfLiteTensor& stride_tensor = tensors[stride_tensor_id];

    TF_LITE_ENSURE_STATUS(CheckShapeTensorShape(logging_context, stride_tensor,
                                                stride_tensor_id, node_index));
    TF_LITE_ENSURE_STATUS(CheckTensorStaticAllocation(
        logging_context, stride_tensor, stride_tensor_id, node_index));
    TF_LITE_ENSURE_STATUS(CheckTensorInt32Type(
        logging_context, stride_tensor, stride_tensor_id, node_index));

    const int num_dims = stride_tensor.dims->data[0];

    // Only support strides = 1.
    auto stride_data = GetTensorData<int32_t>(&stride_tensor);
    for (size_t i = 0; i < num_dims; i++) {
      if (stride_data[i] != 1) {
        TF_LITE_MAYBE_KERNEL_LOG(logging_context,
                                 "stride at dimension %d, %d, must be 1"
                                 "in STRIDED_SLICE node #%d",
                                 i, stride_data[i], node_index);
        return kTfLiteError;
      }
    }

    const int input_tensor_id = node->inputs->data[0];
    const int begin_tensor_id = node->inputs->data[1];
    const int end_tensor_id = node->inputs->data[2];
    const int output_tensor_id = node->outputs->data[0];
    const TfLiteTensor& input_tensor = tensors[input_tensor_id];
    const TfLiteTensor& begin_tensor = tensors[begin_tensor_id];
    const TfLiteTensor& end_tensor = tensors[end_tensor_id];
    const TfLiteTensor& output_tensor = tensors[output_tensor_id];

    TF_LITE_ENSURE_STATUS(CheckShapeTensorShape(logging_context, begin_tensor,
                                                begin_tensor_id, node_index));
    TF_LITE_ENSURE_STATUS(CheckTensorStaticAllocation(
        logging_context, begin_tensor, begin_tensor_id, node_index));
    // TODO: TFLite only supports int32 begin ends and strides,
    // support int64 too when TFLite supports it as well.
    // Refer to https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/delegates/xnnpack/xnnpack_delegate.cc#L5404
    TF_LITE_ENSURE_STATUS(CheckTensorInt32Type(logging_context, begin_tensor,
                                               begin_tensor_id, node_index));

    TF_LITE_ENSURE_STATUS(CheckShapeTensorShape(logging_context, end_tensor,
                                                end_tensor_id, node_index));
    TF_LITE_ENSURE_STATUS(CheckTensorStaticAllocation(
        logging_context, end_tensor, end_tensor_id, node_index));
    TF_LITE_ENSURE_STATUS(CheckTensorInt32Type(logging_context, end_tensor,
                                               end_tensor_id, node_index));

    TF_LITE_ENSURE_STATUS(
        CheckTensorsDimensionMatch(logging_context, stride_tensor, begin_tensor,
                                   0, node_index, "STRIDED_SLICE"));
    TF_LITE_ENSURE_STATUS(
        CheckTensorsDimensionMatch(logging_context, begin_tensor, end_tensor, 0,
                                   node_index, "STRIDED_SLICE"));
    TF_LITE_ENSURE_STATUS(CheckTensorFloat32Type(logging_context, input_tensor,
                                                 input_tensor_id, node_index));
    TF_LITE_ENSURE_STATUS(CheckTensorShape(logging_context, input_tensor,
                                           num_dims, input_tensor_id));
    TF_LITE_ENSURE_STATUS(CheckTensorNonDynamicAllocation(
        logging_context, input_tensor, input_tensor_id, node_index));

    TF_LITE_ENSURE_STATUS(CheckTensorFloat32Type(logging_context, output_tensor,
                                                 output_tensor_id, node_index));
    TF_LITE_ENSURE_STATUS(CheckTensorShape(logging_context, output_tensor,
                                           num_dims, output_tensor_id));
    TF_LITE_ENSURE_STATUS(CheckTensorNonDynamicAllocation(
        logging_context, output_tensor, output_tensor_id, node_index));

    auto begin_data = GetTensorData<int32_t>(&begin_tensor);
    auto end_data = GetTensorData<int32_t>(&end_tensor);
    auto input_shape = input_tensor.dims;
    std::vector<int32_t> begins(num_dims);
    std::vector<int32_t> sizes(num_dims);
    std::vector<int32_t> ends(num_dims);
    for (size_t i = 0; i < num_dims; i++) {
      begins[i] = begin_data[i] < 0 ? input_shape->data[i] + begin_data[i]
                                    : begin_data[i];
      if ((params->begin_mask & (1 << i)) != 0) {
        begins[i] = 0;
      }

      if (begins[i] >= input_shape->data[i]) {
        TF_LITE_MAYBE_KERNEL_LOG(logging_context,
                                 "begin %zu must be less than input dimension "
                                 "%d in STRIDED_SLICE node #%d",
                                 begins[i], input_shape->data[i], node_index);
      }
      // TODO: Consider offset param after code rebase.
      // Relevant commit: https://github.com/tensorflow/tensorflow/commit/8c9d39a0927bef0e81fdbb7d1e9e376d34ed8266

      // If end is negative, we count from the back, -1 is the last element.
      if (end_data[i] < 0) {
        ends[i] = end_data[i] + input_shape->data[i];
      } else {
        ends[i] = end_data[i];
      }

      if ((params->end_mask & (1 << i)) != 0) {
        ends[i] = input_shape->data[i];
      }

      if (ends[i] > input_shape->data[i]) {
        TF_LITE_MAYBE_KERNEL_LOG(logging_context,
                                 "end %zu must be less than or equals to input "
                                 "dimension %d in STRIDED_SLICE node #%d",
                                 ends[i], input_shape->data[i], node_index);
      }

      if (begins[i] >= ends[i]) {
        TF_LITE_MAYBE_KERNEL_LOG(logging_context,
                                 "begin index %zu must be less than end index "
                                 "%zu for STRIDED_SLICE node #%d",
                                 begins[i], ends[i], node_index);
      }

      sizes[i] = ends[i] - begins[i];
    }

    if (detect_supported_op) {
      TF_LITE_ENSURE_STATUS(CheckWebNNOpSupport(builder, "slice"));
    } else {
      TF_LITE_ENSURE(logging_context,
                     webnn_operands.at(input_tensor_id).as<bool>());

      webnn_operands.insert(std::make_pair(
          output_tensor_id, builder.call<emscripten::val>(
                                "slice", webnn_operands.at(input_tensor_id),
                                emscripten::val::array(begins),
                                emscripten::val::array(sizes))));
      TF_LITE_ENSURE(logging_context,
                     webnn_operands.at(output_tensor_id).as<bool>());
    }

    return kTfLiteOk;
  }

  static TfLiteStatus VisitUnaryNode(
      const emscripten::val& builder, TfLiteContext* logging_context,
      int node_index, TfLiteNode* node, const TfLiteTensor* tensors,
      std::unordered_map<int, emscripten::val>& webnn_operands,
      const bool detect_supported_op, const std::string unary_node_name) {
    TF_LITE_ENSURE_STATUS(
        CheckNumInputsAndOutputs(logging_context, node, 1, 1, node_index));

    const int input_tensor_id = node->inputs->data[0];
    const TfLiteTensor& input_tensor = tensors[input_tensor_id];
    TF_LITE_ENSURE_STATUS(CheckTensorFloat32OrQInt8Type(
        logging_context, input_tensor, input_tensor_id, node_index));
    TF_LITE_ENSURE_STATUS(CheckTensorNonDynamicAllocation(
        logging_context, input_tensor, input_tensor_id, node_index));

    const int output_tensor_id = node->outputs->data[0];
    const TfLiteTensor& output_tensor = tensors[output_tensor_id];
    TF_LITE_ENSURE_STATUS(CheckTensorFloat32OrQInt8Type(
        logging_context, output_tensor, output_tensor_id, node_index));
    TF_LITE_ENSURE_STATUS(CheckTensorNonDynamicAllocation(
        logging_context, output_tensor, output_tensor_id, node_index));

    if (detect_supported_op) {
      TF_LITE_ENSURE_STATUS(CheckWebNNOpSupport(builder, unary_node_name));
    } else {
      TF_LITE_ENSURE(logging_context, webnn_operands.at(input_tensor_id).as<bool>());
      webnn_operands.insert(std::make_pair(output_tensor_id,
          builder.call<emscripten::val>(unary_node_name.c_str(),
                                        webnn_operands.at(input_tensor_id))));
      TF_LITE_ENSURE(logging_context, webnn_operands.at(output_tensor_id).as<bool>());
    }

    return kTfLiteOk;
  }

  static TfLiteStatus VisitTransposeNode(
      const emscripten::val& builder, TfLiteContext* logging_context,
      int node_index, TfLiteNode* node, const TfLiteTensor* tensors,
      std::unordered_map<int, emscripten::val>& webnn_operands,
      const bool detect_supported_op) {
    TF_LITE_ENSURE_STATUS(
        CheckNumInputsAndOutputs(logging_context, node, 2, 1, node_index));

    const int input_tensor_id = node->inputs->data[0];
    const TfLiteTensor& input_tensor = tensors[input_tensor_id];
    TF_LITE_ENSURE_STATUS(CheckTensorNonDynamicAllocation(
        logging_context, input_tensor, input_tensor_id, node_index));

    const TfLiteTensor& perm_tensor = tensors[node->inputs->data[1]];
    TF_LITE_ENSURE_STATUS(CheckTensorStaticAllocation(
        logging_context, perm_tensor, node->inputs->data[1], node_index));

    const int32_t* perm_data =
        reinterpret_cast<const int32_t*>(perm_tensor.data.data);
    const int dims_count = NumDimensions(&input_tensor);
    std::vector<uint32_t> perm;
    for (int i = 0; i < dims_count; i++) {
      if (perm_data[i] < 0) {
        TF_LITE_MAYBE_KERNEL_LOG(logging_context, "invalid perm data %d in node %d",
                                 perm_data[i], node_index);
        return kTfLiteError;
      }
      perm.push_back(static_cast<uint32_t>(perm_data[i]));
    }

    const int output_tensor_id = node->outputs->data[0];
    const TfLiteTensor& output_tensor = tensors[output_tensor_id];
    TF_LITE_ENSURE_STATUS(CheckTensorNonDynamicAllocation(
        logging_context, output_tensor, output_tensor_id, node_index));

    if (detect_supported_op) {
      TF_LITE_ENSURE_STATUS(CheckWebNNOpSupport(builder, "transpose"));
    } else {
      TF_LITE_ENSURE(logging_context,
                     webnn_operands.at(input_tensor_id).as<bool>());
      emscripten::val options = emscripten::val::object();
      options.set("permutation", emscripten::val::array(perm));
      webnn_operands.insert(std::make_pair(
          output_tensor_id,
          builder.call<emscripten::val>(
              "transpose", webnn_operands.at(input_tensor_id), options)));
      TF_LITE_ENSURE(logging_context,
                     webnn_operands.at(output_tensor_id).as<bool>());
    }

    return kTfLiteOk;
  }

  static TfLiteStatus VisitTransposeConvNode(
      const emscripten::val& builder, TfLiteContext* logging_context,
      int node_index, TfLiteNode* node, const TfLiteTensor* tensors,
      const TfLiteTransposeConvParams* deconv_params,
      const std::unordered_set<int>& quasi_static_tensors,
      std::unordered_map<int, emscripten::val>& webnn_operands,
      const bool detect_supported_op) {
    TF_LITE_ENSURE_STATUS(
        CheckNumInputsAndOutputs(logging_context, node,
                                 /*min_num_inputs=*/3, /*max_num_inputs=*/4,
                                 /*expected_num_outputs=*/1, node_index));
    const bool use_bias = node->inputs->size >= 4;

    const int output_shape_tensor_id = node->inputs->data[0];
    const TfLiteTensor& output_shape_tensor = tensors[output_shape_tensor_id];
    TF_LITE_ENSURE_STATUS(CheckTensorType(logging_context, output_shape_tensor,
                                          kTfLiteInt32, output_shape_tensor_id,
                                          node_index));
    TF_LITE_ENSURE_STATUS(
        CheckShapeTensorShape(logging_context, output_shape_tensor,
                              output_shape_tensor_id, node_index));
    TF_LITE_ENSURE_STATUS(
        CheckTensorStaticAllocation(logging_context, output_shape_tensor,
                                    output_shape_tensor_id, node_index));
    const int output_shape_dims = SizeOfDimension(&output_shape_tensor, 0);
    if (output_shape_dims != 4) {
      TF_LITE_MAYBE_KERNEL_LOG(
          logging_context,
          "unsupported number of output shape dimensions (%d) in node #%d: "
          "4 dimensions expected",
          output_shape_dims, node_index);
      return kTfLiteError;
    }

    const int filter_tensor_id = node->inputs->data[1];
    const TfLiteTensor& filter_tensor = tensors[filter_tensor_id];
    TF_LITE_ENSURE_STATUS(CheckTensorFloat32Type(
        logging_context, filter_tensor, filter_tensor_id, node_index));
    TF_LITE_ENSURE_STATUS(
        CheckTensorShape(logging_context, filter_tensor, 4, filter_tensor_id));
    if (quasi_static_tensors.count(filter_tensor_id) == 0) {
      TF_LITE_ENSURE_STATUS(CheckTensorStaticAllocation(
          logging_context, filter_tensor, filter_tensor_id, node_index));
    }

    const int input_tensor_id = node->inputs->data[2];
    const TfLiteTensor& input_tensor = tensors[input_tensor_id];
    TF_LITE_ENSURE_STATUS(CheckTensorFloat32Type(
        logging_context, input_tensor, input_tensor_id, node_index));
    TF_LITE_ENSURE_STATUS(
        CheckTensorShape(logging_context, input_tensor, 4, input_tensor_id));
    TF_LITE_ENSURE_STATUS(CheckTensorNonDynamicAllocation(
        logging_context, input_tensor, input_tensor_id, node_index));

    if (use_bias) {
      const int bias_tensor_id = node->inputs->data[3];
      if (bias_tensor_id != kTfLiteOptionalTensor) {
        const TfLiteTensor& bias_tensor = tensors[bias_tensor_id];
        TF_LITE_ENSURE_STATUS(CheckTensorFloat32Type(
            logging_context, bias_tensor, bias_tensor_id, node_index));
        TF_LITE_ENSURE_STATUS(
            CheckTensorShape(logging_context, bias_tensor, 1, bias_tensor_id));
        if (quasi_static_tensors.count(bias_tensor_id) == 0) {
          TF_LITE_ENSURE_STATUS(CheckTensorStaticAllocation(
              logging_context, bias_tensor, bias_tensor_id, node_index));
        }
      }
    }

    const int output_tensor_id = node->outputs->data[0];
    const TfLiteTensor& output_tensor = tensors[output_tensor_id];
    TF_LITE_ENSURE_STATUS(CheckTensorFloat32Type(
        logging_context, output_tensor, output_tensor_id, node_index));
    TF_LITE_ENSURE_STATUS(
        CheckTensorShape(logging_context, output_tensor, 4, output_tensor_id));
    TF_LITE_ENSURE_STATUS(CheckTensorNonDynamicAllocation(
        logging_context, output_tensor, output_tensor_id, node_index));

    const int* input_tensor_dims = input_tensor.dims->data;
    const int* filter_tensor_dims = filter_tensor.dims->data;
    const int output_channels = filter_tensor_dims[0];
    const int input_channels = filter_tensor_dims[3];

    const int32_t* output_shape = GetTensorData<int32_t>(&output_shape_tensor);
    if (output_channels != output_shape[3]) {
      TF_LITE_MAYBE_KERNEL_LOG(
          logging_context,
          "transpose convolution kernel output channel dimension (%d) "
          "doesn't match output shape channel dimension (%d) in node #%d: "
          "4 dimensions expected",
          output_channels, output_shape[3], node_index);
    }
    if (input_channels != input_tensor_dims[3]) {
      TF_LITE_MAYBE_KERNEL_LOG(
          logging_context,
          "transpose convolution kernel input channel dimension (%d) "
          "doesn't match filter input channel (%d) in node #%d",
          input_channels, input_tensor_dims[3]);
      return kTfLiteError;
    }

    TF_LITE_ENSURE_STATUS(CheckTransposedConvolutionParams(
        logging_context, deconv_params, node_index));

    std::string auto_pad;
    TF_LITE_ENSURE_STATUS(CalculatePadding(
        logging_context, deconv_params->padding, auto_pad, node_index));

    if (detect_supported_op) {
      TF_LITE_ENSURE_STATUS(CheckWebNNOpSupport(builder, "convTranspose2d"));
    } else {
      emscripten::val options = emscripten::val::object();
      options.set("autoPad", emscripten::val(auto_pad));
      std::vector<int32_t> strides = {deconv_params->stride_height,
                                      deconv_params->stride_width};
      options.set("strides", emscripten::val::array(strides));
      options.set("inputLayout", emscripten::val("nhwc"));
      options.set("filterLayout", emscripten::val("ohwi"));
      std::vector<int32_t> output_sizes = {output_shape[1], output_shape[2]};
      options.set("outputSizes", emscripten::val::array(output_sizes));
      TF_LITE_ENSURE(logging_context,
                     webnn_operands.at(input_tensor_id).as<bool>());
      TF_LITE_ENSURE(logging_context,
                     webnn_operands.at(filter_tensor_id).as<bool>());
      if (use_bias) {
        TF_LITE_ENSURE(logging_context,
                       webnn_operands.at(node->inputs->data[3]).as<bool>());
        options.set("bias", webnn_operands.at(node->inputs->data[3]));
      }
      emscripten::val output = builder.call<emscripten::val>(
          "convTranspose2d", webnn_operands.at(input_tensor_id),
          webnn_operands.at(filter_tensor_id), options);
      webnn_operands.insert(std::make_pair(output_tensor_id, output));
      TF_LITE_ENSURE(logging_context,
                     webnn_operands.at(output_tensor_id).as<bool>());
    }

    return kTfLiteOk;
  }

  static TfLiteStatus VisitUnpackNode(
      const emscripten::val& builder, TfLiteContext* logging_context,
      int node_index, TfLiteNode* node, const TfLiteTensor* tensors,
      const TfLiteUnpackParams* params,
      std::unordered_map<int, emscripten::val>& webnn_operands,
      const bool detect_supported_op) {
    const int num = params->num;
    int axis = params->axis;

    TF_LITE_ENSURE_STATUS(
        CheckNumInputsAndOutputs(logging_context, node, 1, num, node_index));

    const int input_tensor_id = node->inputs->data[0];
    const TfLiteTensor& input_tensor = tensors[input_tensor_id];
    TF_LITE_ENSURE_STATUS(CheckTensorFloat32OrQInt8Type(
        logging_context, input_tensor, input_tensor_id, node_index));
    TF_LITE_ENSURE_STATUS(CheckTensorNonDynamicAllocation(
        logging_context, input_tensor, input_tensor_id, node_index));

    const int num_dims = input_tensor.dims->size;
    if (axis < 0) {
      axis += num_dims;
    }
    if (axis < 0 || axis >= num_dims) {
      TF_LITE_MAYBE_KERNEL_LOG(
          logging_context,
          "unexpected value of axis %d in the unpack params in node %d",
          axis, node_index);
      return kTfLiteError;
    }

    const int output_size = node->outputs->size;

    if (num != output_size) {
      TF_LITE_MAYBE_KERNEL_LOG(
          logging_context,
          "unexpected value of num %d in the unpack params in node %d",
          num, node_index);
      return kTfLiteError;
    }

    if (detect_supported_op) {
      TF_LITE_ENSURE_STATUS(CheckWebNNOpSupport(builder, "squeeze"));
      if (num != 1) {
        TF_LITE_ENSURE_STATUS(CheckWebNNOpSupport(builder, "split"));
      }
    } else {
      TF_LITE_ENSURE(logging_context, webnn_operands.at(input_tensor_id).as<bool>());
      emscripten::val squeeze_options = emscripten::val::object();
      std::vector<int32_t> axes = {static_cast<int32_t>(axis)};
      squeeze_options.set("axes", emscripten::val::array(axes));
      // Unpack = split + squeeze in WebNN
      // No need split if Unpack's num == 1
      if (num == 1) {
        int output_tensor_id = node->outputs->data[0];
        const TfLiteTensor& output_tensor = tensors[output_tensor_id];
        TF_LITE_ENSURE_STATUS(CheckTensorFloat32Type(
            logging_context, output_tensor, output_tensor_id, node_index));
        TF_LITE_ENSURE_STATUS(CheckTensorNonDynamicAllocation(
            logging_context, output_tensor, output_tensor_id, node_index));

        webnn_operands.insert(std::make_pair(
            output_tensor_id, builder.call<emscripten::val>(
                                  "squeeze", webnn_operands.at(input_tensor_id),
                                  squeeze_options)));
        TF_LITE_ENSURE(logging_context,
                       webnn_operands.at(output_tensor_id).as<bool>());
      } else {
        std::vector<uint32_t> splits = {static_cast<const uint32_t>(num)};
        emscripten::val options = emscripten::val::object();
        options.set("axis", axis);
        emscripten::val split_operand_array = builder.call<emscripten::val>(
            "split", webnn_operands.at(input_tensor_id),
            emscripten::val::array(splits), options);
        TF_LITE_ENSURE(
            logging_context,
            split_operand_array["length"].as<int32_t>() == output_size);

        for (int i = 0; i < output_size; i++) {
          int output_tensor_id = node->outputs->data[i];
          const TfLiteTensor& output_tensor = tensors[output_tensor_id];
          TF_LITE_ENSURE_STATUS(CheckTensorFloat32Type(
              logging_context, output_tensor, output_tensor_id, node_index));
          TF_LITE_ENSURE_STATUS(CheckTensorNonDynamicAllocation(
              logging_context, output_tensor, output_tensor_id, node_index));

          webnn_operands.insert(std::make_pair(
              output_tensor_id,
              builder.call<emscripten::val>("squeeze", split_operand_array[i],
                                            squeeze_options)));
          TF_LITE_ENSURE(logging_context,
                         webnn_operands.at(output_tensor_id).as<bool>());
        }
      }
    }
    return kTfLiteOk;
  }

 private:
  Subgraph(emscripten::val context, emscripten::val graph, std::unordered_set<int>&& inputs, std::unordered_set<int>&& outputs)
      : wnn_context_(context), wnn_graph_(graph), inputs_(inputs), outputs_(outputs) {
    for (auto& i : inputs_) {
      externals_[i] = nullptr;
    }
    for (auto& o : outputs_) {
      externals_[o] = nullptr;
    }
    graph_inputs_ = emscripten::val::object();
    graph_outputs_ = emscripten::val::object();
  }

  emscripten::val wnn_context_ = emscripten::val::object();
  emscripten::val wnn_graph_ = emscripten::val::object();
  // TFLite Tensor IDs == name of input/output tensors for the
  // delegated subgraph.
  std::unordered_set<int> inputs_;
  std::unordered_set<int> outputs_;
  emscripten::val graph_inputs_ = emscripten::val::object();
  emscripten::val graph_outputs_ = emscripten::val::object();
  std::unordered_map<int, void*> externals_;
  char dummy_data_{0};
};

TfLiteIntArray* Delegate::PrepareOpsToDelegate(TfLiteContext* context) {
  // Clear previous data, in case the delegate is reused without re-creation.
  static_unpacked_data_map_.clear();
  static_unpacked_data_.clear();
  static_unpack_nodes_.clear();
  static_sparse_weights_.clear();

  TfLiteIntArray* execution_plan = nullptr;
  if (context->GetExecutionPlan(context, &execution_plan) != kTfLiteOk) {
    TF_LITE_KERNEL_LOG(context, "Unable to get graph execution plan.");
    return nullptr;
  }

  // Mapping for quasi-static (unpacked from static) tensor index to the node
  // index that produced it.
  std::unordered_map<int, int> quasi_static_tensors_producers;
  // Set of all quasi-static tensors in the execution plan.
  std::unordered_set<int> quasi_static_tensors;
  // Set of quasi-static tensors consumed by the delegated nodes.
  std::unordered_set<int> quasi_static_tensors_to_unpack;

  TfLiteIntArray* nodes_to_delegate =
      TfLiteIntArrayCreate(execution_plan->size);
  nodes_to_delegate->size = 0;
  for (int i = 0; i < execution_plan->size; ++i) {
    const int node_index = execution_plan->data[i];

    // Check if TFLite nodes can be delegated to WebNN
    TfLiteNode* node = nullptr;
    TfLiteRegistration* registration = nullptr;
    if (context->GetNodeAndRegistration(context, node_index, &node,
                                        &registration) != kTfLiteOk) {
      TF_LITE_KERNEL_LOG(context,
                         "Unable to get node and registration for node %d.",
                         node_index);
      continue;  // Soft error (skip this node).
    }

    // Prepare to unpack FP16 tensors.
    if (registration->builtin_code == kTfLiteBuiltinDequantize &&
        node->inputs->size == 1 && node->outputs->size == 1) {
      const TfLiteTensor& input_tensor =
          context->tensors[node->inputs->data[0]];
      const TfLiteTensor& output_tensor =
          context->tensors[node->outputs->data[0]];
      if ((input_tensor.allocation_type == kTfLiteMmapRo ||
           quasi_static_tensors.count(node->inputs->data[0]) != 0) &&
          input_tensor.type == kTfLiteFloat16 &&
          output_tensor.type == kTfLiteFloat32) {
        static_unpack_nodes_.insert(node_index);
        quasi_static_tensors_producers[node->outputs->data[0]] = node_index;
        quasi_static_tensors.insert(node->outputs->data[0]);

        if (input_tensor.allocation_type != kTfLiteMmapRo) {
          quasi_static_tensors_to_unpack.insert(node->inputs->data[0]);
        }

        // If dequantized input is sparse, so is its output
        if (static_sparse_weights_.count(node->inputs->data[0]) != 0) {
          static_sparse_weights_.insert(node->outputs->data[0]);
        }

        // Skip this node for now. If output of the node is consumed only by
        // delegated nodes, it will be added to nodes_to_delegate in the end.
        continue;
      }
    }

    // Prepare to unpack sparse tensors.
    // TODO(b/157729695): In the future, we also need to handle the case where a
    // sparse tensor is fed to a TFLite op directly, and no Densify() op is
    // inserted. For now this is not a problem because the Conv() op in tflite
    // can only consume dense tensors.
    if (registration->builtin_code == kTfLiteBuiltinDensify &&
        node->inputs->size == 1 && node->outputs->size == 1) {
      const TfLiteTensor& input_tensor =
          context->tensors[node->inputs->data[0]];
      const TfLiteTensor& output_tensor =
          context->tensors[node->outputs->data[0]];
      if (input_tensor.allocation_type == kTfLiteMmapRo &&
          input_tensor.sparsity != nullptr &&
          (input_tensor.type == kTfLiteFloat16 ||
           input_tensor.type == kTfLiteFloat32) &&
          output_tensor.type == input_tensor.type) {
        static_unpack_nodes_.insert(node_index);
        quasi_static_tensors_producers[node->outputs->data[0]] = node_index;
        quasi_static_tensors.insert(node->outputs->data[0]);
        static_sparse_weights_.insert(node->outputs->data[0]);

        // Skip this node for now. If output of the node is consumed only by
        // delegated nodes, it will be added to nodes_to_delegate in the end.
        continue;
      }
    }

    emscripten::val wnn_builder =
        emscripten::val::global("MLGraphBuilder").new_(wnn_context_);
    if (!wnn_builder.as<bool>()) {
      TF_LITE_KERNEL_LOG(context, "Failed to create WebNN graph builder.");
      return nullptr;
    }
    std::unordered_map<int, emscripten::val> empty_webnn_operands;
    if (Subgraph::VisitNode(wnn_builder, context, registration, node,
                            node_index, quasi_static_tensors,
                            empty_webnn_operands, true) != kTfLiteOk) {
      // If a non-delegated node consumes output of a node that unpacks static
      // data, that node shouldn't be delegated.
      for (int j = 0; j < node->inputs->size; j++) {
        const auto it =
            quasi_static_tensors_producers.find(node->inputs->data[j]);
        if (it != quasi_static_tensors_producers.end()) {
          static_unpack_nodes_.erase(it->second);
        }
      }

      // Non-delegatable node is not an error.
      continue;
    }

    for (int j = 0; j < node->inputs->size; j++) {
      if (quasi_static_tensors.count(node->inputs->data[j]) != 0) {
        quasi_static_tensors_to_unpack.insert(node->inputs->data[j]);
      }
    }

    nodes_to_delegate->data[nodes_to_delegate->size++] = node_index;
  }

  // Sort quasi-static tensors to be unpacked by the node index the produced
  // them. This ensures that in situations where quasi-static tensor is
  // produced from another quasi-static tensor, the tensors are unpacked in
  // the original execution plan order.
  std::vector<int> sorted_quasi_static_tensors_to_unpack(
      quasi_static_tensors_to_unpack.cbegin(),
      quasi_static_tensors_to_unpack.cend());
  std::sort(sorted_quasi_static_tensors_to_unpack.begin(),
            sorted_quasi_static_tensors_to_unpack.end(),
            [&quasi_static_tensors_producers](int t1, int t2) {
              return quasi_static_tensors_producers[t1] <
                     quasi_static_tensors_producers[t2];
            });

  // Unpack static data of all tensors
  for (int t : sorted_quasi_static_tensors_to_unpack) {
    const int producer_index = quasi_static_tensors_producers[t];
    // Check if TFLite nodes can be delegated to WebNN
    TfLiteNode* node = nullptr;
    TfLiteRegistration* registration = nullptr;
    if (context->GetNodeAndRegistration(context, producer_index, &node,
                                        &registration) != kTfLiteOk) {
      TF_LITE_KERNEL_LOG(context,
                         "Unable to get node and registration for node %d.",
                         producer_index);
      TfLiteIntArrayFree(nodes_to_delegate);
      return nullptr;  // Hard error.
    }

    if (node->inputs->size != 1) {
      TF_LITE_KERNEL_LOG(context, "unexpected number of inputs (%d) in node %d",
                         node->inputs->size, producer_index);
      TfLiteIntArrayFree(nodes_to_delegate);
      return nullptr;  // Hard error.
    }

    if (node->outputs->size != 1) {
      TF_LITE_KERNEL_LOG(context,
                         "unexpected number of outputs (%d) in node %d",
                         node->outputs->size, producer_index);
      TfLiteIntArrayFree(nodes_to_delegate);
      return nullptr;  // Hard error.
    }

    const TfLiteTensor& input_tensor = context->tensors[node->inputs->data[0]];

    // Consider the case when the input to unpacking node is quasi-static.
    const auto static_unpacked_input_it_ =
        static_unpacked_data_map_.find(node->inputs->data[0]);
    if (static_unpacked_input_it_ == static_unpacked_data_map_.end()) {
      if (input_tensor.allocation_type != kTfLiteMmapRo) {
        TF_LITE_KERNEL_LOG(
            context,
            "unexpected allocation type (%d) in tensor %d in node %d (%d)",
            input_tensor.allocation_type, node->inputs->data[0], producer_index,
            registration->builtin_code);
        TfLiteIntArrayFree(nodes_to_delegate);
        return nullptr;  // Hard error.
      }
    }

    const TfLiteTensor& output_tensor = context->tensors[t];
    size_t tensor_elements = output_tensor.bytes;
    switch (output_tensor.type) {
      case kTfLiteFloat32:
        tensor_elements /= sizeof(float);
        break;
      case kTfLiteFloat16:
        tensor_elements /= sizeof(uint16_t);
        break;
      default: {
        TF_LITE_KERNEL_LOG(context,
                           "unexpected datatype (%s) in tensor %d in node %d",
                           TfLiteTypeGetName(output_tensor.type),
                           node->outputs->data[0], producer_index);
        TfLiteIntArrayFree(nodes_to_delegate);
        return nullptr;  // Hard error.
      }
    }

    const size_t tensor_offset = static_unpacked_data_.size();
    static_unpacked_data_.resize(tensor_offset + context->tensors[t].bytes);

    char* unpacked_data = static_unpacked_data_.data() + tensor_offset;
    const char* packed_data =
        static_unpacked_input_it_ != static_unpacked_data_map_.end()
            ? static_unpacked_data_.data() + static_unpacked_input_it_->second
            : static_cast<const char*>(input_tensor.data.data);
    switch (registration->builtin_code) {
      case kTfLiteBuiltinDequantize: {
        if (input_tensor.type != kTfLiteFloat16) {
          TF_LITE_KERNEL_LOG(
              context, "unexpected tensor %d data type (%s) in node %d",
              node->inputs->data[0], TfLiteTypeGetName(input_tensor.type),
              producer_index);
          TfLiteIntArrayFree(nodes_to_delegate);
          return nullptr;  // Hard error.
        }

        if (input_tensor.sparsity != nullptr) {
          TF_LITE_KERNEL_LOG(context,
                             "unexpected FP16 sparse tensor %d in node %d",
                             node->inputs->data[0], producer_index);
          TfLiteIntArrayFree(nodes_to_delegate);
          return nullptr;  // Hard error.
        }

        // Actual data unpacking
        float* unpacked_fp32_data = reinterpret_cast<float*>(unpacked_data);
        const uint16_t* packed_fp16_data =
            reinterpret_cast<const uint16_t*>(packed_data);
        for (size_t i = 0; i < tensor_elements; i++) {
          unpacked_fp32_data[i] = fp16_ieee_to_fp32_value(packed_fp16_data[i]);
        }
        break;
      }
      case kTfLiteBuiltinDensify: {
        if (input_tensor.sparsity == nullptr) {
          TF_LITE_KERNEL_LOG(context, "unexpected dense tensor %d in node %d",
                             node->inputs->data[0], producer_index);
          TfLiteIntArrayFree(nodes_to_delegate);
          return nullptr;  // Hard error.
        }

        const int dims_count = output_tensor.dims->size;
        std::vector<int> vector_shape(dims_count);
        for (int i = 0; i < dims_count; i++) {
          vector_shape[i] = output_tensor.dims->data[i];
        }

        switch (input_tensor.type) {
          case kTfLiteFloat32: {
            const size_t dense_size = context->tensors[t].bytes / sizeof(float);
            float* unpacked_fp32_data = reinterpret_cast<float*>(unpacked_data);
            tflite::internal::sparsity::FormatConverter<float> converter(
                vector_shape, *input_tensor.sparsity);
            converter.SparseToDense(
                static_cast<const float*>(input_tensor.data.data), dense_size,
                unpacked_fp32_data, context);
            break;
          }
          case kTfLiteFloat16: {
            const size_t dense_size =
                context->tensors[t].bytes / sizeof(Eigen::half);
            Eigen::half* unpacked_fp16_data =
                reinterpret_cast<Eigen::half*>(unpacked_data);
            tflite::internal::sparsity::FormatConverter<Eigen::half> converter(
                vector_shape, *input_tensor.sparsity);
            converter.SparseToDense(
                static_cast<const Eigen::half*>(input_tensor.data.data),
                dense_size, unpacked_fp16_data, context);
            break;
          }
          default: {
            TF_LITE_KERNEL_LOG(
                context, "unexpected tensor %d data type (%s) in node %d",
                node->inputs->data[0], TfLiteTypeGetName(input_tensor.type),
                producer_index);
            TfLiteIntArrayFree(nodes_to_delegate);
            return nullptr;  // Hard error.
          }
        }
        break;
      }
      default:
        TF_LITE_KERNEL_LOG(context, "unexpected op registration %d at node %d",
                           registration->builtin_code, producer_index);
        TfLiteIntArrayFree(nodes_to_delegate);
        return nullptr;  // Hard error.
    }

    static_unpacked_data_map_[t] = tensor_offset;
  }

  // Add nodes that unpack static data consumed by delegated nodes.
  // Note: this is done purely to avoid the overhead of running these nodes
  // again in TFLite interpreter which would allocate memory for their outputs.
  // We mark them as delegated, but the delegate would simply ignore these nodes
  // as the static weights are already unpacked.
  for (int node_index : static_unpack_nodes_) {
    nodes_to_delegate->data[nodes_to_delegate->size++] = node_index;
  }
  std::sort(&nodes_to_delegate->data[0],
            &nodes_to_delegate->data[nodes_to_delegate->size]);

#ifdef WEBNN_DELEGATE_TEST_MODE
  // In the test mode build (used by unit tests), WebNN delegate claims to
  // support all operators in the execution plan to disable fallback to the
  // default TensorFlow Lite kernels. Thus, if any of the ops in the model are
  // not supported by the delegate, they will cause a failure in
  // ::tflite::Interpreter::ModifyGraphWithDelegate, to be caught in the unit
  // tests.
  nodes_to_delegate->size = execution_plan->size;
  std::copy(&execution_plan->data[0],
            &execution_plan->data[execution_plan->size],
            &nodes_to_delegate->data[0]);
#endif

  TFLITE_LOG_PROD_ONCE(tflite::TFLITE_LOG_INFO,
                       "TensorFlow Lite WebNN delegate capability:"
                       " number of nodes in the graph: %d,"
                       " number of nodes supported by WebNN: %d.",
                       execution_plan->size, nodes_to_delegate->size);
  return nodes_to_delegate;
}

void* SubgraphInit(TfLiteContext* context, const char* buffer, size_t length) {
  const TfLiteDelegateParams* params =
      reinterpret_cast<const TfLiteDelegateParams*>(buffer);

  return static_cast<void*>(Subgraph::Create(
      context, params,
      static_cast<::tflite::webnn::Delegate*>(params->delegate->data_)));
}

TfLiteStatus SubgraphPrepare(TfLiteContext* context, TfLiteNode* node) {
  if (node->user_data == nullptr) {
    return kTfLiteError;
  }

  return static_cast<Subgraph*>(node->user_data)->Prepare(context);
}

TfLiteStatus SubgraphInvoke(TfLiteContext* context, TfLiteNode* node) {
  if (node->user_data == nullptr) {
    return kTfLiteError;
  }

  return static_cast<Subgraph*>(node->user_data)->Invoke(context);
}

void SubgraphFree(TfLiteContext* context, void* buffer) {
  if (buffer != nullptr) {
    delete static_cast<Subgraph*>(buffer);
  }
}

const TfLiteRegistration kSubgraphRegistration = {
    /*.init=*/SubgraphInit,
    /*.free=*/SubgraphFree,
    /*.prepare=*/SubgraphPrepare,
    /*.invoke=*/SubgraphInvoke,
    /*.profiling_string=*/nullptr,
    /*.builtin_code=*/0,
    /*.custom_name=*/"TfLiteWebNNDelegate",
    /*.version=*/2,
};

TfLiteStatus DelegatePrepare(TfLiteContext* context, TfLiteDelegate* delegate) {
  TfLiteIntArray* ops_to_replace =
      static_cast<::tflite::webnn::Delegate*>(delegate->data_)
          ->PrepareOpsToDelegate(context);
  if (ops_to_replace == nullptr) {
    return kTfLiteError;
  }

  const TfLiteStatus status = context->ReplaceNodeSubsetsWithDelegateKernels(
      context, kSubgraphRegistration, ops_to_replace, delegate);
  TfLiteIntArrayFree(ops_to_replace);
  return status;
}

}  // namespace
}  // namespace webnn
}  // namespace tflite

TfLiteWebNNDelegateOptions TfLiteWebNNDelegateOptionsDefault() {
  TfLiteWebNNDelegateOptions options = {0, 2, 0};
  return options;
}

TfLiteDelegate* TfLiteWebNNDelegateCreate(
    const TfLiteWebNNDelegateOptions* options) {
  std::unordered_map<uint32_t, std::string> device_type_name_s = {
      {0, "cpu"}, {1, "gpu"}, {2, "npu"}};
  std::unordered_map<uint32_t, std::string> power_preference_name_s = {
      {0, "default"}, {1, "high-performance"}, {2, "low-power"}};
  std::string device_type_name = device_type_name_s[options->deviceType];
  std::string power_preference_name =
      power_preference_name_s[options->powerPreference];
  TFLITE_LOG_PROD_ONCE(tflite::TFLITE_LOG_INFO,
                       "Created TensorFlow Lite WebNN delegate for device"
                       " %s, power %s and numThreads %d.",
                       device_type_name.c_str(), power_preference_name.c_str(),
                       options->numThreads);

  // Check if WebNN is supported
  const emscripten::val graph_builder =
      emscripten::val::global("MLGraphBuilder");
  if (!graph_builder.as<bool>()) {
    TFLITE_LOG_PROD_ONCE(tflite::TFLITE_LOG_ERROR,
                         "WebNN is not supported on current platform.");
    return nullptr;
  }

  // Create WebNN context
  const emscripten::val ml = emscripten::val::global("navigator")["ml"];
  emscripten::val context_options = emscripten::val::object();
  context_options.set("deviceType", emscripten::val(device_type_name));
  context_options.set("powerPreference",
                      emscripten::val(power_preference_name));
  context_options.set("numThreads", options->numThreads);
  emscripten::val wnn_context =
      ml.call<emscripten::val>("createContextSync", context_options);

  if (!wnn_context.as<bool>()) {
    TFLITE_LOG_PROD_ONCE(tflite::TFLITE_LOG_ERROR,
                         "Failed to create WebNN context.");
    return nullptr;
  }

  auto* webnn_delegate = new ::tflite::webnn::Delegate(wnn_context);
  return webnn_delegate ? webnn_delegate->tflite_delegate() : nullptr;
}

void TfLiteWebNNDelegateDelete(TfLiteDelegate* delegate) {
  if (delegate != nullptr) {
    delete static_cast<::tflite::webnn::Delegate*>(delegate->data_);
  }
}
