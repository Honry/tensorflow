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

#ifndef TENSORFLOW_LITE_DELEGATES_WEBNN_WEBNN_DELEGATE_H_
#define TENSORFLOW_LITE_DELEGATES_WEBNN_WEBNN_DELEGATE_H_

#include "tensorflow/lite/c/common.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct {
  // enum class DeviceType : uint32_t {
  //     Cpu = 0x00000000,
  //     Gpu = 0x00000001,
  //     Npu = 0x00000002,
  // };
  uint32_t deviceType;
  // enum class PowerPreference : uint32_t {
  //     Default = 0x00000000,
  //     High_performance = 0x00000001,
  //     Low_power = 0x00000002,
  // };
  uint32_t powerPreference;
  uint32_t numThreads;
} TfLiteWebNNDelegateOptions;

// Returns a structure with the default WebNN delegate options.
TfLiteWebNNDelegateOptions TfLiteWebNNDelegateOptionsDefault();

// Creates a new delegate instance that need to be destroyed with
// `TfLiteWebNNDelegateDelete` when delegate is no longer used by TFLite.
// When `options` is set to `nullptr`, the following default values are used:
TfLiteDelegate* TfLiteWebNNDelegateCreate(
    const TfLiteWebNNDelegateOptions* options);

// Destroys a delegate created with `TfLiteWebNNDelegateCreate` call.
void TfLiteWebNNDelegateDelete(TfLiteDelegate* delegate);

#ifdef __cplusplus
}
#endif  // __cplusplus

#endif  // TENSORFLOW_LITE_DELEGATES_WEBNN_WEBNN_DELEGATE_H_
