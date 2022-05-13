/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include <cstdint>
#include <functional>
#include <memory>
#include <random>

#include <gtest/gtest.h>
#include "tensorflow/lite/delegates/webnn/fully_connected_tester.h"
#include "tensorflow/lite/delegates/webnn/webnn_delegate.h"

namespace tflite {
namespace webnn {

TEST(FullyConnected, 1D) {
  TfLiteWebNNDelegateOptions delegate_options =
      TfLiteWebNNDelegateOptionsDefault();
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteWebNNDelegateDelete)>
      webnn_delegate(TfLiteWebNNDelegateCreate(&delegate_options),
                     TfLiteWebNNDelegateDelete);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto channels_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 9), std::ref(rng));
  const auto input_channels = channels_rng();
  const auto output_channels = channels_rng();


  FullyConnectedTester()
      .InputShape({input_channels})
      .InputChannels(input_channels)
      .OutputChannels(output_channels)
      .Test(webnn_delegate.get());
}

TEST(FullyConnected, 1DKeepDims) {
  TfLiteWebNNDelegateOptions delegate_options =
      TfLiteWebNNDelegateOptionsDefault();
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteWebNNDelegateDelete)>
      webnn_delegate(TfLiteWebNNDelegateCreate(&delegate_options),
                     TfLiteWebNNDelegateDelete);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto channels_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 9), std::ref(rng));
  const auto input_channels = channels_rng();
  const auto output_channels = channels_rng();

  FullyConnectedTester()
      .InputShape({input_channels})
      .InputChannels(input_channels)
      .OutputChannels(output_channels)
      .KeepDims(true)
      .Test(webnn_delegate.get());
}

TEST(FullyConnected, 2D) {
  TfLiteWebNNDelegateOptions delegate_options =
      TfLiteWebNNDelegateOptionsDefault();
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteWebNNDelegateDelete)>
      webnn_delegate(TfLiteWebNNDelegateCreate(&delegate_options),
                     TfLiteWebNNDelegateDelete);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto batch_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 5), std::ref(rng));
  auto channels_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 9), std::ref(rng));
  const auto batch = batch_rng();
  const auto input_channels = channels_rng();
  const auto output_channels = channels_rng();

  FullyConnectedTester()
      .InputShape({batch, input_channels})
      .InputChannels(input_channels)
      .OutputChannels(output_channels)
      .Test(webnn_delegate.get());
}

TEST(FullyConnected, 2DKeepDims) {
  TfLiteWebNNDelegateOptions delegate_options =
      TfLiteWebNNDelegateOptionsDefault();
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteWebNNDelegateDelete)>
      webnn_delegate(TfLiteWebNNDelegateCreate(&delegate_options),
                     TfLiteWebNNDelegateDelete);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto batch_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 5), std::ref(rng));
  auto channels_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 9), std::ref(rng));
  const auto batch = batch_rng();
  const auto input_channels = channels_rng();
  const auto output_channels = channels_rng();

  FullyConnectedTester()
      .InputShape({batch, input_channels})
      .InputChannels(input_channels)
      .OutputChannels(output_channels)
      .KeepDims(true)
      .Test(webnn_delegate.get());
}

TEST(FullyConnected, 3D) {
  TfLiteWebNNDelegateOptions delegate_options =
      TfLiteWebNNDelegateOptionsDefault();
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteWebNNDelegateDelete)>
      webnn_delegate(TfLiteWebNNDelegateCreate(&delegate_options),
                     TfLiteWebNNDelegateDelete);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto shape_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 5), std::ref(rng));
  auto channels_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 9), std::ref(rng));
  const auto batch = shape_rng();
  const auto width = shape_rng();
  const auto input_channels = channels_rng();
  const auto output_channels = channels_rng();

  FullyConnectedTester()
      .InputShape({batch, width, input_channels})
      .InputChannels(input_channels)
      .OutputChannels(output_channels)
      .Test(webnn_delegate.get());
}

TEST(FullyConnected, 3DReshape) {
  TfLiteWebNNDelegateOptions delegate_options =
      TfLiteWebNNDelegateOptionsDefault();
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteWebNNDelegateDelete)>
      webnn_delegate(TfLiteWebNNDelegateCreate(&delegate_options),
                     TfLiteWebNNDelegateDelete);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto shape_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 5), std::ref(rng));
  auto channels_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 9), std::ref(rng));
  const auto batch = shape_rng();
  const auto width = shape_rng();
  const auto input_channels = channels_rng();
  const auto output_channels = channels_rng();

  FullyConnectedTester()
      .InputShape({batch, width, input_channels})
      .InputChannels(width * input_channels)
      .OutputChannels(output_channels)
      .Test(webnn_delegate.get());
}

TEST(FullyConnected, 3DKeepDims) {
  TfLiteWebNNDelegateOptions delegate_options =
      TfLiteWebNNDelegateOptionsDefault();
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteWebNNDelegateDelete)>
      webnn_delegate(TfLiteWebNNDelegateCreate(&delegate_options),
                     TfLiteWebNNDelegateDelete);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto shape_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 5), std::ref(rng));
  auto channels_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 9), std::ref(rng));
  const auto batch = shape_rng();
  const auto width = shape_rng();
  const auto input_channels = channels_rng();
  const auto output_channels = channels_rng();

  FullyConnectedTester()
      .InputShape({batch, width, input_channels})
      .InputChannels(input_channels)
      .OutputChannels(output_channels)
      .KeepDims(true)
      .Test(webnn_delegate.get());
}

TEST(FullyConnected, 4D) {
  TfLiteWebNNDelegateOptions delegate_options =
      TfLiteWebNNDelegateOptionsDefault();
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteWebNNDelegateDelete)>
      webnn_delegate(TfLiteWebNNDelegateCreate(&delegate_options),
                     TfLiteWebNNDelegateDelete);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto shape_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 5), std::ref(rng));
  auto channels_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 9), std::ref(rng));
  const auto batch = shape_rng();
  const auto height = shape_rng();
  const auto width = shape_rng();
  const auto input_channels = channels_rng();
  const auto output_channels = channels_rng();

  FullyConnectedTester()
      .InputShape({batch, height, width, input_channels})
      .InputChannels(input_channels)
      .OutputChannels(output_channels)
      .Test(webnn_delegate.get());
}

TEST(FullyConnected, 4DKeepDims) {
  TfLiteWebNNDelegateOptions delegate_options =
      TfLiteWebNNDelegateOptionsDefault();
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteWebNNDelegateDelete)>
      webnn_delegate(TfLiteWebNNDelegateCreate(&delegate_options),
                     TfLiteWebNNDelegateDelete);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto shape_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 5), std::ref(rng));
  auto channels_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 9), std::ref(rng));
  const auto batch = shape_rng();
  const auto height = shape_rng();
  const auto width = shape_rng();
  const auto input_channels = channels_rng();
  const auto output_channels = channels_rng();

  FullyConnectedTester()
      .InputShape({batch, height, width, input_channels})
      .InputChannels(input_channels)
      .OutputChannels(output_channels)
      .KeepDims(true)
      .Test(webnn_delegate.get());
}

TEST(FullyConnected, NoBias) {
  TfLiteWebNNDelegateOptions delegate_options =
      TfLiteWebNNDelegateOptionsDefault();
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteWebNNDelegateDelete)>
      webnn_delegate(TfLiteWebNNDelegateCreate(&delegate_options),
                     TfLiteWebNNDelegateDelete);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto batch_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 5), std::ref(rng));
  auto channels_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 9), std::ref(rng));
  const auto batch = batch_rng();
  const auto input_channels = channels_rng();
  const auto output_channels = channels_rng();

  FullyConnectedTester()
      .InputShape({batch, input_channels})
      .InputChannels(input_channels)
      .OutputChannels(output_channels)
      .NoBias()
      .Test(webnn_delegate.get());
}

TEST(FullyConnected, FP16Weights) {
  TfLiteWebNNDelegateOptions delegate_options =
      TfLiteWebNNDelegateOptionsDefault();
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteWebNNDelegateDelete)>
      webnn_delegate(TfLiteWebNNDelegateCreate(&delegate_options),
                     TfLiteWebNNDelegateDelete);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto batch_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 5), std::ref(rng));
  auto channels_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 9), std::ref(rng));
  const auto batch = batch_rng();
  const auto input_channels = channels_rng();
  const auto output_channels = channels_rng();

  FullyConnectedTester()
      .InputShape({batch, input_channels})
      .InputChannels(input_channels)
      .OutputChannels(output_channels)
      .FP16Weights()
      .Test(webnn_delegate.get());
}

TEST(FullyConnected, FP16WeightsNoBias) {
  TfLiteWebNNDelegateOptions delegate_options =
      TfLiteWebNNDelegateOptionsDefault();
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteWebNNDelegateDelete)>
      webnn_delegate(TfLiteWebNNDelegateCreate(&delegate_options),
                     TfLiteWebNNDelegateDelete);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto batch_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 5), std::ref(rng));
  auto channels_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 9), std::ref(rng));
  const auto batch = batch_rng();
  const auto input_channels = channels_rng();
  const auto output_channels = channels_rng();

  FullyConnectedTester()
      .InputShape({batch, input_channels})
      .InputChannels(input_channels)
      .OutputChannels(output_channels)
      .FP16Weights()
      .NoBias()
      .Test(webnn_delegate.get());
}

TEST(FullyConnected, INT8Weights) {
  TfLiteWebNNDelegateOptions delegate_options =
      TfLiteWebNNDelegateOptionsDefault();
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteWebNNDelegateDelete)>
      webnn_delegate(TfLiteWebNNDelegateCreate(&delegate_options),
                     TfLiteWebNNDelegateDelete);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto batch_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 5), std::ref(rng));
  auto channels_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 9), std::ref(rng));
  const auto batch = batch_rng();
  const auto input_channels = channels_rng();
  const auto output_channels = channels_rng();

  FullyConnectedTester()
      .InputShape({batch, input_channels})
      .InputChannels(input_channels)
      .OutputChannels(output_channels)
      .INT8Weights()
      .Test(webnn_delegate.get());
}

TEST(FullyConnected, INT8WeightsNoBias) {
  TfLiteWebNNDelegateOptions delegate_options =
      TfLiteWebNNDelegateOptionsDefault();
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteWebNNDelegateDelete)>
      webnn_delegate(TfLiteWebNNDelegateCreate(&delegate_options),
                     TfLiteWebNNDelegateDelete);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto batch_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 5), std::ref(rng));
  auto channels_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 9), std::ref(rng));
  const auto batch = batch_rng();
  const auto input_channels = channels_rng();
  const auto output_channels = channels_rng();

  FullyConnectedTester()
      .InputShape({batch, input_channels})
      .InputChannels(input_channels)
      .OutputChannels(output_channels)
      .INT8Weights()
      .NoBias()
      .Test(webnn_delegate.get());
}

TEST(FullyConnected, INT8ChannelWiseWeights) {
  TfLiteWebNNDelegateOptions delegate_options =
      TfLiteWebNNDelegateOptionsDefault();
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteWebNNDelegateDelete)>
      webnn_delegate(TfLiteWebNNDelegateCreate(&delegate_options),
                     TfLiteWebNNDelegateDelete);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto batch_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 5), std::ref(rng));
  auto channels_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 9), std::ref(rng));
  const auto batch = batch_rng();
  const auto input_channels = channels_rng();
  const auto output_channels = channels_rng();

  FullyConnectedTester()
      .InputShape({batch, input_channels})
      .InputChannels(input_channels)
      .OutputChannels(output_channels)
      .INT8ChannelWiseWeights()
      .Test(webnn_delegate.get());
}

TEST(FullyConnected, INT8ChannelWiseWeightsNoBias) {
  TfLiteWebNNDelegateOptions delegate_options =
      TfLiteWebNNDelegateOptionsDefault();
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteWebNNDelegateDelete)>
      webnn_delegate(TfLiteWebNNDelegateCreate(&delegate_options),
                     TfLiteWebNNDelegateDelete);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto batch_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 5), std::ref(rng));
  auto channels_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 9), std::ref(rng));
  const auto batch = batch_rng();
  const auto input_channels = channels_rng();
  const auto output_channels = channels_rng();

  FullyConnectedTester()
      .InputShape({batch, input_channels})
      .InputChannels(input_channels)
      .OutputChannels(output_channels)
      .INT8ChannelWiseWeights()
      .NoBias()
      .Test(webnn_delegate.get());
}

TEST(FullyConnected, ReluActivation) {
  TfLiteWebNNDelegateOptions delegate_options =
      TfLiteWebNNDelegateOptionsDefault();
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteWebNNDelegateDelete)>
      webnn_delegate(TfLiteWebNNDelegateCreate(&delegate_options),
                     TfLiteWebNNDelegateDelete);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto batch_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 5), std::ref(rng));
  auto channels_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 9), std::ref(rng));
  const auto batch = batch_rng();
  const auto input_channels = channels_rng();
  const auto output_channels = channels_rng();

  FullyConnectedTester()
      .InputShape({batch, input_channels})
      .InputChannels(input_channels)
      .OutputChannels(output_channels)
      .ReluActivation()
      .Test(webnn_delegate.get());
}

TEST(FullyConnected, Relu6Activation) {
  TfLiteWebNNDelegateOptions delegate_options =
      TfLiteWebNNDelegateOptionsDefault();
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteWebNNDelegateDelete)>
      webnn_delegate(TfLiteWebNNDelegateCreate(&delegate_options),
                     TfLiteWebNNDelegateDelete);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto batch_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 5), std::ref(rng));
  auto channels_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 9), std::ref(rng));
  const auto batch = batch_rng();
  const auto input_channels = channels_rng();
  const auto output_channels = channels_rng();

  FullyConnectedTester()
      .InputShape({batch, input_channels})
      .InputChannels(input_channels)
      .OutputChannels(output_channels)
      .Relu6Activation()
      .Test(webnn_delegate.get());
}

TEST(FullyConnected, ReluMinus1To1Activation) {
  TfLiteWebNNDelegateOptions delegate_options =
      TfLiteWebNNDelegateOptionsDefault();
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteWebNNDelegateDelete)>
      webnn_delegate(TfLiteWebNNDelegateCreate(&delegate_options),
                     TfLiteWebNNDelegateDelete);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto batch_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 5), std::ref(rng));
  auto channels_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 9), std::ref(rng));
  const auto batch = batch_rng();
  const auto input_channels = channels_rng();
  const auto output_channels = channels_rng();

  FullyConnectedTester()
      .InputShape({batch, input_channels})
      .InputChannels(input_channels)
      .OutputChannels(output_channels)
      .ReluMinus1To1Activation()
      .Test(webnn_delegate.get());
}

}  // namespace webnn
}  // namespace tflite
