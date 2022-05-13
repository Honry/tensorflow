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
#include "tensorflow/lite/delegates/webnn/reduce_tester.h"
#include "tensorflow/lite/delegates/webnn/webnn_delegate.h"

namespace tflite {
namespace webnn {

TEST(Mean, DISABLED_4DReduceBatchSqueezeDims) {
  TfLiteWebNNDelegateOptions delegate_options =
      TfLiteWebNNDelegateOptionsDefault();
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteWebNNDelegateDelete)>
      webnn_delegate(TfLiteWebNNDelegateCreate(&delegate_options),
                     TfLiteWebNNDelegateDelete);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto shape_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 5), std::ref(rng));
  const auto batch = shape_rng();
  const auto height = shape_rng();
  const auto width = shape_rng();
  const auto channels = shape_rng();

  ReduceTester()
      .InputShape({batch, height, width, channels})
      .Axes({0})
      .KeepDims(false)
      .Test(BuiltinOperator_MEAN, webnn_delegate.get());
}

TEST(Mean, DISABLED_4DReduceBatchKeepDims) {
  TfLiteWebNNDelegateOptions delegate_options =
      TfLiteWebNNDelegateOptionsDefault();
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteWebNNDelegateDelete)>
      webnn_delegate(TfLiteWebNNDelegateCreate(&delegate_options),
                     TfLiteWebNNDelegateDelete);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto shape_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 5), std::ref(rng));
  const auto batch = shape_rng();
  const auto height = shape_rng();
  const auto width = shape_rng();
  const auto channels = shape_rng();

  ReduceTester()
      .InputShape({batch, height, width, channels})
      .Axes({0})
      .KeepDims(true)
      .Test(BuiltinOperator_MEAN, webnn_delegate.get());
}

TEST(Mean, DISABLED_4DReduceHeightSqueezeDims) {
  TfLiteWebNNDelegateOptions delegate_options =
      TfLiteWebNNDelegateOptionsDefault();
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteWebNNDelegateDelete)>
      webnn_delegate(TfLiteWebNNDelegateCreate(&delegate_options),
                     TfLiteWebNNDelegateDelete);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto shape_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 5), std::ref(rng));
  const auto batch = shape_rng();
  const auto height = shape_rng();
  const auto width = shape_rng();
  const auto channels = shape_rng();

  ReduceTester()
      .InputShape({batch, height, width, channels})
      .Axes({1})
      .KeepDims(false)
      .Test(BuiltinOperator_MEAN, webnn_delegate.get());
}

TEST(Mean, DISABLED_4DReduceHeightKeepDims) {
  TfLiteWebNNDelegateOptions delegate_options =
      TfLiteWebNNDelegateOptionsDefault();
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteWebNNDelegateDelete)>
      webnn_delegate(TfLiteWebNNDelegateCreate(&delegate_options),
                     TfLiteWebNNDelegateDelete);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto shape_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 5), std::ref(rng));
  const auto batch = shape_rng();
  const auto height = shape_rng();
  const auto width = shape_rng();
  const auto channels = shape_rng();

  ReduceTester()
      .InputShape({batch, height, width, channels})
      .Axes({1})
      .KeepDims(true)
      .Test(BuiltinOperator_MEAN, webnn_delegate.get());
}

TEST(Mean, DISABLED_4DReduceWidthSqueezeDims) {
  TfLiteWebNNDelegateOptions delegate_options =
      TfLiteWebNNDelegateOptionsDefault();
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteWebNNDelegateDelete)>
      webnn_delegate(TfLiteWebNNDelegateCreate(&delegate_options),
                     TfLiteWebNNDelegateDelete);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto shape_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 5), std::ref(rng));
  const auto batch = shape_rng();
  const auto height = shape_rng();
  const auto width = shape_rng();
  const auto channels = shape_rng();

  ReduceTester()
      .InputShape({batch, height, width, channels})
      .Axes({2})
      .KeepDims(false)
      .Test(BuiltinOperator_MEAN, webnn_delegate.get());
}

TEST(Mean, DISABLED_4DReduceWidthKeepDims) {
  TfLiteWebNNDelegateOptions delegate_options =
      TfLiteWebNNDelegateOptionsDefault();
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteWebNNDelegateDelete)>
      webnn_delegate(TfLiteWebNNDelegateCreate(&delegate_options),
                     TfLiteWebNNDelegateDelete);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto shape_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 5), std::ref(rng));
  const auto batch = shape_rng();
  const auto height = shape_rng();
  const auto width = shape_rng();
  const auto channels = shape_rng();

  ReduceTester()
      .InputShape({batch, height, width, channels})
      .Axes({2})
      .KeepDims(true)
      .Test(BuiltinOperator_MEAN, webnn_delegate.get());
}

TEST(Mean, 4DReduceHeightWidthSqueezeDims) {
  TfLiteWebNNDelegateOptions delegate_options =
      TfLiteWebNNDelegateOptionsDefault();
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteWebNNDelegateDelete)>
      webnn_delegate(TfLiteWebNNDelegateCreate(&delegate_options),
                     TfLiteWebNNDelegateDelete);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto shape_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 5), std::ref(rng));
  const auto batch = shape_rng();
  const auto height = shape_rng();
  const auto width = shape_rng();
  const auto channels = shape_rng();

  ReduceTester()
      .InputShape({batch, height, width, channels})
      .Axes({1, 2})
      .KeepDims(false)
      .Test(BuiltinOperator_MEAN, webnn_delegate.get());

  ReduceTester()
      .InputShape({batch, height, width, channels})
      .Axes({2, 1})
      .KeepDims(false)
      .Test(BuiltinOperator_MEAN, webnn_delegate.get());
}

TEST(Mean, 4DReduceHeightWidthKeepDims) {
  TfLiteWebNNDelegateOptions delegate_options =
      TfLiteWebNNDelegateOptionsDefault();
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteWebNNDelegateDelete)>
      webnn_delegate(TfLiteWebNNDelegateCreate(&delegate_options),
                     TfLiteWebNNDelegateDelete);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto shape_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 5), std::ref(rng));
  const auto batch = shape_rng();
  const auto height = shape_rng();
  const auto width = shape_rng();
  const auto channels = shape_rng();

  ReduceTester()
      .InputShape({batch, height, width, channels})
      .Axes({1, 2})
      .KeepDims(true)
      .Test(BuiltinOperator_MEAN, webnn_delegate.get());

  ReduceTester()
      .InputShape({batch, height, width, channels})
      .Axes({2, 1})
      .KeepDims(true)
      .Test(BuiltinOperator_MEAN, webnn_delegate.get());
}

TEST(Mean, DISABLED_4DReduceChannelsSqueezeDims) {
  TfLiteWebNNDelegateOptions delegate_options =
      TfLiteWebNNDelegateOptionsDefault();
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteWebNNDelegateDelete)>
      webnn_delegate(TfLiteWebNNDelegateCreate(&delegate_options),
                     TfLiteWebNNDelegateDelete);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto shape_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 5), std::ref(rng));
  const auto batch = shape_rng();
  const auto height = shape_rng();
  const auto width = shape_rng();
  const auto channels = shape_rng();

  ReduceTester()
      .InputShape({batch, height, width, channels})
      .Axes({3})
      .KeepDims(false)
      .Test(BuiltinOperator_MEAN, webnn_delegate.get());
}

TEST(Mean, DISABLED_4DReduceChannelsKeepDims) {
  TfLiteWebNNDelegateOptions delegate_options =
      TfLiteWebNNDelegateOptionsDefault();
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteWebNNDelegateDelete)>
      webnn_delegate(TfLiteWebNNDelegateCreate(&delegate_options),
                     TfLiteWebNNDelegateDelete);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto shape_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 5), std::ref(rng));
  const auto batch = shape_rng();
  const auto height = shape_rng();
  const auto width = shape_rng();
  const auto channels = shape_rng();

  ReduceTester()
      .InputShape({batch, height, width, channels})
      .Axes({3})
      .KeepDims(true)
      .Test(BuiltinOperator_MEAN, webnn_delegate.get());
}

TEST(Mean, DISABLED_3DReduceBatchSqueezeDims) {
  TfLiteWebNNDelegateOptions delegate_options =
      TfLiteWebNNDelegateOptionsDefault();
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteWebNNDelegateDelete)>
      webnn_delegate(TfLiteWebNNDelegateCreate(&delegate_options),
                     TfLiteWebNNDelegateDelete);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto shape_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 5), std::ref(rng));
  const auto batch = shape_rng();
  const auto width = shape_rng();
  const auto channels = shape_rng();

  ReduceTester()
      .InputShape({batch, width, channels})
      .Axes({0})
      .KeepDims(false)
      .Test(BuiltinOperator_MEAN, webnn_delegate.get());
}

TEST(Mean, DISABLED_3DReduceBatchKeepDims) {
  TfLiteWebNNDelegateOptions delegate_options =
      TfLiteWebNNDelegateOptionsDefault();
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteWebNNDelegateDelete)>
      webnn_delegate(TfLiteWebNNDelegateCreate(&delegate_options),
                     TfLiteWebNNDelegateDelete);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto shape_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 5), std::ref(rng));
  const auto batch = shape_rng();
  const auto width = shape_rng();
  const auto channels = shape_rng();

  ReduceTester()
      .InputShape({batch, width, channels})
      .Axes({0})
      .KeepDims(true)
      .Test(BuiltinOperator_MEAN, webnn_delegate.get());
}

TEST(Mean, DISABLED_3DReduceWidthSqueezeDims) {
  TfLiteWebNNDelegateOptions delegate_options =
      TfLiteWebNNDelegateOptionsDefault();
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteWebNNDelegateDelete)>
      webnn_delegate(TfLiteWebNNDelegateCreate(&delegate_options),
                     TfLiteWebNNDelegateDelete);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto shape_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 5), std::ref(rng));
  const auto batch = shape_rng();
  const auto width = shape_rng();
  const auto channels = shape_rng();

  ReduceTester()
      .InputShape({batch, width, channels})
      .Axes({1})
      .KeepDims(false)
      .Test(BuiltinOperator_MEAN, webnn_delegate.get());
}

TEST(Mean, DISABLED_3DReduceWidthKeepDims) {
  TfLiteWebNNDelegateOptions delegate_options =
      TfLiteWebNNDelegateOptionsDefault();
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteWebNNDelegateDelete)>
      webnn_delegate(TfLiteWebNNDelegateCreate(&delegate_options),
                     TfLiteWebNNDelegateDelete);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto shape_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 5), std::ref(rng));
  const auto batch = shape_rng();
  const auto width = shape_rng();
  const auto channels = shape_rng();

  ReduceTester()
      .InputShape({batch, width, channels})
      .Axes({1})
      .KeepDims(true)
      .Test(BuiltinOperator_MEAN, webnn_delegate.get());
}

TEST(Mean, DISABLED_3DReduceChannelsSqueezeDims) {
  TfLiteWebNNDelegateOptions delegate_options =
      TfLiteWebNNDelegateOptionsDefault();
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteWebNNDelegateDelete)>
      webnn_delegate(TfLiteWebNNDelegateCreate(&delegate_options),
                     TfLiteWebNNDelegateDelete);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto shape_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 5), std::ref(rng));
  const auto batch = shape_rng();
  const auto width = shape_rng();
  const auto channels = shape_rng();

  ReduceTester()
      .InputShape({batch, width, channels})
      .Axes({2})
      .KeepDims(false)
      .Test(BuiltinOperator_MEAN, webnn_delegate.get());
}

TEST(Mean, DISABLED_3DReduceChannelsKeepDims) {
  TfLiteWebNNDelegateOptions delegate_options =
      TfLiteWebNNDelegateOptionsDefault();
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteWebNNDelegateDelete)>
      webnn_delegate(TfLiteWebNNDelegateCreate(&delegate_options),
                     TfLiteWebNNDelegateDelete);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto shape_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 5), std::ref(rng));
  const auto batch = shape_rng();
  const auto width = shape_rng();
  const auto channels = shape_rng();

  ReduceTester()
      .InputShape({batch, width, channels})
      .Axes({2})
      .KeepDims(true)
      .Test(BuiltinOperator_MEAN, webnn_delegate.get());
}

TEST(Mean, DISABLED_2DReduceBatchSqueezeDims) {
  TfLiteWebNNDelegateOptions delegate_options =
      TfLiteWebNNDelegateOptionsDefault();
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteWebNNDelegateDelete)>
      webnn_delegate(TfLiteWebNNDelegateCreate(&delegate_options),
                     TfLiteWebNNDelegateDelete);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto shape_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 5), std::ref(rng));
  const auto batch = shape_rng();
  const auto channels = shape_rng();

  ReduceTester()
      .InputShape({batch, channels})
      .Axes({0})
      .KeepDims(false)
      .Test(BuiltinOperator_MEAN, webnn_delegate.get());
}

TEST(Mean, DISABLED_2DReduceBatchKeepDims) {
  TfLiteWebNNDelegateOptions delegate_options =
      TfLiteWebNNDelegateOptionsDefault();
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteWebNNDelegateDelete)>
      webnn_delegate(TfLiteWebNNDelegateCreate(&delegate_options),
                     TfLiteWebNNDelegateDelete);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto shape_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 5), std::ref(rng));
  const auto batch = shape_rng();
  const auto channels = shape_rng();

  ReduceTester()
      .InputShape({batch, channels})
      .Axes({0})
      .KeepDims(true)
      .Test(BuiltinOperator_MEAN, webnn_delegate.get());
}

TEST(Mean, DISABLED_2DReduceChannelsSqueezeDims) {
  TfLiteWebNNDelegateOptions delegate_options =
      TfLiteWebNNDelegateOptionsDefault();
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteWebNNDelegateDelete)>
      webnn_delegate(TfLiteWebNNDelegateCreate(&delegate_options),
                     TfLiteWebNNDelegateDelete);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto shape_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 5), std::ref(rng));
  const auto batch = shape_rng();
  const auto channels = shape_rng();

  ReduceTester()
      .InputShape({batch, channels})
      .Axes({1})
      .KeepDims(false)
      .Test(BuiltinOperator_MEAN, webnn_delegate.get());
}

TEST(Mean, DISABLED_2DReduceChannelsKeepDims) {
  TfLiteWebNNDelegateOptions delegate_options =
      TfLiteWebNNDelegateOptionsDefault();
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteWebNNDelegateDelete)>
      webnn_delegate(TfLiteWebNNDelegateCreate(&delegate_options),
                     TfLiteWebNNDelegateDelete);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto shape_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 5), std::ref(rng));
  const auto batch = shape_rng();
  const auto channels = shape_rng();

  ReduceTester()
      .InputShape({batch, channels})
      .Axes({1})
      .KeepDims(true)
      .Test(BuiltinOperator_MEAN, webnn_delegate.get());
}

TEST(Mean, DISABLED_1DSqueezeDims) {
  TfLiteWebNNDelegateOptions delegate_options =
      TfLiteWebNNDelegateOptionsDefault();
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteWebNNDelegateDelete)>
      webnn_delegate(TfLiteWebNNDelegateCreate(&delegate_options),
                     TfLiteWebNNDelegateDelete);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto shape_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 5), std::ref(rng));
  const auto batch = shape_rng();

  ReduceTester().InputShape({batch}).Axes({0}).KeepDims(false).Test(
      BuiltinOperator_MEAN, webnn_delegate.get());
}

TEST(Mean, DISABLED_1DKeepDims) {
  TfLiteWebNNDelegateOptions delegate_options =
      TfLiteWebNNDelegateOptionsDefault();
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteWebNNDelegateDelete)>
      webnn_delegate(TfLiteWebNNDelegateCreate(&delegate_options),
                     TfLiteWebNNDelegateDelete);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto shape_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 5), std::ref(rng));
  const auto batch = shape_rng();

  ReduceTester().InputShape({batch}).Axes({0}).KeepDims(true).Test(
      BuiltinOperator_MEAN, webnn_delegate.get());
}

}  // namespace webnn
}  // namespace tflite
