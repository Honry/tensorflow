/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/delegates/webnn/binary_elementwise_tester.h"
#include "tensorflow/lite/delegates/webnn/webnn_delegate.h"

namespace tflite {
namespace webnn {

TEST(Add, 4DBy4D) {
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

  BinaryElementwiseTester()
      .Input1Shape({batch, height, width, channels})
      .Input2Shape({batch, height, width, channels})
      .Test(BuiltinOperator_ADD, webnn_delegate.get());
}

TEST(Add, 4DBy4DBroadcastChannels) {
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

  BinaryElementwiseTester()
      .Input1Shape({1, 1, 1, channels})
      .Input2Shape({batch, height, width, channels})
      .Test(BuiltinOperator_ADD, webnn_delegate.get());

  BinaryElementwiseTester()
      .Input1Shape({batch, height, width, channels})
      .Input2Shape({1, 1, 1, channels})
      .Test(BuiltinOperator_ADD, webnn_delegate.get());
}

TEST(Add, 4DBy4DBroadcastWidth) {
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

  BinaryElementwiseTester()
      .Input1Shape({1, 1, width, 1})
      .Input2Shape({batch, height, width, channels})
      .Test(BuiltinOperator_ADD, webnn_delegate.get());

  BinaryElementwiseTester()
      .Input1Shape({batch, height, width, channels})
      .Input2Shape({1, 1, width, 1})
      .Test(BuiltinOperator_ADD, webnn_delegate.get());
}

TEST(Add, 4DBy4DBroadcastHeight) {
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

  BinaryElementwiseTester()
      .Input1Shape({1, height, 1, 1})
      .Input2Shape({batch, height, width, channels})
      .Test(BuiltinOperator_ADD, webnn_delegate.get());

  BinaryElementwiseTester()
      .Input1Shape({batch, height, width, channels})
      .Input2Shape({1, height, 1, 1})
      .Test(BuiltinOperator_ADD, webnn_delegate.get());
}

TEST(Add, 4DBy4DBroadcastBatch) {
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

  BinaryElementwiseTester()
      .Input1Shape({batch, 1, 1, 1})
      .Input2Shape({batch, height, width, channels})
      .Test(BuiltinOperator_ADD, webnn_delegate.get());

  BinaryElementwiseTester()
      .Input1Shape({batch, height, width, channels})
      .Input2Shape({batch, 1, 1, 1})
      .Test(BuiltinOperator_ADD, webnn_delegate.get());
}

TEST(Add, 4DBy4DBroadcastHeightWidthChannels) {
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

  BinaryElementwiseTester()
      .Input1Shape({1, height, width, channels})
      .Input2Shape({batch, height, width, channels})
      .Test(BuiltinOperator_ADD, webnn_delegate.get());

  BinaryElementwiseTester()
      .Input1Shape({batch, height, width, channels})
      .Input2Shape({1, height, width, channels})
      .Test(BuiltinOperator_ADD, webnn_delegate.get());
}

TEST(Add, 4DBy3D) {
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

  BinaryElementwiseTester()
      .Input1Shape({height, width, channels})
      .Input2Shape({batch, height, width, channels})
      .Test(BuiltinOperator_ADD, webnn_delegate.get());

  BinaryElementwiseTester()
      .Input1Shape({batch, height, width, channels})
      .Input2Shape({height, width, channels})
      .Test(BuiltinOperator_ADD, webnn_delegate.get());
}

TEST(Add, 4DBy2D) {
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

  BinaryElementwiseTester()
      .Input1Shape({width, channels})
      .Input2Shape({batch, height, width, channels})
      .Test(BuiltinOperator_ADD, webnn_delegate.get());

  BinaryElementwiseTester()
      .Input1Shape({batch, height, width, channels})
      .Input2Shape({width, channels})
      .Test(BuiltinOperator_ADD, webnn_delegate.get());
}

TEST(Add, 4DBy1D) {
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

  BinaryElementwiseTester()
      .Input1Shape({channels})
      .Input2Shape({batch, height, width, channels})
      .Test(BuiltinOperator_ADD, webnn_delegate.get());

  BinaryElementwiseTester()
      .Input1Shape({batch, height, width, channels})
      .Input2Shape({channels})
      .Test(BuiltinOperator_ADD, webnn_delegate.get());
}

TEST(Add, 4DBy0D) {
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

  BinaryElementwiseTester()
      .Input1Shape({})
      .Input2Shape({batch, height, width, channels})
      .Test(BuiltinOperator_ADD, webnn_delegate.get());

  BinaryElementwiseTester()
      .Input1Shape({batch, height, width, channels})
      .Input2Shape({})
      .Test(BuiltinOperator_ADD, webnn_delegate.get());
}

TEST(Add, 2DBy2D) {
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

  BinaryElementwiseTester()
      .Input1Shape({batch, channels})
      .Input2Shape({batch, channels})
      .Test(BuiltinOperator_ADD, webnn_delegate.get());
}

TEST(Add, 2DBy1D) {
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

  BinaryElementwiseTester()
      .Input1Shape({channels})
      .Input2Shape({batch, channels})
      .Test(BuiltinOperator_ADD, webnn_delegate.get());

  BinaryElementwiseTester()
      .Input1Shape({batch, channels})
      .Input2Shape({channels})
      .Test(BuiltinOperator_ADD, webnn_delegate.get());
}

TEST(Add, 2DBy0D) {
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

  BinaryElementwiseTester()
      .Input1Shape({})
      .Input2Shape({batch, channels})
      .Test(BuiltinOperator_ADD, webnn_delegate.get());

  BinaryElementwiseTester()
      .Input1Shape({batch, channels})
      .Input2Shape({})
      .Test(BuiltinOperator_ADD, webnn_delegate.get());
}

TEST(Add, 4DByStatic4D) {
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

  BinaryElementwiseTester()
      .Input1Shape({batch, height, width, channels})
      .Input2Shape({batch, height, width, channels})
      .Input1Static(true)
      .Test(BuiltinOperator_ADD, webnn_delegate.get());

  BinaryElementwiseTester()
      .Input1Shape({batch, height, width, channels})
      .Input2Shape({batch, height, width, channels})
      .Input2Static(true)
      .Test(BuiltinOperator_ADD, webnn_delegate.get());
}

TEST(Add, 4DByStatic4DBroadcastChannels) {
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

  BinaryElementwiseTester()
      .Input1Shape({1, 1, 1, channels})
      .Input2Shape({batch, height, width, channels})
      .Input1Static(true)
      .Test(BuiltinOperator_ADD, webnn_delegate.get());

  BinaryElementwiseTester()
      .Input1Shape({batch, height, width, channels})
      .Input2Shape({1, 1, 1, channels})
      .Input2Static(true)
      .Test(BuiltinOperator_ADD, webnn_delegate.get());
}

TEST(Add, 4DByStatic4DBroadcastWidth) {
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

  BinaryElementwiseTester()
      .Input1Shape({1, 1, width, 1})
      .Input2Shape({batch, height, width, channels})
      .Input1Static(true)
      .Test(BuiltinOperator_ADD, webnn_delegate.get());

  BinaryElementwiseTester()
      .Input1Shape({batch, height, width, channels})
      .Input2Shape({1, 1, width, 1})
      .Input2Static(true)
      .Test(BuiltinOperator_ADD, webnn_delegate.get());
}

TEST(Add, 4DByStatic4DBroadcastHeight) {
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

  BinaryElementwiseTester()
      .Input1Shape({1, height, 1, 1})
      .Input2Shape({batch, height, width, channels})
      .Input1Static(true)
      .Test(BuiltinOperator_ADD, webnn_delegate.get());

  BinaryElementwiseTester()
      .Input1Shape({batch, height, width, channels})
      .Input2Shape({1, height, 1, 1})
      .Input2Static(true)
      .Test(BuiltinOperator_ADD, webnn_delegate.get());
}

TEST(Add, 4DByStatic4DBroadcastBatch) {
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

  BinaryElementwiseTester()
      .Input1Shape({batch, 1, 1, 1})
      .Input2Shape({batch, height, width, channels})
      .Input1Static(true)
      .Test(BuiltinOperator_ADD, webnn_delegate.get());

  BinaryElementwiseTester()
      .Input1Shape({batch, height, width, channels})
      .Input2Shape({batch, 1, 1, 1})
      .Input2Static(true)
      .Test(BuiltinOperator_ADD, webnn_delegate.get());
}

TEST(Add, 4DByStatic4DBroadcastHeightWidthChannels) {
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

  BinaryElementwiseTester()
      .Input1Shape({1, height, width, channels})
      .Input2Shape({batch, height, width, channels})
      .Input1Static(true)
      .Test(BuiltinOperator_ADD, webnn_delegate.get());

  BinaryElementwiseTester()
      .Input1Shape({batch, height, width, channels})
      .Input2Shape({1, height, width, channels})
      .Input2Static(true)
      .Test(BuiltinOperator_ADD, webnn_delegate.get());
}

TEST(Add, 4DByStatic3D) {
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

  BinaryElementwiseTester()
      .Input1Shape({height, width, channels})
      .Input2Shape({batch, height, width, channels})
      .Input1Static(true)
      .Test(BuiltinOperator_ADD, webnn_delegate.get());

  BinaryElementwiseTester()
      .Input1Shape({batch, height, width, channels})
      .Input2Shape({height, width, channels})
      .Input2Static(true)
      .Test(BuiltinOperator_ADD, webnn_delegate.get());
}

TEST(Add, 4DByStatic2D) {
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

  BinaryElementwiseTester()
      .Input1Shape({width, channels})
      .Input2Shape({batch, height, width, channels})
      .Input1Static(true)
      .Test(BuiltinOperator_ADD, webnn_delegate.get());

  BinaryElementwiseTester()
      .Input1Shape({batch, height, width, channels})
      .Input2Shape({width, channels})
      .Input2Static(true)
      .Test(BuiltinOperator_ADD, webnn_delegate.get());
}

TEST(Add, 4DByStatic1D) {
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

  BinaryElementwiseTester()
      .Input1Shape({channels})
      .Input2Shape({batch, height, width, channels})
      .Input1Static(true)
      .Test(BuiltinOperator_ADD, webnn_delegate.get());

  BinaryElementwiseTester()
      .Input1Shape({batch, height, width, channels})
      .Input2Shape({channels})
      .Input2Static(true)
      .Test(BuiltinOperator_ADD, webnn_delegate.get());
}

TEST(Add, 4DByStatic0D) {
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

  BinaryElementwiseTester()
      .Input1Shape({})
      .Input2Shape({batch, height, width, channels})
      .Input1Static(true)
      .Test(BuiltinOperator_ADD, webnn_delegate.get());

  BinaryElementwiseTester()
      .Input1Shape({batch, height, width, channels})
      .Input2Shape({})
      .Input2Static(true)
      .Test(BuiltinOperator_ADD, webnn_delegate.get());
}

TEST(Add, 2DByStatic2D) {
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

  BinaryElementwiseTester()
      .Input1Shape({batch, channels})
      .Input2Shape({batch, channels})
      .Input1Static(true)
      .Test(BuiltinOperator_ADD, webnn_delegate.get());

  BinaryElementwiseTester()
      .Input1Shape({batch, channels})
      .Input2Shape({batch, channels})
      .Input2Static(true)
      .Test(BuiltinOperator_ADD, webnn_delegate.get());
}

TEST(Add, 2DByStatic1D) {
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

  BinaryElementwiseTester()
      .Input1Shape({channels})
      .Input2Shape({batch, channels})
      .Input1Static(true)
      .Test(BuiltinOperator_ADD, webnn_delegate.get());

  BinaryElementwiseTester()
      .Input1Shape({batch, channels})
      .Input2Shape({channels})
      .Input2Static(true)
      .Test(BuiltinOperator_ADD, webnn_delegate.get());
}

TEST(Add, 2DByStatic0D) {
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

  BinaryElementwiseTester()
      .Input1Shape({})
      .Input2Shape({batch, channels})
      .Input1Static(true)
      .Test(BuiltinOperator_ADD, webnn_delegate.get());

  BinaryElementwiseTester()
      .Input1Shape({batch, channels})
      .Input2Shape({})
      .Input2Static(true)
      .Test(BuiltinOperator_ADD, webnn_delegate.get());
}

TEST(Add, FP16Weights) {
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

  BinaryElementwiseTester()
      .Input1Shape({batch, height, width, channels})
      .Input2Shape({batch, height, width, channels})
      .Input1Static(true)
      .FP16Weights()
      .Test(BuiltinOperator_ADD, webnn_delegate.get());

  BinaryElementwiseTester()
      .Input1Shape({batch, height, width, channels})
      .Input2Shape({batch, height, width, channels})
      .Input2Static(true)
      .FP16Weights()
      .Test(BuiltinOperator_ADD, webnn_delegate.get());
}

TEST(Add, SparseWeights) {
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

  BinaryElementwiseTester()
      .Input1Shape({batch, height, width, channels})
      .Input2Shape({batch, height, width, channels})
      .Input1Static(true)
      .SparseWeights()
      .Test(BuiltinOperator_ADD, webnn_delegate.get());

  BinaryElementwiseTester()
      .Input1Shape({batch, height, width, channels})
      .Input2Shape({batch, height, width, channels})
      .Input2Static(true)
      .SparseWeights()
      .Test(BuiltinOperator_ADD, webnn_delegate.get());
}

TEST(Add, ReluActivation) {
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

  BinaryElementwiseTester()
      .Input1Shape({batch, height, width, channels})
      .Input2Shape({batch, height, width, channels})
      .ReluActivation()
      .Test(BuiltinOperator_ADD, webnn_delegate.get());
}

TEST(Add, Relu6Activation) {
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

  BinaryElementwiseTester()
      .Input1Shape({batch, height, width, channels})
      .Input2Shape({batch, height, width, channels})
      .Relu6Activation()
      .Test(BuiltinOperator_ADD, webnn_delegate.get());
}

TEST(Add, ReluMinus1To1Activation) {
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

  BinaryElementwiseTester()
      .Input1Shape({batch, height, width, channels})
      .Input2Shape({batch, height, width, channels})
      .ReluMinus1To1Activation()
      .Test(BuiltinOperator_ADD, webnn_delegate.get());
}

TEST(Add, DISABLED_TanhActivation) {
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

  BinaryElementwiseTester()
      .Input1Shape({batch, height, width, channels})
      .Input2Shape({batch, height, width, channels})
      .TanhActivation()
      .Test(BuiltinOperator_ADD, webnn_delegate.get());
}

TEST(Add, DISABLED_SignBitActivation) {
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

  BinaryElementwiseTester()
      .Input1Shape({batch, height, width, channels})
      .Input2Shape({batch, height, width, channels})
      .SignBitActivation()
      .Test(BuiltinOperator_ADD, webnn_delegate.get());
}

// TEST(Add, MultiThreading) {
//   TfLiteWebNNDelegateOptions delegate_options =
//       TfLiteWebNNDelegateOptionsDefault();
//   delegate_options.num_threads = 2;
//   std::unique_ptr<TfLiteDelegate, decltype(&TfLiteWebNNDelegateDelete)>
//       webnn_delegate(TfLiteWebNNDelegateCreate(&delegate_options),
//                        TfLiteWebNNDelegateDelete);

//   std::random_device random_device;
//   auto rng = std::mt19937(random_device());
//   auto shape_rng =
//       std::bind(std::uniform_int_distribution<int32_t>(2, 5), std::ref(rng));
//   const auto batch = shape_rng();
//   const auto height = shape_rng();
//   const auto width = shape_rng();
//   const auto channels = shape_rng();

//   BinaryElementwiseTester()
//       .Input1Shape({batch, height, width, channels})
//       .Input2Shape({batch, height, width, channels})
//       .Test(BuiltinOperator_ADD, webnn_delegate.get());
// }

}  // namespace webnn
}  // namespace tflite
