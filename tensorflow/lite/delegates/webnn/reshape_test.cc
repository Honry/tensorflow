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

#include <algorithm>
#include <cstdint>
#include <functional>
#include <memory>
#include <random>

#include <gtest/gtest.h>
#include "tensorflow/lite/delegates/webnn/reshape_tester.h"
#include "tensorflow/lite/delegates/webnn/webnn_delegate.h"

namespace tflite {
namespace webnn {

TEST(Reshape, 4DShapeAsInput) {
  TfLiteWebNNDelegateOptions delegate_options =
      TfLiteWebNNDelegateOptionsDefault();
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteWebNNDelegateDelete)>
      webnn_delegate(TfLiteWebNNDelegateCreate(&delegate_options),
                     TfLiteWebNNDelegateDelete);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto shape_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 10), std::ref(rng));
  const std::vector<int32_t> input_shape{
      {shape_rng(), shape_rng(), shape_rng(), shape_rng()}};
  std::vector<int32_t> output_shape(input_shape.cbegin(), input_shape.cend());
  std::shuffle(output_shape.begin(), output_shape.end(), rng);

  ReshapeTester()
      .InputShape(input_shape)
      .OutputShape(output_shape)
      .OutputShapeAsInput(true)
      .Test(webnn_delegate.get());
}

TEST(Reshape, 4DShapeAsParam) {
  TfLiteWebNNDelegateOptions delegate_options =
      TfLiteWebNNDelegateOptionsDefault();
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteWebNNDelegateDelete)>
      webnn_delegate(TfLiteWebNNDelegateCreate(&delegate_options),
                     TfLiteWebNNDelegateDelete);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto shape_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 10), std::ref(rng));
  const std::vector<int32_t> input_shape{
      {shape_rng(), shape_rng(), shape_rng(), shape_rng()}};
  std::vector<int32_t> output_shape(input_shape.cbegin(), input_shape.cend());
  std::shuffle(output_shape.begin(), output_shape.end(), rng);

  ReshapeTester()
      .InputShape(input_shape)
      .OutputShape(output_shape)
      .OutputShapeAsInput(false)
      .Test(webnn_delegate.get());
}

TEST(Reshape, 3DShapeAsInput) {
  TfLiteWebNNDelegateOptions delegate_options =
      TfLiteWebNNDelegateOptionsDefault();
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteWebNNDelegateDelete)>
      webnn_delegate(TfLiteWebNNDelegateCreate(&delegate_options),
                     TfLiteWebNNDelegateDelete);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto shape_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 10), std::ref(rng));
  const std::vector<int32_t> input_shape{
      {shape_rng(), shape_rng(), shape_rng()}};
  std::vector<int32_t> output_shape(input_shape.cbegin(), input_shape.cend());
  std::shuffle(output_shape.begin(), output_shape.end(), rng);

  ReshapeTester()
      .InputShape(input_shape)
      .OutputShape(output_shape)
      .OutputShapeAsInput(true)
      .Test(webnn_delegate.get());
}

TEST(Reshape, 3DShapeAsParam) {
  TfLiteWebNNDelegateOptions delegate_options =
      TfLiteWebNNDelegateOptionsDefault();
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteWebNNDelegateDelete)>
      webnn_delegate(TfLiteWebNNDelegateCreate(&delegate_options),
                     TfLiteWebNNDelegateDelete);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto shape_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 10), std::ref(rng));
  const std::vector<int32_t> input_shape{
      {shape_rng(), shape_rng(), shape_rng()}};
  std::vector<int32_t> output_shape(input_shape.cbegin(), input_shape.cend());
  std::shuffle(output_shape.begin(), output_shape.end(), rng);

  ReshapeTester()
      .InputShape(input_shape)
      .OutputShape(output_shape)
      .OutputShapeAsInput(false)
      .Test(webnn_delegate.get());
}

TEST(Reshape, 2DShapeAsInput) {
  TfLiteWebNNDelegateOptions delegate_options =
      TfLiteWebNNDelegateOptionsDefault();
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteWebNNDelegateDelete)>
      webnn_delegate(TfLiteWebNNDelegateCreate(&delegate_options),
                     TfLiteWebNNDelegateDelete);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto shape_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 10), std::ref(rng));
  const std::vector<int32_t> input_shape{{shape_rng(), shape_rng()}};
  std::vector<int32_t> output_shape(input_shape.cbegin(), input_shape.cend());
  std::shuffle(output_shape.begin(), output_shape.end(), rng);

  ReshapeTester()
      .InputShape(input_shape)
      .OutputShape(output_shape)
      .OutputShapeAsInput(true)
      .Test(webnn_delegate.get());
}

TEST(Reshape, 2DShapeAsParam) {
  TfLiteWebNNDelegateOptions delegate_options =
      TfLiteWebNNDelegateOptionsDefault();
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteWebNNDelegateDelete)>
      webnn_delegate(TfLiteWebNNDelegateCreate(&delegate_options),
                     TfLiteWebNNDelegateDelete);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto shape_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 10), std::ref(rng));
  const std::vector<int32_t> input_shape{{shape_rng(), shape_rng()}};
  std::vector<int32_t> output_shape(input_shape.cbegin(), input_shape.cend());
  std::shuffle(output_shape.begin(), output_shape.end(), rng);

  ReshapeTester()
      .InputShape(input_shape)
      .OutputShape(output_shape)
      .OutputShapeAsInput(false)
      .Test(webnn_delegate.get());
}

TEST(Reshape, 1DShapeAsInput) {
  TfLiteWebNNDelegateOptions delegate_options =
      TfLiteWebNNDelegateOptionsDefault();
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteWebNNDelegateDelete)>
      webnn_delegate(TfLiteWebNNDelegateCreate(&delegate_options),
                     TfLiteWebNNDelegateDelete);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto shape_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 10), std::ref(rng));
  const std::vector<int32_t> shape({shape_rng()});

  ReshapeTester()
      .InputShape(shape)
      .OutputShape(shape)
      .OutputShapeAsInput(true)
      .Test(webnn_delegate.get());
}

TEST(Reshape, 1DShapeAsParam) {
  TfLiteWebNNDelegateOptions delegate_options =
      TfLiteWebNNDelegateOptionsDefault();
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteWebNNDelegateDelete)>
      webnn_delegate(TfLiteWebNNDelegateCreate(&delegate_options),
                     TfLiteWebNNDelegateDelete);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto shape_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 10), std::ref(rng));
  const std::vector<int32_t> shape({shape_rng()});

  ReshapeTester()
      .InputShape(shape)
      .OutputShape(shape)
      .OutputShapeAsInput(false)
      .Test(webnn_delegate.get());
}

TEST(Reshape, 0D) {
  TfLiteWebNNDelegateOptions delegate_options =
      TfLiteWebNNDelegateOptionsDefault();
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteWebNNDelegateDelete)>
      webnn_delegate(TfLiteWebNNDelegateCreate(&delegate_options),
                     TfLiteWebNNDelegateDelete);

  ReshapeTester()
      .InputShape(std::vector<int32_t>())
      .OutputShape(std::vector<int32_t>())
      .Test(webnn_delegate.get());
}

}  // namespace webnn
}  // namespace tflite
