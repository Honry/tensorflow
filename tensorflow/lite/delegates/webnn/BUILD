load("//tensorflow/lite:special_rules.bzl", "tflite_portable_test_suite_combined")
load("//tensorflow:tensorflow.bzl", "get_compatible_with_portable")
load("//tensorflow/lite:build_def.bzl", "tflite_cc_shared_object")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

EMSCRIPTEN_LINKOPTS = [
    "-s ASSERTIONS=2",
    "-s ERROR_ON_UNDEFINED_SYMBOLS=1",
    "-s DEMANGLE_SUPPORT=1",
    "-s EXIT_RUNTIME=1",
    "-s ALLOW_MEMORY_GROWTH=1",
    "-s TOTAL_MEMORY=134217728",
]

exports_files([
    "webnn_delegate.h",
])

cc_library(
    name = "webnn_delegate",
    srcs = ["webnn_delegate.cc"],
    hdrs = ["webnn_delegate.h"],
    linkstatic = True,
    deps = [
        "//tensorflow/lite:kernel_api",
        "//tensorflow/lite:minimal_logging",
        "//tensorflow/lite:util",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/schema:schema_fbs",
        "//tensorflow/lite/kernels/internal/utils:sparsity_format_converter",
        "@webnn_native_project//:webnn-native",
        "@FP16",
    ],
)

cc_library(
    name = "webnn_delegate_adaptor",
    srcs = ["webnn_delegate_adaptor.cc"],
    hdrs =  ["webnn_delegate_adaptor.h"],
    deps = [
        ":webnn_delegate",
        "//tensorflow/lite:shared_library",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/tools:command_line_flags",
        "//tensorflow/lite/tools:logging",
    ],
)

tflite_cc_shared_object(
    name = "webnn_external_delegate_obj",
    linkopts = select({
        "//tensorflow:windows": [
            "-Wl,-exported_symbols_list,$(location //tensorflow/lite/c:exported_symbols.lds)",
        ],
        "//conditions:default": [
            "-Wl,-z,defs",
            "-Wl,--version-script,$(location //tensorflow/lite:tflite_version_script.lds)",
        ],
    }),
    per_os_targets = True,
    srcs = select({
        "//tensorflow:windows": [
            "webnn_delegate_adaptor.cc",
        ],
        "//conditions:default": [],
    }),
    deps = [
        ":webnn_delegate_adaptor",
        "//tensorflow/lite/c:exported_symbols.lds",
        "//tensorflow/lite:tflite_version_script.lds",
    ],
)

cc_library(
    name = "webnn_delegate_hdrs_only",
    hdrs = ["webnn_delegate.h"],
    compatible_with = get_compatible_with_portable(),
    visibility = ["//tensorflow/lite:__subpackages__"],
    deps = [
        "//tensorflow/lite/c:common",
    ],
)

cc_library(
    name = "webnn_delegate_test_mode",
    srcs = ["webnn_delegate.cc"],
    hdrs = ["webnn_delegate.h"],
    copts = ["-DWEBNN_DELEGATE_TEST_MODE=1"],
    linkstatic = True,
    deps = [
        "//tensorflow/lite:kernel_api",
        "//tensorflow/lite:minimal_logging",
        "//tensorflow/lite:util",
        "//tensorflow/lite:simple_memory_arena_debug_dump",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/schema:schema_fbs",
        "//tensorflow/lite/kernels/internal/utils:sparsity_format_converter",
        "//tensorflow/lite/kernels:padding",
        "//tensorflow/lite/kernels/internal:compatibility",
        "//tensorflow/lite/kernels/internal:tensor",
        "//tensorflow/lite/tools/optimize:reduced_precision_support",
        "@webnn_native_project//:webnn-native",
        "@FP16",
    ],
)

################################ Tester classes ################################

cc_library(
    name = "binary_elementwise_tester",
    testonly = 1,
    srcs = ["binary_elementwise_tester.cc"],
    hdrs = ["binary_elementwise_tester.h"],
    deps = [
        "//tensorflow/lite:framework",
        "//tensorflow/lite:schema_fbs_version",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/kernels:builtin_ops",
        "//tensorflow/lite/schema:schema_conversion_utils",
        "//tensorflow/lite/schema:schema_fbs",
        "@FP16",
        "@com_google_googletest//:gtest",
        "@flatbuffers",
    ],
)

############################## Integration tests ###############################

cc_library(
    name = "test_main",
    testonly = 1,
    linkopts = select({
        "//tensorflow:emscripten": EMSCRIPTEN_LINKOPTS,
        "//conditions:default": [],
    }),
    deps = [
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "add_test",
    srcs = ["add_test.cc"],
    linkopts = select({
        "//tensorflow:emscripten": EMSCRIPTEN_LINKOPTS,
        "//conditions:default": [],
    }),
    deps = [
        ":binary_elementwise_tester",
        ":test_main",
        ":webnn_delegate_test_mode",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "delegate_test",
    srcs = ["delegate_test.cc"],
    linkopts = select({
        "//tensorflow:emscripten": EMSCRIPTEN_LINKOPTS,
        "//conditions:default": [],
    }),
    deps = [
        ":test_main",
        ":webnn_delegate_test_mode",
        "@com_google_googletest//:gtest",
    ],
)