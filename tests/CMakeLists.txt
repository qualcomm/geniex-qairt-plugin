# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

# CPU-only unit tests for geniex-qairt. Enabled with -DGENIEX_BUILD_TESTS=ON.
# Layout mirrors core/src/; shared fixtures + doubles live in tests/testing/.

include(CTest)
include(FetchContent)

# ── GoogleTest ───────────────────────────────────────────────────────────────
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        v1.15.2
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)  # match MSVC shared CRT
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)
include(GoogleTest)  # provides gtest_discover_tests()

# ── Coverage instrumentation (opt-in: -DGENIEX_COVERAGE=ON) ──────────────────
# Applied only to the test targets, which compile core/src directly, so the
# instrumented region is exactly the code under test. clang-cl only.
if(GENIEX_COVERAGE)
    if(NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        message(FATAL_ERROR
            "GENIEX_COVERAGE requires a Clang toolchain. Re-run cmake with "
            "-T ClangCL (Visual Studio generator) or a clang/clang-cl compiler.")
    endif()
    # clang-cl needs the flags forwarded via /clang:; plain clang takes them
    # directly. The linker needs them too, to emit the profile runtime.
    if(MSVC)  # clang-cl presents an MSVC-style driver
        set(GENIEX_COVERAGE_COMPILE_FLAGS
            /clang:-fprofile-instr-generate /clang:-fcoverage-mapping)
        set(GENIEX_COVERAGE_LINK_FLAGS -fprofile-instr-generate)
    else()
        set(GENIEX_COVERAGE_COMPILE_FLAGS
            -fprofile-instr-generate -fcoverage-mapping)
        set(GENIEX_COVERAGE_LINK_FLAGS -fprofile-instr-generate)
    endif()
endif()

# ── Helper: register one unit-test executable ────────────────────────────────
# geniex_add_unit_test(<name> SOURCES <a.cpp>... [LIBS <lib>...])
function(geniex_add_unit_test NAME)
    cmake_parse_arguments(ARG "" "" "SOURCES;LIBS" ${ARGN})
    add_executable(${NAME} ${ARG_SOURCES})
    target_link_libraries(${NAME} PRIVATE GTest::gtest_main ${ARG_LIBS})
    target_include_directories(${NAME} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/core/include
        ${GENIEX_INCLUDE_DIRS}
        # Header-only transitive deps from core/include/types.h. We do NOT link
        # geniex-proc: its tokenizers-cpp/sentencepiece/abseil build tree blows
        # past the Windows 260-char MAX_PATH limit.
        ${CMAKE_SOURCE_DIR}/third-party/geniex-proc/include
        ${CMAKE_SOURCE_DIR}/third-party/geniex-proc/third-party/xtensor/include
        ${CMAKE_SOURCE_DIR}/third-party/geniex-proc/third-party/xtl/include
    )
    target_compile_definitions(${NAME} PRIVATE ${GENIEX_COMPILE_DEFS})
    # We compile geniex_core/geniex_vlm TUs directly into the test exe, so the
    # export macros must resolve to dllexport (definition), not dllimport.
    if(WIN32)
        target_compile_definitions(${NAME} PRIVATE geniex_core_EXPORTS geniex_vlm_EXPORTS)
    endif()
    if(GENIEX_COVERAGE)
        target_compile_options(${NAME} PRIVATE ${GENIEX_COVERAGE_COMPILE_FLAGS})
        target_link_options(${NAME} PRIVATE ${GENIEX_COVERAGE_LINK_FLAGS})
    endif()
    set_target_properties(${NAME} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    )
    gtest_discover_tests(${NAME}
        # PRE_TEST avoids a GoogleTestAddTests bug that mis-splits VS install
        # paths containing spaces ("Microsoft Visual Studio") at build time.
        DISCOVERY_MODE PRE_TEST
        PROPERTIES
            LABELS  "unit;cpu"
            TIMEOUT 60
    )
endfunction()

# ── Pure-logic suites (no test doubles) ──────────────────────────────────────
geniex_add_unit_test(utils_test
    SOURCES
        core/utils_test.cpp
        ${CMAKE_SOURCE_DIR}/core/src/utils.cpp
        ${CMAKE_SOURCE_DIR}/core/src/logging.cpp
)

# ── Graph round-trips (real Graph + link-time QnnApi stub) ───────────────────
geniex_add_unit_test(graph_test
    SOURCES
        core/graph_test.cpp
        testing/stub_qnnapi.cpp
        ${CMAKE_SOURCE_DIR}/core/src/graph.cpp
        ${CMAKE_SOURCE_DIR}/core/src/utils.cpp
        ${CMAKE_SOURCE_DIR}/core/src/logging.cpp
        ${CMAKE_SOURCE_DIR}/qnn-api/src/IOTensor.cpp
        ${CMAKE_SOURCE_DIR}/qnn-api/src/ClientBuffer.cpp
        ${CMAKE_SOURCE_DIR}/qnn-api/src/RpcMem.cpp
        ${CMAKE_SOURCE_DIR}/qnn-api/src/dlwrap.cpp
)

# ── LLM CPU-side input providers (real Graph + link-time QnnApi stub) ─────────
geniex_add_unit_test(input_provider_test
    SOURCES
        core/llm/input_provider_test.cpp
        testing/stub_qnnapi.cpp
        ${CMAKE_SOURCE_DIR}/core/src/graph.cpp
        ${CMAKE_SOURCE_DIR}/core/src/utils.cpp
        ${CMAKE_SOURCE_DIR}/core/src/logging.cpp
        ${CMAKE_SOURCE_DIR}/core/src/llm/input_provider.cpp
        ${CMAKE_SOURCE_DIR}/core/src/llm/quantized_lut.cpp
        ${CMAKE_SOURCE_DIR}/core/src/llm/llm_utils.cpp
        ${CMAKE_SOURCE_DIR}/qnn-api/src/IOTensor.cpp
        ${CMAKE_SOURCE_DIR}/qnn-api/src/ClientBuffer.cpp
        ${CMAKE_SOURCE_DIR}/qnn-api/src/RpcMem.cpp
        ${CMAKE_SOURCE_DIR}/qnn-api/src/dlwrap.cpp
)

# ── LLM pure-math utilities (RoPE variants, masking, tensor helpers) ─────────
geniex_add_unit_test(llm_utils_test
    SOURCES
        core/llm/llm_utils_test.cpp
        ${CMAKE_SOURCE_DIR}/core/src/llm/llm_utils.cpp
        ${CMAKE_SOURCE_DIR}/core/src/utils.cpp
        ${CMAKE_SOURCE_DIR}/core/src/logging.cpp
)

# ── Model accessors + applyConnections (real Graph + link-time QnnApi stub) ──
# model_init.cpp is included so Model::initialize() links (it is never called
# by model_test). stub_qnnapi.cpp satisfies all QNN SDK symbols it references.
# model_init.cpp reads the bundle core count via llm/llm_spec_loader.h, which
# pulls the spec-loader/provider/model TUs into the link (same set as
# llm_spec_loader_test below).
geniex_add_unit_test(model_test
    SOURCES
        core/model_test.cpp
        testing/stub_qnnapi.cpp
        ${CMAKE_SOURCE_DIR}/core/src/model.cpp
        ${CMAKE_SOURCE_DIR}/core/src/model_init.cpp
        ${CMAKE_SOURCE_DIR}/core/src/runtime.cpp
        ${CMAKE_SOURCE_DIR}/core/src/graph.cpp
        ${CMAKE_SOURCE_DIR}/core/src/utils.cpp
        ${CMAKE_SOURCE_DIR}/core/src/logging.cpp
        ${CMAKE_SOURCE_DIR}/core/src/threadpool.cpp
        ${CMAKE_SOURCE_DIR}/core/src/llm/llm_model.cpp
        ${CMAKE_SOURCE_DIR}/core/src/llm/llm_utils.cpp
        ${CMAKE_SOURCE_DIR}/core/src/llm/llm_spec_loader.cpp
        ${CMAKE_SOURCE_DIR}/core/src/llm/input_provider.cpp
        ${CMAKE_SOURCE_DIR}/core/src/llm/quantized_lut.cpp
        ${CMAKE_SOURCE_DIR}/qnn-api/src/IOTensor.cpp
        ${CMAKE_SOURCE_DIR}/qnn-api/src/ClientBuffer.cpp
        ${CMAKE_SOURCE_DIR}/qnn-api/src/RpcMem.cpp
        ${CMAKE_SOURCE_DIR}/qnn-api/src/dlwrap.cpp
        ${CMAKE_SOURCE_DIR}/qnn-api/src/qnn-utils.cpp
    LIBS
        geniex-proc
)

# ── LLM bundle/spec loading (parseHtpCoreCount, modelConfigFromDirectory) ────
# llm_spec_loader.cpp pulls the provider/model TUs, so the source list matches
# llm_model_test minus its orchestration test file.
geniex_add_unit_test(llm_spec_loader_test
    SOURCES
        core/llm/llm_spec_loader_test.cpp
        testing/stub_qnnapi.cpp
        ${CMAKE_SOURCE_DIR}/core/src/graph.cpp
        ${CMAKE_SOURCE_DIR}/core/src/utils.cpp
        ${CMAKE_SOURCE_DIR}/core/src/logging.cpp
        ${CMAKE_SOURCE_DIR}/core/src/model.cpp
        ${CMAKE_SOURCE_DIR}/core/src/model_init.cpp
        ${CMAKE_SOURCE_DIR}/core/src/runtime.cpp
        ${CMAKE_SOURCE_DIR}/core/src/threadpool.cpp
        ${CMAKE_SOURCE_DIR}/core/src/llm/llm_model.cpp
        ${CMAKE_SOURCE_DIR}/core/src/llm/llm_utils.cpp
        ${CMAKE_SOURCE_DIR}/core/src/llm/llm_spec_loader.cpp
        ${CMAKE_SOURCE_DIR}/core/src/llm/input_provider.cpp
        ${CMAKE_SOURCE_DIR}/core/src/llm/quantized_lut.cpp
        ${CMAKE_SOURCE_DIR}/qnn-api/src/IOTensor.cpp
        ${CMAKE_SOURCE_DIR}/qnn-api/src/ClientBuffer.cpp
        ${CMAKE_SOURCE_DIR}/qnn-api/src/RpcMem.cpp
        ${CMAKE_SOURCE_DIR}/qnn-api/src/dlwrap.cpp
        ${CMAKE_SOURCE_DIR}/qnn-api/src/qnn-utils.cpp
    LIBS
        geniex-proc
)

# ── LLM prefill/decode orchestration (real LLMModel + link-time QnnApi stub) ──
# model_init.cpp included for link completeness; never called. See model_test note above.
geniex_add_unit_test(llm_model_test
    SOURCES
        core/llm/llm_model_test.cpp
        testing/stub_qnnapi.cpp
        ${CMAKE_SOURCE_DIR}/core/src/graph.cpp
        ${CMAKE_SOURCE_DIR}/core/src/utils.cpp
        ${CMAKE_SOURCE_DIR}/core/src/logging.cpp
        ${CMAKE_SOURCE_DIR}/core/src/model.cpp
        ${CMAKE_SOURCE_DIR}/core/src/model_init.cpp
        ${CMAKE_SOURCE_DIR}/core/src/runtime.cpp
        ${CMAKE_SOURCE_DIR}/core/src/threadpool.cpp
        ${CMAKE_SOURCE_DIR}/core/src/llm/llm_model.cpp
        ${CMAKE_SOURCE_DIR}/core/src/llm/llm_utils.cpp
        ${CMAKE_SOURCE_DIR}/core/src/llm/llm_spec_loader.cpp
        ${CMAKE_SOURCE_DIR}/core/src/llm/input_provider.cpp
        ${CMAKE_SOURCE_DIR}/core/src/llm/quantized_lut.cpp
        ${CMAKE_SOURCE_DIR}/qnn-api/src/IOTensor.cpp
        ${CMAKE_SOURCE_DIR}/qnn-api/src/ClientBuffer.cpp
        ${CMAKE_SOURCE_DIR}/qnn-api/src/RpcMem.cpp
        ${CMAKE_SOURCE_DIR}/qnn-api/src/dlwrap.cpp
        ${CMAKE_SOURCE_DIR}/qnn-api/src/qnn-utils.cpp
    LIBS
        geniex-proc
)

# ── VLM pure-math utilities (no test doubles) ────────────────────────────────
geniex_add_unit_test(vit_utils_test
    SOURCES
        core/vlm/vit_utils_test.cpp
        ${CMAKE_SOURCE_DIR}/core/src/vlm/vit_utils.cpp
)

geniex_add_unit_test(vlm_utils_test
    SOURCES
        core/vlm/vlm_utils_test.cpp
        ${CMAKE_SOURCE_DIR}/core/src/vlm/vlm_utils.cpp
)

# ── VLM input providers (real Graph + link-time QnnApi stub) ─────────────────
geniex_add_unit_test(vlm_input_provider_test
    SOURCES
        core/vlm/vlm_input_provider_test.cpp
        testing/stub_qnnapi.cpp
        ${CMAKE_SOURCE_DIR}/core/src/vlm/vlm_input_provider.cpp
        ${CMAKE_SOURCE_DIR}/core/src/llm/input_provider.cpp
        ${CMAKE_SOURCE_DIR}/core/src/llm/quantized_lut.cpp
        ${CMAKE_SOURCE_DIR}/core/src/llm/llm_utils.cpp
        ${CMAKE_SOURCE_DIR}/core/src/graph.cpp
        ${CMAKE_SOURCE_DIR}/core/src/utils.cpp
        ${CMAKE_SOURCE_DIR}/core/src/logging.cpp
        ${CMAKE_SOURCE_DIR}/qnn-api/src/IOTensor.cpp
        ${CMAKE_SOURCE_DIR}/qnn-api/src/ClientBuffer.cpp
        ${CMAKE_SOURCE_DIR}/qnn-api/src/RpcMem.cpp
        ${CMAKE_SOURCE_DIR}/qnn-api/src/dlwrap.cpp
    LIBS
        geniex-proc
)

# ── VLM prefill/decode orchestration (real VLMModel + link-time QnnApi stub) ──
# model_init.cpp included for link completeness; never called. See model_test note above.
geniex_add_unit_test(vlm_model_test
    SOURCES
        core/vlm/vlm_model_test.cpp
        testing/stub_qnnapi.cpp
        ${CMAKE_SOURCE_DIR}/core/src/vlm/vlm_model.cpp
        ${CMAKE_SOURCE_DIR}/core/src/vlm/vlm_input_provider.cpp
        ${CMAKE_SOURCE_DIR}/core/src/vlm/vision_encoder.cpp
        ${CMAKE_SOURCE_DIR}/core/src/graph.cpp
        ${CMAKE_SOURCE_DIR}/core/src/utils.cpp
        ${CMAKE_SOURCE_DIR}/core/src/logging.cpp
        ${CMAKE_SOURCE_DIR}/core/src/model.cpp
        ${CMAKE_SOURCE_DIR}/core/src/model_init.cpp
        ${CMAKE_SOURCE_DIR}/core/src/runtime.cpp
        ${CMAKE_SOURCE_DIR}/core/src/threadpool.cpp
        ${CMAKE_SOURCE_DIR}/core/src/llm/llm_model.cpp
        ${CMAKE_SOURCE_DIR}/core/src/llm/llm_utils.cpp
        ${CMAKE_SOURCE_DIR}/core/src/llm/llm_spec_loader.cpp
        ${CMAKE_SOURCE_DIR}/core/src/llm/input_provider.cpp
        ${CMAKE_SOURCE_DIR}/core/src/llm/quantized_lut.cpp
        ${CMAKE_SOURCE_DIR}/qnn-api/src/IOTensor.cpp
        ${CMAKE_SOURCE_DIR}/qnn-api/src/ClientBuffer.cpp
        ${CMAKE_SOURCE_DIR}/qnn-api/src/RpcMem.cpp
        ${CMAKE_SOURCE_DIR}/qnn-api/src/dlwrap.cpp
        ${CMAKE_SOURCE_DIR}/qnn-api/src/qnn-utils.cpp
    LIBS
        geniex-proc
)

