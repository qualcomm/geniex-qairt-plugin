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
    # We compile geniex_core TUs directly into the test exe, so GENIEX_API must
    # resolve to dllexport (definition), not dllimport (consume-from-DLL).
    if(WIN32)
        target_compile_definitions(${NAME} PRIVATE geniex_core_EXPORTS)
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
        ${CMAKE_SOURCE_DIR}/core/src/llm/llm_utils.cpp
        ${CMAKE_SOURCE_DIR}/qnn-api/src/IOTensor.cpp
        ${CMAKE_SOURCE_DIR}/qnn-api/src/ClientBuffer.cpp
        ${CMAKE_SOURCE_DIR}/qnn-api/src/RpcMem.cpp
        ${CMAKE_SOURCE_DIR}/qnn-api/src/dlwrap.cpp
)
