# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

cmake_minimum_required(VERSION 3.17)
project(geniex)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(MSVC)
    add_compile_options(/utf-8)
endif()

# ── Build options ─────────────────────────────────────────────────────────────
option(GENIEX_BUILD_VLM   "Build geniex_vlm  (VLM/CV models, requires OpenCV)"           OFF)
option(GENIEX_BUILD_EXAMPLES "Build example executables"                                   OFF)
option(GENIEX_BUILD_TESTS "Build and register CTest smoke tests for examples"             OFF)
option(GENIEX_DEBUG       "Enable debug logging (includes file/line/func in log output)"  OFF)
option(GENIEX_COVERAGE    "Instrument unit tests for llvm-cov source coverage"            OFF)

# ── Third-party dependencies ──────────────────────────────────────────────────
if(GENIEX_BUILD_VLM)
    set(GENIEXPROC_ENABLE_VISION ON CACHE BOOL "" FORCE)
endif()
set(GENIEXPROC_BUILD_SHARED_LIBS ON CACHE BOOL "" FORCE)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/third-party/geniex-proc third-party/geniex-proc)

# ── Compile definitions ───────────────────────────────────────────────────────
set(GENIEX_COMPILE_DEFS
    qualla_EXPORTS
    GENIE_API=
    NOMINMAX
    NOGDI
    SPILLFILL
    QUALLA_APPS=OFF
    FMT_HEADER_ONLY=1
    FMT_CONSTEVAL=constexpr
    QUALLA_INTERNAL_QNN_SDK
    QUALLA_ENGINE_QNN_HTP
    GENIE_SSD_FEATURE
    GENIE_SPD_FEATURE
    GENIE_LADE_FEATURE
    GENIE_MULTISTREAM_FEATURE
    GENIE_LORA_FEATURE
    GENIE_E2T_FEATURE
    JSON_HAS_THREE_WAY_COMPARISON=0
)

# Report the QAIRT version this build's ABI targets (consumed by version.h).
list(APPEND GENIEX_COMPILE_DEFS "GENIEX_QAIRT_VERSION=\"${QAIRT_QNN_VERSION}\"")

# ── Debug logging definitions (must come after set(GENIEX_COMPILE_DEFS)) ─────
if(GENIEX_DEBUG)
    list(APPEND GENIEX_COMPILE_DEFS GENIEX_DEBUG)
    list(APPEND GENIEX_COMPILE_DEFS "PROJECT_SOURCE_DIR=\"${CMAKE_SOURCE_DIR}\"")
endif()

# ── QAIRT/QNN version selection ───────────────────────────────────────────────
# This tree ships NO QNN runtime libraries. The plugin is compiled once per
# supported QAIRT version against that version's QNN headers (which fix the
# IBackend vtable + struct ABI); the matching runtime libraries are supplied at
# load time via --qnn-lib / GENIEX_QNN_LIB. QAIRT_QNN_VERSION selects the header
# set under qnn-api/qnn-versions/<ver>/include and suffixes the target names so
# several versions can be built side by side (geniex_core_2_45, _2_47, _2_48 …).
set(QAIRT_QNN_VERSION "2.45" CACHE STRING "QAIRT/QNN version whose headers to compile against")
set(QAIRT_QNN_VERSION_DIR "${CMAKE_CURRENT_SOURCE_DIR}/qnn-api/qnn-versions/${QAIRT_QNN_VERSION}/include")
if(NOT EXISTS "${QAIRT_QNN_VERSION_DIR}/QnnCommon.h")
    message(FATAL_ERROR "QAIRT_QNN_VERSION='${QAIRT_QNN_VERSION}': no headers at ${QAIRT_QNN_VERSION_DIR}")
endif()
# Target-name suffix: 2.45 -> _2_45
string(REPLACE "." "_" QAIRT_VER_SUFFIX "_${QAIRT_QNN_VERSION}")
message(STATUS "geniex-qairt: building against QNN headers ${QAIRT_QNN_VERSION} (suffix '${QAIRT_VER_SUFFIX}')")

# ── Common include paths ──────────────────────────────────────────────────────
# The version-specific header dir MUST come first so its QnnCommon.h / IBackend.hpp
# / IOTensor/IOTensorTypes.hpp shadow anything under qnn-api/src.
set(GENIEX_INCLUDE_DIRS
    ${QAIRT_QNN_VERSION_DIR}
    ${QAIRT_QNN_VERSION_DIR}/HTP
    ${QAIRT_QNN_VERSION_DIR}/System
    ${CMAKE_CURRENT_SOURCE_DIR}/qnn-api/src
    ${CMAKE_CURRENT_SOURCE_DIR}/qnn-api/src/config
    ${CMAKE_CURRENT_SOURCE_DIR}/third-party/qairt/src/qualla/include
    ${CMAKE_CURRENT_SOURCE_DIR}/third-party/qairt/src/trace/include
    ${CMAKE_CURRENT_SOURCE_DIR}/third-party/geniex-proc/include
    ${CMAKE_CURRENT_SOURCE_DIR}/third-party/geniex-proc/third_party/xtensor/include
    ${CMAKE_CURRENT_SOURCE_DIR}/third-party/geniex-proc/third_party/xtl/include
)

# ── qnn-api sources ───────────────────────────────────────────────────────────
file(GLOB_RECURSE QNN_API_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/qnn-api/src/*.cpp")

# Exclude Linux-specific DMA allocator sources on Windows.
if(WIN32)
    list(FILTER QNN_API_SOURCES EXCLUDE REGEX "DmaBufAllocator\.cpp")
endif()

# ── core sources ─────────────────────────────────────────────────────────────
set(CORE_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/core/src/graph.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/core/src/logging.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/core/src/model.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/core/src/model_init.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/core/src/runtime.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/core/src/utils.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/core/src/threadpool.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/core/src/llm/llm_model.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/core/src/llm/llm_spec_loader.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/core/src/llm/llm_utils.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/core/src/llm/input_provider.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/core/src/llm/quantized_lut.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/core/src/llm/speculative_llm_model.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/core/src/llm/eagle_model.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/core/src/llm/eagle_model_init.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/core/src/pipeline/llm_pipeline.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/models/llama3_2_ssd/ssd_model.cpp
)

# ── LLM model specs (header-only; no .cpp to collect) ────────────────────────
# models_ref/llm/ contains only .h files — no glob needed.
# They are exposed via target_include_directories on geniex_core.

# ════════════════════════════════════════════════════════════════════════════
# Tier 1: geniex_core  —  always built
# Deps: QNN API, core, LLM model specs, tokenizers (via geniex-proc)
# ════════════════════════════════════════════════════════════════════════════
add_library(geniex_core SHARED
    ${QNN_API_SOURCES}
    ${CORE_SOURCES}
)

target_include_directories(geniex_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/core/include
    ${CMAKE_CURRENT_SOURCE_DIR}/models
    ${GENIEX_INCLUDE_DIRS}
)

target_compile_definitions(geniex_core PUBLIC ${GENIEX_COMPILE_DEFS})
target_link_libraries(geniex_core PUBLIC geniex-proc)
if(WIN32)
    target_link_libraries(geniex_core PRIVATE Version Shlwapi)
endif()

# ── Workaround: GCC 11 autovec miscompile in graph.cpp ───────────────────────
# core/src/graph.cpp contains hot scalar↔quantized templated cast loops
# (floatToTfN / tfNToFloat / castToFloat / castFromFloat). GCC 11.2's
# tree-loop vectorizer miscompiles them on aarch64-linux-gnu at -O3,
# producing garbage tokens during LLM inference. Bisected to this single TU
# in 2026-05; -O2 and -fno-tree-vectorize both fix it. Scoped to the one
# file and to GCC only — Clang on Windows / Android NDK is unaffected.
#
# Note: applied *after* add_library(geniex_core ...) so the source file is
# already known to the target
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set_source_files_properties(
        ${CMAKE_CURRENT_SOURCE_DIR}/core/src/graph.cpp
        PROPERTIES COMPILE_OPTIONS "-fno-tree-vectorize"
    )
endif()

set_target_properties(geniex_core PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    # On Windows the DLL import lib (.lib) is an ARCHIVE artifact.
    # examples/ links against it as ..\Release\geniex_core.lib, so it
    # must land in build\Release\ (one level above examples\Release\).
    ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)

# ════════════════════════════════════════════════════════════════════════════
# Tier 2: geniex_vlm  —  optional, requires OpenCV / mm-process
# Deps: geniex_core + VLM model subclasses + image preprocessing
# ════════════════════════════════════════════════════════════════════════════
if(GENIEX_BUILD_VLM)
    file(GLOB_RECURSE VLM_SOURCES
        "${CMAKE_CURRENT_SOURCE_DIR}/core/src/vlm/*.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/core/src/pipeline/vlm_pipeline.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/models/qwen2_5_vl/qwen2_5_vl.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/models/qwen3_vl/qwen3_vl.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/models/intern3_5_vl/intern3_5_vl.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/models/gemma4/gemma4_vlm.cpp"
    )

    add_library(geniex_vlm SHARED ${VLM_SOURCES})

    target_include_directories(geniex_vlm PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/models
    )

    target_link_libraries(geniex_vlm PUBLIC geniex_core geniex-proc-vision)

    set_target_properties(geniex_vlm PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    )
endif()

# ════════════════════════════════════════════════════════════════════════════
# Tier 3: geniex_audio  —  optional, requires libsndfile / soxr / fftw3
# Deps: geniex_core + audio model subclasses
# ════════════════════════════════════════════════════════════════════════════
if(GENIEX_BUILD_AUDIO)
    file(GLOB_RECURSE AUDIO_SOURCES
        "${CMAKE_CURRENT_SOURCE_DIR}/models_ref/audio/*.cpp"
    )

    add_library(geniex_audio SHARED ${AUDIO_SOURCES})

    target_include_directories(geniex_audio PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/models_ref/audio
    )

    target_link_libraries(geniex_audio PUBLIC geniex_core geniex-proc)

    set_target_properties(geniex_audio PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    )
endif()

# ════════════════════════════════════════════════════════════════════════════
# Tier 4: geniex_ocr  —  optional, requires OpenCV + clipper2
# Deps: geniex_core + OCR model subclasses
# ════════════════════════════════════════════════════════════════════════════
if(GENIEX_BUILD_OCR)
    file(GLOB_RECURSE OCR_SOURCES
        "${CMAKE_CURRENT_SOURCE_DIR}/models_ref/ocr/*.cpp"
    )

    add_library(geniex_ocr SHARED ${OCR_SOURCES})

    target_include_directories(geniex_ocr PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/models_ref/ocr
    )

    target_link_libraries(geniex_ocr PUBLIC geniex_core geniex-proc)

    set_target_properties(geniex_ocr PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    )
endif()

# ── examples (one per model folder) ──────────────────────────────────────────
if(GENIEX_BUILD_EXAMPLES)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/models/phi3_5)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/models/qwen3)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/models/qwen2_5)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/models/llama3)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/models/llama3_2_ssd)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/models/qwen3_eaglet)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/models/falcon3)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/models/gemma4)

    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/examples)
endif()

if(GENIEX_BUILD_VLM)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/models/qwen2_5_vl)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/models/qwen3_vl)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/models/intern3_5_vl)
endif()

# ── Tests (CTest generation smoke tests; require NPU hardware) ───────────────
if(GENIEX_BUILD_TESTS)
    enable_testing()
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/tests)
endif()

# ── HTP runtime libraries ─────────────────────────────────────────────────────
# This tree bundles NO QNN/HTP runtime libraries. They are supplied at load time
# via --qnn-lib / GENIEX_QNN_LIB, which points at a QAIRT SDK of the version this
# plugin was compiled against (QAIRT_QNN_VERSION). There is therefore no
# htp-files/ POST_BUILD copy and no htp_runtime install component.
#
# On Windows the DLL goes to bin/; on other platforms to lib/.
if(WIN32)
    set(GENIEX_RUNTIME_INSTALL_DIR "bin")
else()
    set(GENIEX_RUNTIME_INSTALL_DIR "lib")
endif()

# Install geniex_core itself.
install(TARGETS geniex_core
    RUNTIME DESTINATION "${GENIEX_RUNTIME_INSTALL_DIR}"  # .dll on Windows
    LIBRARY DESTINATION "lib"                             # .so on Linux/Android
    ARCHIVE DESTINATION "lib"                             # .lib import lib
)

# Install public headers.
install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/core/include/"
    DESTINATION "include/geniex"
)

