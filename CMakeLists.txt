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
option(GENIEX_DEBUG       "Enable debug logging (includes file/line/func in log output)"  OFF)

# ── Third-party dependencies ──────────────────────────────────────────────────
if(GENIEX_BUILD_VLM)
    set(GENIEXPROC_ENABLE_VISION ON CACHE BOOL "" FORCE)
endif()
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

# ── Debug logging definitions (must come after set(GENIEX_COMPILE_DEFS)) ─────
if(GENIEX_DEBUG)
    list(APPEND GENIEX_COMPILE_DEFS GENIEX_DEBUG)
    list(APPEND GENIEX_COMPILE_DEFS "PROJECT_SOURCE_DIR=\"${CMAKE_SOURCE_DIR}\"")
endif()

# ── Common include paths ──────────────────────────────────────────────────────
set(GENIEX_INCLUDE_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}/qnn-api/include
    ${CMAKE_CURRENT_SOURCE_DIR}/qnn-api/include/HTP
    ${CMAKE_CURRENT_SOURCE_DIR}/qnn-api/include/System
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
    ${CMAKE_CURRENT_SOURCE_DIR}/core/src/runtime.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/core/src/utils.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/core/src/llm/llm_model.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/core/src/llm/llm_utils.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/core/src/llm/input_provider.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/core/src/pipeline/chat_template.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/core/src/pipeline/llm_pipeline.cpp
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
target_include_directories(geniex_core PUBLIC
    $<TARGET_PROPERTY:geniex-proc,INTERFACE_INCLUDE_DIRECTORIES>)
target_link_libraries(geniex_core PRIVATE geniex-proc)
if(WIN32)
    target_link_libraries(geniex_core PRIVATE Version Shlwapi)
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
        "${CMAKE_CURRENT_SOURCE_DIR}/models_ref/vlm/*.cpp"
    )

    add_library(geniex_vlm SHARED ${VLM_SOURCES})

    target_include_directories(geniex_vlm PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/models_ref/vlm
    )

    target_link_libraries(geniex_vlm PUBLIC geniex_core PRIVATE geniex-proc)

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
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/models/llama3_2)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/models/llama3_2_ssd)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/models/llama3_1)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/models/falcon3)

    # Continuous-batching demos (see examples/continuous_batching/README.md).
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/examples/continuous_batching)
endif()

if(GENIEX_BUILD_VLM)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/models/qwen2_5_vl)
endif()

# ── HTP runtime libraries ─────────────────────────────────────────────────────
# Each platform has a single HTP runtime folder under third-party/:
#   third-party/windows/        — Windows (all supported arch stubs bundled)
#   third-party/android/        — Android (arm64-v8a)
#   third-party/linux-gcc11.2/  — Linux (IQ-9075, gcc 11.2)
#
# The folder is:
#   • Copied into the build-tree output dir as htp-files/ (POST_BUILD) so
#     examples work without running cmake --install.
#   • Installed next to geniex_core as htp-files/ via install() so that
#     downstream deployments get the layout expected by runtime.h.
#
# runtime.h looks for htp-files/ relative to the geniex_core DLL/SO.

if(WIN32)
    set(GENIEX_HTP_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third-party/windows")
elseif(ANDROID)
    set(GENIEX_HTP_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third-party/android")
elseif(UNIX)
    set(GENIEX_HTP_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third-party/linux-gcc11.2")
endif()

if(DEFINED GENIEX_HTP_DIR)
    # On Windows the DLL goes to bin/; on other platforms to lib/.
    if(WIN32)
        set(GENIEX_RUNTIME_INSTALL_DIR "bin")
    else()
        set(GENIEX_RUNTIME_INSTALL_DIR "lib")
    endif()

    # Copy into build tree after every geniex_core build so examples run
    # directly from build/bin/Release/ without a separate install step.
    add_custom_command(TARGET geniex_core POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
            "${GENIEX_HTP_DIR}"
            "$<TARGET_FILE_DIR:geniex_core>/htp-files"
        COMMENT "Copying HTP runtime libs to build output dir (htp-files/)"
    )

    # Install alongside geniex_core for deployment / downstream use.
    install(DIRECTORY "${GENIEX_HTP_DIR}/"
        DESTINATION "${GENIEX_RUNTIME_INSTALL_DIR}/htp-files"
        COMPONENT htp_runtime
    )

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
endif()

