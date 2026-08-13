# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

# Gemma4 example. Links geniex_vlm like the other VLM families, because a Gemma4
# bundle that ships a vision encoder context binary is a VLM — the example picks
# LLMPipeline or VLMPipeline from the bundle's contents, so one binary serves both
# text-only and multimodal bundles.
#
# E2B and E4B share gemma4_example.cpp and gemma4.h verbatim. The two differ only
# in dimensions (hidden 1536 vs 2560, 35 vs 42 layers, per-layer embedding stream
# 8960 vs 10752, 1 vs 2 KV heads), all of which come from the bundle's
# genie_config.json or are inferred from the loaded graphs — so a variant needs
# nothing but a different default bundle name.
set(GEMMA4_VARIANTS
    "gemma4_e2b|Gemma4-E2B"
    "gemma4_e4b|Gemma4-E4B"
)

foreach(VARIANT ${GEMMA4_VARIANTS})
    string(REPLACE "|" ";" VARIANT_PARTS ${VARIANT})
    list(GET VARIANT_PARTS 0 EXAMPLE_NAME)
    list(GET VARIANT_PARTS 1 EXAMPLE_LABEL)

    add_executable(${EXAMPLE_NAME} gemma4_example.cpp)
    target_include_directories(${EXAMPLE_NAME} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
    )
    target_link_libraries(${EXAMPLE_NAME} PRIVATE geniex_vlm geniex-proc-vision)
    target_compile_definitions(${EXAMPLE_NAME} PRIVATE
        GEMMA4_VARIANT="${EXAMPLE_NAME}"
        GEMMA4_VARIANT_LABEL="${EXAMPLE_LABEL}"
    )
    set_target_properties(${EXAMPLE_NAME} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    )
endforeach()
