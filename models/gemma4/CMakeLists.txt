# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

# E2B and E4B share gemma4_example.cpp and gemma4.h verbatim. The two differ
# only in dimensions (hidden 1536 vs 2560, 35 vs 42 layers, per-layer embedding
# stream 8960 vs 10752, 1 vs 2 KV heads), all of which come from the bundle's
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
    target_link_libraries(${EXAMPLE_NAME} PRIVATE geniex_core geniex-proc)
    target_compile_definitions(${EXAMPLE_NAME} PRIVATE
        GEMMA4_VARIANT="${EXAMPLE_NAME}"
        GEMMA4_VARIANT_LABEL="${EXAMPLE_LABEL}"
    )
    set_target_properties(${EXAMPLE_NAME} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    )
endforeach()

# Vision-dependent targets. geniex-proc-vision carries the Gemma4 image
# processor, so these only exist when the VLM tier is enabled.
if(GENIEX_BUILD_VLM)
    # Numerical check of the C++ preprocessing against the transformers reference.
    add_executable(gemma4_prep_check gemma4_prep_check.cpp)
    target_link_libraries(gemma4_prep_check PRIVATE geniex_core geniex-proc geniex-proc-vision)
    set_target_properties(gemma4_prep_check PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    )

    # Full VLM: image -> VEG -> splice into inputs_embeds -> decoder.
    # --veg-only stops after the VEG for vision bring-up without the decoder.
    #
    # This drives the VEG and the decoder by hand, which is useful for vision
    # bring-up (--veg-only, --ref, --dump). The integrated path that geniex-bench
    # and the CLI take is gemma4::makeVLMPipeline in gemma4_vlm.cpp, compiled
    # into geniex_vlm.
    add_executable(gemma4_vlm gemma4_vlm_example.cpp)
    target_link_libraries(gemma4_vlm PRIVATE geniex_vlm geniex-proc geniex-proc-vision)
    set_target_properties(gemma4_vlm PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    )
endif()
