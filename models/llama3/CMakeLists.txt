# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

# All Llama 3 / 3.1 / 3.2 variants share one runtime (models/llama3/llama3.h);
# per-variant differences come from each bundle's config.json + metadata.json.
foreach(EXAMPLE_NAME
        llama_v3_8b_instruct
        llama_v3_elyza_jp_8b
        llama_v3_taide_8b_chat
        llama3_1_8b
        sea_lion_8b
        llama3_2_1b
        llama3_2_3b)
    add_executable(${EXAMPLE_NAME} ${EXAMPLE_NAME}_example.cpp)
    target_link_libraries(${EXAMPLE_NAME} PRIVATE geniex_core geniex-proc)
    set_target_properties(${EXAMPLE_NAME} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    )
endforeach()
