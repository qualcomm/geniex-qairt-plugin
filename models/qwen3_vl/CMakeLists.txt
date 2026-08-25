# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

add_executable(qwen3_vl_4b
    qwen3_vl_4b_example.cpp
)
target_include_directories(qwen3_vl_4b PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)
target_link_libraries(qwen3_vl_4b PRIVATE geniex_vlm geniex-proc-vision)
set_target_properties(qwen3_vl_4b PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# VLMPipeline-based variant of the same model.
add_executable(qwen3_vl_4b_pipeline
    qwen3_vl_4b_pipeline_example.cpp
)
target_include_directories(qwen3_vl_4b_pipeline PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)
target_link_libraries(qwen3_vl_4b_pipeline PRIVATE geniex_vlm geniex-proc-vision)
set_target_properties(qwen3_vl_4b_pipeline PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# Qwen3-VL-8B-Instruct, VLMPipeline-based.
add_executable(qwen3_vl_8b_pipeline
    qwen3_vl_8b_pipeline_example.cpp
)
target_include_directories(qwen3_vl_8b_pipeline PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)
target_link_libraries(qwen3_vl_8b_pipeline PRIVATE geniex_vlm geniex-proc-vision)
set_target_properties(qwen3_vl_8b_pipeline PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)
