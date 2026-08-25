# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

add_executable(qwen2_5_vl_7b
    qwen2_5_vl_7b_example.cpp
)
target_include_directories(qwen2_5_vl_7b PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)
target_link_libraries(qwen2_5_vl_7b PRIVATE geniex_vlm geniex-proc-vision)
set_target_properties(qwen2_5_vl_7b PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# VLMPipeline-based variant of the same model.
add_executable(qwen2_5_vl_7b_pipeline
    qwen2_5_vl_7b_pipeline_example.cpp
)
target_include_directories(qwen2_5_vl_7b_pipeline PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)
target_link_libraries(qwen2_5_vl_7b_pipeline PRIVATE geniex_vlm geniex-proc-vision)
set_target_properties(qwen2_5_vl_7b_pipeline PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)
