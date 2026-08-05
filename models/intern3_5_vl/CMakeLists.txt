# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

# Intern3.5-VL-2B, VLMPipeline-based.
add_executable(intern3_5_vl_2b_pipeline
    intern3_5_vl_2b_pipeline_example.cpp
)
target_include_directories(intern3_5_vl_2b_pipeline PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)
target_link_libraries(intern3_5_vl_2b_pipeline PRIVATE geniex_vlm geniex-proc-vision)
set_target_properties(intern3_5_vl_2b_pipeline PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)
