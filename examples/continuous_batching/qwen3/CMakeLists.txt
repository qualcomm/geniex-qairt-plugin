# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

add_executable(qwen3_cb qwen3_cb_example.cpp)

# Include paths:
#   - examples/continuous_batching/common  → cb/*.h shared headers
#   - examples/continuous_batching/qwen3   → qwen3_cb.h (current dir)
# The model spec `qwen3_4b_instruct_2507::makeSpec` is reused from
# core/include/pipeline/llm_family.h, which ships with geniex_core.
target_include_directories(qwen3_cb PRIVATE
    ${CMAKE_SOURCE_DIR}/examples/continuous_batching/common
    ${CMAKE_SOURCE_DIR}/examples/continuous_batching/qwen3
)

target_link_libraries(qwen3_cb PRIVATE geniex_core geniex-proc)

set_target_properties(qwen3_cb PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)
