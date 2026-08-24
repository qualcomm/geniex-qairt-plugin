# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

add_executable(kv_layout_check kv_layout_check.cpp)
target_link_libraries(kv_layout_check PRIVATE geniex_core geniex-proc)
set_target_properties(kv_layout_check PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# Dumps a graph's own KV output bytes so the hardware tile layout can be
# recovered from ground truth (flat bundle vs its ENABLE_NATIVE_KV twin) instead
# of inferred. Uses the eaglet driver because that is what the natKV target
# export needs to initialize.
add_executable(kv_layout_probe kv_layout_probe.cpp)
target_link_libraries(kv_layout_probe PRIVATE geniex_core geniex-proc)
target_include_directories(kv_layout_probe PRIVATE ${CMAKE_SOURCE_DIR}/models/qwen3_eaglet)
set_target_properties(kv_layout_probe PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)
