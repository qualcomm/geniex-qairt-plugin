# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

# EAGLE (eaglet) speculative-decoding runner for Qwen3-4B.
add_executable(eaglet_example eaglet_example.cpp)
target_link_libraries(eaglet_example PRIVATE geniex_core geniex-proc)
target_include_directories(eaglet_example PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
set_target_properties(eaglet_example PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)
