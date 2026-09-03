# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

add_executable(kv_layout_check kv_layout_check.cpp)
target_link_libraries(kv_layout_check PRIVATE geniex_core geniex-proc)
set_target_properties(kv_layout_check PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)
