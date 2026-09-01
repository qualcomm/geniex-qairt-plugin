# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

foreach(EXAMPLE_NAME phi4_mini_instruct)
    add_executable(${EXAMPLE_NAME} ${EXAMPLE_NAME}_example.cpp)
    target_link_libraries(${EXAMPLE_NAME} PRIVATE geniex_core geniex-proc)
    set_target_properties(${EXAMPLE_NAME} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    )
endforeach()
