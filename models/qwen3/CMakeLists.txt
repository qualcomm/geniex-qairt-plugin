# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

foreach(EXAMPLE_NAME qwen3_1_7b qwen3_4b qwen3_4b_instruct_2507 qwen3_8b)
    add_executable(${EXAMPLE_NAME} ${EXAMPLE_NAME}_example.cpp)
    target_link_libraries(${EXAMPLE_NAME} PRIVATE geniex_core geniex-proc)
    set_target_properties(${EXAMPLE_NAME} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    )
endforeach()

# # Xtensor-based example needs xtensor_utils.cpp compiled as an additional source
# # (not part of geniex_core to avoid adding xtensor dependency to the core library).
# add_executable(qwen3_4b_xtensor
#     qwen3_4b_xtensor_example.cpp
#     ${CMAKE_SOURCE_DIR}/core/src/llm/xtensor_utils.cpp
# )
# target_link_libraries(qwen3_4b_xtensor PRIVATE geniex_core)
# set_target_properties(qwen3_4b_xtensor PROPERTIES
#     RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
# )
