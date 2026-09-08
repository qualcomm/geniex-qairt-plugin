# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

# Aggregator for everything under examples/. Add new example directories
# here so the root CMakeLists stays a single line per category.

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/auto_llm)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/continuous_batching)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/kv_layout_check)
