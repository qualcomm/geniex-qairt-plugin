# Copyright (c) 2026 Nexa AI. SPDX-License-Identifier: BSD-3-Clause

add_executable(gemma4_e2b gemma4_e2b_example.cpp)
target_link_libraries(gemma4_e2b PRIVATE geniex_core geniex-proc)
set_target_properties(gemma4_e2b PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# AR=1 decode driver: prefill(ar128)-once + decode(ar1)-loop over a 2-graph
# weight-shared context binary.
add_executable(gemma4_e2b_decode gemma4_e2b_decode_example.cpp)
target_link_libraries(gemma4_e2b_decode PRIVATE geniex_core geniex-proc)
set_target_properties(gemma4_e2b_decode PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)
