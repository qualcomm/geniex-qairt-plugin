# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
add_executable(phonon_tts phonon_example.cpp)
target_link_libraries(phonon_tts PRIVATE geniex_tts)
set_target_properties(phonon_tts PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
