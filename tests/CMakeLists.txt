# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

include(CTest)

# All tests run from the repo root so `./modelfiles/<name>/…` resolves.
set(_TEST_WD "${CMAKE_SOURCE_DIR}")

# ── llm_test ────────────────────────────────────────────────────────────────
add_executable(llm_test llm.cpp)
target_link_libraries(llm_test PRIVATE geniex_core geniex-proc)
set_target_properties(llm_test PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# Keep in sync with llm_model_registry() and modelFilesTable() in tests/llm.cpp.
set(_LLM_MODELS
    qwen3_4b
    qwen3_4b_instruct_2507
    # qwen2_5_7b_instruct
    # phi_3_5_mini_instruct
    # llama_v3_8b_instruct
    # llama_v3_elyza_jp_8b
    # llama_v3_taide_8b_chat
    # llama_v3_1_8b_instruct
    # llama_v3_1_sea_lion_3_5_8b_r
    # llama_v3_2_1b_instruct
    # llama_v3_2_3b_instruct
    # falcon_v3_7b_instruct
)

foreach(_model IN LISTS _LLM_MODELS)
    add_test(
        NAME              "llm_${_model}_generation"
        COMMAND           llm_test --model "${_model}"
        WORKING_DIRECTORY "${_TEST_WD}"
    )
    set_tests_properties("llm_${_model}_generation" PROPERTIES
        TIMEOUT 240
        LABELS  "llm;generation"
    )
endforeach()

# ── vlm_test ────────────────────────────────────────────────────────────────
if(GENIEX_BUILD_VLM)
    # qwen2_5_vl.cpp defines geniex::qwen2_5_vl_7b::makeModel. It is not in
    # geniex_vlm (lives with the examples), so we compile it into vlm_test.
    add_executable(vlm_test
        vlm.cpp
        ${CMAKE_SOURCE_DIR}/models/qwen2_5_vl/qwen2_5_vl.cpp
    )
    target_link_libraries(vlm_test PRIVATE geniex_vlm geniex-proc-vision)
    target_include_directories(vlm_test PRIVATE
        ${CMAKE_SOURCE_DIR}/models/qwen2_5_vl
    )
    set_target_properties(vlm_test PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    )

    add_test(
        NAME              "vlm_qwen2_5_vl_7b_generation"
        COMMAND           vlm_test --model qwen2_5_vl_7b
        WORKING_DIRECTORY "${_TEST_WD}"
    )
    set_tests_properties("vlm_qwen2_5_vl_7b_generation" PROPERTIES
        TIMEOUT 360  # VLM prefill is heavier
        LABELS  "vlm;generation"
    )
endif()
