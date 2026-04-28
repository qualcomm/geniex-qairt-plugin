// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <functional>
#include <optional>
#include <string>
#include <unordered_map>

#include "pipeline/llm_pipeline.h"
#include "types.h"

// Per-model makePipeline factories.
#include "llama3_2/llama3_2.h"
#include "phi3_5/phi3_5.h"
#include "qwen2_5/qwen2_5.h"
#include "qwen3/qwen3.h"

namespace geniex {

struct LlmModelEntry {
    std::function<std::optional<LLMPipeline>(const QnnRuntimeConfig&, const ModelConfig&)> make_pipeline;
};

inline const std::unordered_map<std::string, LlmModelEntry>& llm_model_registry() {
    static const std::unordered_map<std::string, LlmModelEntry> registry = {
        // model names here should match QAIHub model IDs
        {"qwen3_4b",                  {qwen3_4b::makePipeline}},
        {"qwen3_4b_instruct_2507",    {qwen3_4b_instruct_2507::makePipeline}},
        {"qwen3_8b",                  {qwen3_8b::makePipeline}},
        {"phi_3_5_mini_instruct",     {phi3_5::makePipeline}},
        {"llama_v3_2_1b_instruct",    {llama3_2_1b::makePipeline}},
        {"llama_v3_2_3b_instruct",    {llama3_2_3b::makePipeline}},
    };
    return registry;
}

}  // namespace geniex
