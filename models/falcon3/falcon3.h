// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "llm/llm_types.h"
#include "llm/llm_model.h"
#include "llm/input_provider.h"
#include "pipeline/chat_template.h"
#include "pipeline/llm_pipeline.h"

namespace geniex {

// Falcon3 instruct format: <|system|>\n...\n<|user|>\n...\n<|assistant|>\n
inline std::string falcon3ChatTemplate(const std::string& user_message,
                                       const std::string& system_prompt,
                                       bool /*enable_thinking*/) {
    std::string result;
    std::string sys = system_prompt.empty()
                          ? "You are a helpful AI assistant"
                          : system_prompt;
    result += "<|system|>\n" + sys + "\n";
    result += "<|user|>\n" + user_message + "\n<|assistant|>\n";
    return result;
}

namespace falcon3_7b {

static constexpr size_t kHeadDim   = 256;
static constexpr float  kRopeTheta = 1000042.0f;

// Returns the architecture spec for Falcon3-7B-Instruct (5 shards, CL 4096).
//
// Shard layout (per the Genie export):
//   shard 0 : embedding only   – input_ids → embeddings                (no KV cache)
//   shard 1 : layers  0 –  7   – embeddings → hidden_state             (KV layers 0–7)
//   shard 2 : layers  8 – 15   – hidden_state → hidden_state           (KV layers 8–15)
//   shard 3 : layers 16 – 23   – hidden_state → hidden_state           (KV layers 16–23)
//   shard 4 : layers 24 – 27   – hidden_state → logits                 (KV layers 24–27)
//
// Tensor names use underscores (QNN runtime replaces ONNX slashes).
// Graph names use prompt_/token_ prefix.
inline LLMSpec makeSpec() {
    return LLMSpec{
        .shards = {
            {"input_ids",
             "_model_model_embed_tokens_Gather_output_0"},
            {"_model_model_embed_tokens_Gather_output_0",
             "_model_model_layers_7_Add_1_output_0"},
            {"_model_model_layers_7_Add_1_output_0",
             "_model_model_layers_15_Add_1_output_0"},
            {"_model_model_layers_15_Add_1_output_0",
             "_model_model_layers_23_Add_1_output_0"},
            {"_model_model_layers_23_Add_1_output_0",
             "logits"},
        },
        .state_blocks = {
            makeKVOnlyStateBlock({std::nullopt, LayerRange{0, 7}, LayerRange{8, 15}, LayerRange{16, 23}, LayerRange{24, 27}}),
        },

        .seq_len_prefill = 128,
        .seq_len_decode  = 1,

        .hidden_size   = 3072,
        .num_heads     = 12,
        .num_kv_heads  = 4,
        .head_dim      = kHeadDim,
        .vocab_size    = 131072,

        .context_lengths = {4096},

        .graph_name_pattern = "{phase}_ar{ar}_cl{cl}_{shard}_of_{total}",

        .eos_token_ids = {11},
    };
}

inline LLMModel makeModel() {
    LLMModel m(makeSpec());
    m.addInputProvider(std::make_unique<TokenIdInputProvider>("input_ids", 11));
    m.addInputProvider(std::make_unique<RoPEInputProvider>(kHeadDim, kRopeTheta));
    return m;
}

inline ChatTemplateFunc chatTemplate = falcon3ChatTemplate;

inline std::optional<LLMPipeline> makePipeline(const QnnRuntimeConfig& runtime_cfg,
                                               const ModelConfig& model_cfg) {
    LLMPipeline pipe;
    if (!pipe.create(chatTemplate, makeModel(), runtime_cfg, model_cfg))
        return std::nullopt;
    return pipe;
}

} // namespace falcon3_7b
} // namespace geniex
