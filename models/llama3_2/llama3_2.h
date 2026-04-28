// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "llm/llm_types.h"
#include "llm/llm_model.h"
#include "llm/input_provider.h"
#include "llama3/llama3.h"

namespace geniex {
namespace llama3_2_1b {

static constexpr size_t kHeadDim   = 64;
static constexpr float  kRopeTheta = 500000.0f;

// Returns the architecture spec for Llama-3.2-1B-Instruct (3 shards, CL 4096).
//
// Shard layout (per the Genie export):
//   shard 0 : embedding only   – input_ids → embeddings        (no KV cache)
//   shard 1 : layers  0 –  7   – embeddings → hidden_state     (KV layers 0–7)
//   shard 2 : layers  8 – 15   – hidden_state → logits         (KV layers 8–15)
inline LLMSpec makeSpec() {
    return LLMSpec{
        .shards = {
            {"input_ids",
             "_model_model_embed_tokens_Gather_output_0"},
            {"_model_model_embed_tokens_Gather_output_0",
             "_model_model_layers_7_Add_1_output_0"},
            {"_model_model_layers_7_Add_1_output_0",
             "logits"},
        },
        .state_blocks = {
            makeKVOnlyStateBlock({std::nullopt, LayerRange{0, 7}, LayerRange{8, 15}}),
        },

        .seq_len_prefill = 128,
        .seq_len_decode  = 1,

        .hidden_size   = 2048,
        .num_heads     = 32,
        .num_kv_heads  = 8,
        .head_dim      = kHeadDim,
        .vocab_size    = 128256,

        .context_lengths = {4096},

        .graph_name_pattern = "{phase}_ar{ar}_cl{cl}_{shard}_of_{total}",

        .eos_token_ids = {128001, 128008, 128009},
    };
}

inline LLMModel makeModel() {
    LLMModel m(makeSpec());
    m.addInputProvider(std::make_unique<TokenIdInputProvider>("input_ids", 128001));
    m.addInputProvider(std::make_unique<RoPEInputProvider>(kHeadDim, kRopeTheta));
    return m;
}

inline ChatTemplateFunc chatTemplate = llama3ChatTemplate;

inline std::optional<LLMPipeline> makePipeline(const QnnRuntimeConfig& runtime_cfg,
                                               const ModelConfig& model_cfg) {
    LLMPipeline pipe;
    if (!pipe.create(chatTemplate, makeModel(), runtime_cfg, model_cfg))
        return std::nullopt;
    return pipe;
}

} // namespace llama3_2_1b

namespace llama3_2_3b {

static constexpr size_t kHeadDim   = 128;
static constexpr float  kRopeTheta = 500000.0f;

// Returns the architecture spec for Llama-3.2-3B-Instruct (3 shards, CL 4096).
//
// Shard layout (per the exported CB):
//   shard 0 : embedding only   – input_ids → embeddings        (no KV cache)
//   shard 1 : layers  0 – 13   – embeddings → hidden_state     (KV layers 0–13)
//   shard 2 : layers 14 – 27   – hidden_state → logits         (KV layers 14–27)
//
// Tensor names use underscores (QNN runtime replaces ONNX slashes).
// Graph names use prompt_/token_ prefix.
inline LLMSpec makeSpec() {
    return LLMSpec{
        .shards = {
            {"input_ids",
             "_model_model_embed_tokens_Gather_output_0"},
            {"_model_model_embed_tokens_Gather_output_0",
             "_model_model_layers_13_Add_1_output_0"},
            {"_model_model_layers_13_Add_1_output_0",
             "logits"},
        },
        .state_blocks = {
            makeKVOnlyStateBlock({std::nullopt, LayerRange{0, 13}, LayerRange{14, 27}}),
        },

        .seq_len_prefill = 128,
        .seq_len_decode  = 1,

        .hidden_size   = 3072,
        .num_heads     = 24,
        .num_kv_heads  = 8,
        .head_dim      = kHeadDim,
        .vocab_size    = 128256,

        .context_lengths = {4096},

        .graph_name_pattern = "{phase}_ar{ar}_cl{cl}_{shard}_of_{total}",

        .eos_token_ids = {128001, 128008, 128009},
    };
}

inline LLMModel makeModel() {
    LLMModel m(makeSpec());
    m.addInputProvider(std::make_unique<TokenIdInputProvider>("input_ids", 128001));
    m.addInputProvider(std::make_unique<RoPEInputProvider>(kHeadDim, kRopeTheta));
    return m;
}

inline ChatTemplateFunc chatTemplate = llama3ChatTemplate;

inline std::optional<LLMPipeline> makePipeline(const QnnRuntimeConfig& runtime_cfg,
                                               const ModelConfig& model_cfg) {
    LLMPipeline pipe;
    if (!pipe.create(chatTemplate, makeModel(), runtime_cfg, model_cfg))
        return std::nullopt;
    return pipe;
}

} // namespace llama3_2_3b
} // namespace geniex
