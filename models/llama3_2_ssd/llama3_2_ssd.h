// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "llm/llm_types.h"
#include "ssd_model.h"
#include "ssd_types.h"
#include "llm/input_provider.h"
#include "pipeline/llm_pipeline.h"
#include "pipeline/chat_template.h"

namespace geniex {
namespace llama3_2_3b_ssd {

static constexpr size_t kHeadDim   = 128;
static constexpr float  kRopeTheta = 500000.0f;

// Returns the architecture spec for Llama-3.2-3B-Instruct-SSD (3 shards, CL 4096).
//
// This is the Self-Speculative Decoding variant. It has TWO AR graph variants:
//   AR-128: prompt prefill (same as non-SSD)
//   AR-32:  SSD tree verification (30 tokens: 10 draft + 20 forecast, padded to 32)
//
// Shard layout (per the Genie export):
//   shard 0 : embedding only   – input_ids → embeddings        (no KV cache)
//   shard 1 : layers  0 – 13   – embeddings → hidden_state     (KV layers 0–13)
//   shard 2 : layers 14 – 27   – hidden_state → logits         (KV layers 14–27)
//
// Tensor names use underscores (QNN runtime replaces ONNX slashes).
// Graph names use prompt_ prefix for both AR variants.
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

        .seq_len_prefill = 128,   // AR-128 for prompt prefill
        .seq_len_decode  = 32,    // AR-32 for SSD tree verification

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

inline SSDConfig makeSSDConfig(const std::string& forecast_prefix_path) {
    return SSDConfig{
        .branches = {3, 2},
        .forecast_prefix = 16,
        .forecast_prefix_path = forecast_prefix_path,
        .rope_theta = kRopeTheta,
    };
}

inline SSDModel makeModel(const std::string& forecast_prefix_path) {
    SSDModel m(makeSpec(), makeSSDConfig(forecast_prefix_path));
    m.addInputProvider(std::make_unique<TokenIdInputProvider>("input_ids", 128001));
    m.addInputProvider(std::make_unique<RoPEInputProvider>(kHeadDim, kRopeTheta));
    return m;
}

inline ChatTemplateFunc chatTemplate = llama3ChatTemplate;

inline std::optional<LLMPipeline> makePipeline(const QnnRuntimeConfig& runtime_cfg,
                                               const ModelConfig& model_cfg) {
    LLMPipeline pipe;
    if (!pipe.create(chatTemplate,
                     makeModel(model_cfg.forecast_prefix_path.value_or("")),
                     runtime_cfg, model_cfg))
        return std::nullopt;
    return pipe;
}

} // namespace llama3_2_3b_ssd
} // namespace geniex
