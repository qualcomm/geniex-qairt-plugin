// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "llm/llm_types.h"
#include "llm/llm_model.h"
#include "llm/input_provider.h"
#include "pipeline/chat_template.h"
#include "pipeline/llm_pipeline.h"

namespace geniex {

// Llama3 header-id format: <|start_header_id|>role<|end_header_id|>...<|eot_id|>
// Used by Llama 3, Llama 3.1, and Llama 3.2 model families.
inline std::string llama3ChatTemplate(const std::string& user_message,
                                      const std::string& system_prompt,
                                      bool /*enable_thinking*/) {
    std::string out = "<|begin_of_text|>";
    if (!system_prompt.empty()) {
        out += "<|start_header_id|>system<|end_header_id|>\n"
             + system_prompt + "<|eot_id|>";
    }
    out += "<|start_header_id|>user<|end_header_id|>\n";
    return out + user_message + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n";
}

namespace llama_v3_8b_instruct {

static constexpr size_t  kHeadDim    = 128;
static constexpr float   kRopeTheta  = 500000.0f;

// Returns the architecture spec for Llama 3 8B Instruct (5 shards, CL 4096).
//
// Shard layout:
//   shard 0 : embedding only   – input_ids → embeddings                     (no KV cache)
//   shard 1 : layers  0 –  8   – embeddings → hidden                       (KV layers 0–8)
//   shard 2 : layers  9 – 17   – hidden → hidden                           (KV layers 9–17)
//   shard 3 : layers 18 – 26   – hidden → hidden                           (KV layers 18–26)
//   shard 4 : layers 27 – 31   – hidden → logits                           (KV layers 27–31)
inline LLMSpec makeSpec() {
    return LLMSpec{
        .shards = {
            {"input_ids",
             "_model_model_embed_tokens_Gather_output_0"},
            {"_model_model_embed_tokens_Gather_output_0",
             "_model_model_layers_8_Add_1_output_0"},
            {"_model_model_layers_8_Add_1_output_0",
             "_model_model_layers_17_Add_1_output_0"},
            {"_model_model_layers_17_Add_1_output_0",
             "_model_model_layers_26_Add_1_output_0"},
            {"_model_model_layers_26_Add_1_output_0",
             "logits"},
        },
        .state_blocks = {
            makeKVOnlyStateBlock({std::nullopt, LayerRange{0, 8}, LayerRange{9, 17}, LayerRange{18, 26}, LayerRange{27, 31}}),
        },

        .seq_len_prefill = 128,
        .seq_len_decode  = 1,

        .hidden_size   = 4096,
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

} // namespace llama_v3_8b_instruct

namespace llama_v3_elyza_jp_8b {

static constexpr size_t  kHeadDim    = 128;
static constexpr float   kRopeTheta  = 500000.0f;

// Returns the architecture spec for Llama 3 Elyza JP 8B (5 shards, CL 4096).
// Same architecture as llama_v3_8b_instruct.
//
// Shard layout:
//   shard 0 : embedding only   – input_ids → embeddings                     (no KV cache)
//   shard 1 : layers  0 –  8   – embeddings → hidden                       (KV layers 0–8)
//   shard 2 : layers  9 – 17   – hidden → hidden                           (KV layers 9–17)
//   shard 3 : layers 18 – 26   – hidden → hidden                           (KV layers 18–26)
//   shard 4 : layers 27 – 31   – hidden → logits                           (KV layers 27–31)
inline LLMSpec makeSpec() {
    return LLMSpec{
        .shards = {
            {"input_ids",
             "_model_model_embed_tokens_Gather_output_0"},
            {"_model_model_embed_tokens_Gather_output_0",
             "_model_model_layers_8_Add_1_output_0"},
            {"_model_model_layers_8_Add_1_output_0",
             "_model_model_layers_17_Add_1_output_0"},
            {"_model_model_layers_17_Add_1_output_0",
             "_model_model_layers_26_Add_1_output_0"},
            {"_model_model_layers_26_Add_1_output_0",
             "logits"},
        },
        .state_blocks = {
            makeKVOnlyStateBlock({std::nullopt, LayerRange{0, 8}, LayerRange{9, 17}, LayerRange{18, 26}, LayerRange{27, 31}}),
        },

        .seq_len_prefill = 128,
        .seq_len_decode  = 1,

        .hidden_size   = 4096,
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

} // namespace llama_v3_elyza_jp_8b

namespace llama_v3_taide_8b_chat {

static constexpr size_t  kHeadDim    = 128;
static constexpr float   kRopeTheta  = 500000.0f;

// Returns the architecture spec for Llama V3 TAIDE 8B Chat (5 shards, CL 4096).
// Same architecture as llama_v3_8b_instruct.
//
// Shard layout:
//   shard 0 : embedding only   – input_ids → embeddings                     (no KV cache)
//   shard 1 : layers  0 –  8   – embeddings → hidden                       (KV layers 0–8)
//   shard 2 : layers  9 – 17   – hidden → hidden                           (KV layers 9–17)
//   shard 3 : layers 18 – 26   – hidden → hidden                           (KV layers 18–26)
//   shard 4 : layers 27 – 31   – hidden → logits                           (KV layers 27–31)
inline LLMSpec makeSpec() {
    return LLMSpec{
        .shards = {
            {"input_ids",
             "_model_model_embed_tokens_Gather_output_0"},
            {"_model_model_embed_tokens_Gather_output_0",
             "_model_model_layers_8_Add_1_output_0"},
            {"_model_model_layers_8_Add_1_output_0",
             "_model_model_layers_17_Add_1_output_0"},
            {"_model_model_layers_17_Add_1_output_0",
             "_model_model_layers_26_Add_1_output_0"},
            {"_model_model_layers_26_Add_1_output_0",
             "logits"},
        },
        .state_blocks = {
            makeKVOnlyStateBlock({std::nullopt, LayerRange{0, 8}, LayerRange{9, 17}, LayerRange{18, 26}, LayerRange{27, 31}}),
        },

        .seq_len_prefill = 128,
        .seq_len_decode  = 1,

        .hidden_size   = 4096,
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

} // namespace llama_v3_taide_8b_chat
} // namespace geniex
