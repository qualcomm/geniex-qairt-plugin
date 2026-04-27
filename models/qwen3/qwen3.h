#pragma once

#include "llm/llm_types.h"
#include "llm/llm_model.h"
#include "llm/input_provider.h"
#include "pipeline/chat_template.h"
#include "pipeline/llm_pipeline.h"

namespace geniex {
namespace qwen3_4b_xtensor {

static constexpr size_t  kHeadDim    = 128;
static constexpr float   kRopeTheta  = 1000000.0f;

// Returns the architecture spec for Qwen3-4B weight-sharing export (2 shards, CL 4096).
inline LLMSpec makeSpec() {
    return LLMSpec{
        .shards = {
            {"input_embeds",         "_Add_88_Add_output_0"},
            {"_Add_88_Add_output_0", "logits"},
        },
        .state_blocks = {
            makeKVOnlyStateBlock({LayerRange{0, 17}, LayerRange{18, 35}}),
        },

        .seq_len_prefill = 128,
        .seq_len_decode  = 1,

        .hidden_size   = 2560,
        .num_heads     = 32,
        .num_kv_heads  = 8,
        .head_dim      = kHeadDim,
        .vocab_size    = 151936,

        .context_lengths = {4096},

        .eos_token_ids = {151645},
    };
}

inline LLMModel makeModel() {
    LLMModel m(makeSpec());
    m.addInputProvider(std::make_unique<EmbeddingInputProvider>());
    m.addInputProvider(std::make_unique<RoPEInputProvider>(kHeadDim, kRopeTheta));
    return m;
}

inline ChatTemplateFunc chatTemplate = chatMLTemplate;

inline std::optional<LLMPipeline> makePipeline(const QnnRuntimeConfig& runtime_cfg,
                                               const ModelConfig& model_cfg) {
    LLMPipeline pipe;
    if (!pipe.create(chatTemplate, makeModel(), runtime_cfg, model_cfg))
        return std::nullopt;
    return pipe;
}

} // namespace qwen3_4b_xtensor

namespace qwen3_4b {

static constexpr size_t  kHeadDim    = 128;
static constexpr float   kRopeTheta  = 1000000.0f;

// Returns the architecture spec for Qwen3-4B (4 shards, 5 CL variants, on-device embedding).
//
// Shard layout:
//   shard 0 : embedding only   – input_ids → embeddings  (no KV cache)
//   shard 1 : layers  0 – 11   – embeddings → hidden     (KV layers 0–11)
//   shard 2 : layers 12 – 23   – hidden → hidden         (KV layers 12–23)
//   shard 3 : layers 24 – 35   – hidden → logits         (KV layers 24–35)
inline LLMSpec makeSpec() {
    return LLMSpec{
        .shards = {
            {"input_ids",
             "_model_model_embed_tokens_Gather_output_0"},
            {"_model_model_embed_tokens_Gather_output_0",
             "_model_model_layers_11_Add_1_output_0"},
            {"_model_model_layers_11_Add_1_output_0",
             "_model_model_layers_23_Add_1_output_0"},
            {"_model_model_layers_23_Add_1_output_0",
             "logits"},
        },
        .state_blocks = {
            makeKVOnlyStateBlock({std::nullopt, LayerRange{0, 11}, LayerRange{12, 23}, LayerRange{24, 35}}),
        },

        .seq_len_prefill = 128,
        .seq_len_decode  = 1,

        .hidden_size   = 2560,
        .num_heads     = 32,
        .num_kv_heads  = 8,
        .head_dim      = kHeadDim,
        .vocab_size    = 151936,

        .context_lengths = {512, 1024, 2048, 3072, 4096},

        .graph_name_pattern = "{phase}_ar{ar}_cl{cl}_{shard}_of_{total}",

        .eos_token_ids = {151645},
    };
}

inline LLMModel makeModel() {
    LLMModel m(makeSpec());
    m.addInputProvider(std::make_unique<TokenIdInputProvider>("input_ids", 151645));
    m.addInputProvider(std::make_unique<RoPEInputProvider>(kHeadDim, kRopeTheta));
    return m;
}

inline ChatTemplateFunc chatTemplate = chatMLTemplate;

inline std::optional<LLMPipeline> makePipeline(const QnnRuntimeConfig& runtime_cfg,
                                               const ModelConfig& model_cfg) {
    LLMPipeline pipe;
    if (!pipe.create(chatTemplate, makeModel(), runtime_cfg, model_cfg))
        return std::nullopt;
    return pipe;
}

} // namespace qwen3_4b

namespace qwen3_4b_instruct_2507 {

static constexpr size_t  kHeadDim    = 128;
static constexpr float   kRopeTheta  = 1000000.0f;

// Returns the architecture spec for Qwen3-4B Instruct 2507 (4 shards, 5 CL variants).
//
// Shard layout (same as qwen3_4b):
//   shard 0 : embedding only   – input_ids → embeddings  (no KV cache)
//   shard 1 : layers  0 – 11   – embeddings → hidden     (KV layers 0–11)
//   shard 2 : layers 12 – 23   – hidden → hidden         (KV layers 12–23)
//   shard 3 : layers 24 – 35   – hidden → logits         (KV layers 24–35)
inline LLMSpec makeSpec() {
    return LLMSpec{
        .shards = {
            {"input_ids",
             "_model_model_embed_tokens_Gather_output_0"},
            {"_model_model_embed_tokens_Gather_output_0",
             "_model_model_layers_11_Add_1_output_0"},
            {"_model_model_layers_11_Add_1_output_0",
             "_model_model_layers_23_Add_1_output_0"},
            {"_model_model_layers_23_Add_1_output_0",
             "logits"},
        },
        .state_blocks = {
            makeKVOnlyStateBlock({std::nullopt, LayerRange{0, 11}, LayerRange{12, 23}, LayerRange{24, 35}}),
        },

        .seq_len_prefill = 128,
        .seq_len_decode  = 1,

        .hidden_size   = 2560,
        .num_heads     = 32,
        .num_kv_heads  = 8,
        .head_dim      = kHeadDim,
        .vocab_size    = 151936,

        .context_lengths = {512, 1024, 2048, 3072, 4096},

        .graph_name_pattern = "{phase}_ar{ar}_cl{cl}_{shard}_of_{total}",

        .eos_token_ids = {151645},
    };
}

inline LLMModel makeModel() {
    LLMModel m(makeSpec());
    m.addInputProvider(std::make_unique<TokenIdInputProvider>("input_ids", 151645));
    m.addInputProvider(std::make_unique<RoPEInputProvider>(kHeadDim, kRopeTheta));
    return m;
}

inline ChatTemplateFunc chatTemplate = chatMLTemplate;

inline std::optional<LLMPipeline> makePipeline(const QnnRuntimeConfig& runtime_cfg,
                                               const ModelConfig& model_cfg) {
    LLMPipeline pipe;
    if (!pipe.create(chatTemplate, makeModel(), runtime_cfg, model_cfg))
        return std::nullopt;
    return pipe;
}

} // namespace qwen3_4b_instruct_2507

namespace qwen3_4b_instruct_2507_aihub {

static constexpr size_t  kHeadDim    = 128;
static constexpr float   kRopeTheta  = 1000000.0f;

// Returns the architecture spec for Qwen3-4B Instruct 2507 AI Hub export (4 shards, 5 CL variants).
//
// Shard layout (same as qwen3_4b_instruct_2507):
//   shard 0 : embedding only   – input_ids → embeddings  (no KV cache)
//   shard 1 : layers  0 – 11   – embeddings → hidden     (KV layers 0–11)
//   shard 2 : layers 12 – 23   – hidden → hidden         (KV layers 12–23)
//   shard 3 : layers 24 – 35   – hidden → logits         (KV layers 24–35)
inline LLMSpec makeSpec() {
    return LLMSpec{
        .shards = {
            {"input_ids",
             "_model_model_embed_tokens_Gather_output_0"},
            {"_model_model_embed_tokens_Gather_output_0",
             "_model_model_layers_11_Add_1_output_0"},
            {"_model_model_layers_11_Add_1_output_0",
             "_model_model_layers_23_Add_1_output_0"},
            {"_model_model_layers_23_Add_1_output_0",
             "logits"},
        },
        .state_blocks = {
            makeKVOnlyStateBlock({std::nullopt, LayerRange{0, 11}, LayerRange{12, 23}, LayerRange{24, 35}}),
        },

        .seq_len_prefill = 128,
        .seq_len_decode  = 1,

        .hidden_size   = 2560,
        .num_heads     = 32,
        .num_kv_heads  = 8,
        .head_dim      = kHeadDim,
        .vocab_size    = 151936,

        .context_lengths = {512, 1024, 2048, 3072, 4096},

        .graph_name_pattern = "{phase}_ar{ar}_cl{cl}_{shard}_of_{total}",

        .eos_token_ids = {151645},
    };
}

inline LLMModel makeModel() {
    LLMModel m(makeSpec());
    m.addInputProvider(std::make_unique<TokenIdInputProvider>("input_ids", 151645));
    m.addInputProvider(std::make_unique<RoPEInputProvider>(kHeadDim, kRopeTheta));
    return m;
}

inline ChatTemplateFunc chatTemplate = chatMLTemplate;

inline std::optional<LLMPipeline> makePipeline(const QnnRuntimeConfig& runtime_cfg,
                                               const ModelConfig& model_cfg) {
    LLMPipeline pipe;
    if (!pipe.create(chatTemplate, makeModel(), runtime_cfg, model_cfg))
        return std::nullopt;
    return pipe;
}

} // namespace qwen3_4b_instruct_2507_aihub

namespace qwen3_8b {

static constexpr size_t  kHeadDim    = 128;
static constexpr float   kRopeTheta  = 1000000.0f;

// Returns the architecture spec for Qwen3-8B (4 shards, CL 4096).
//
// Shard layout (9 layers each):
//   shard 0 : layers  0 –  8   – input_embeds → _Add_43_Add_output_0
//   shard 1 : layers  9 – 17   – _Add_43_Add_output_0 → _Add_88_Add_output_0
//   shard 2 : layers 18 – 26   – _Add_88_Add_output_0 → _Add_133_Add_output_0
//   shard 3 : layers 27 – 35   – _Add_133_Add_output_0 → logits
inline LLMSpec makeSpec() {
    return LLMSpec{
        .shards = {
            {"input_embeds",          "_Add_43_Add_output_0"},
            {"_Add_43_Add_output_0",  "_Add_88_Add_output_0"},
            {"_Add_88_Add_output_0",  "_Add_133_Add_output_0"},
            {"_Add_133_Add_output_0", "logits"},
        },
        .state_blocks = {
            makeKVOnlyStateBlock({LayerRange{0, 8}, LayerRange{9, 17}, LayerRange{18, 26}, LayerRange{27, 35}}),
        },

        .seq_len_prefill = 128,
        .seq_len_decode  = 1,

        .hidden_size   = 4096,
        .num_heads     = 32,
        .num_kv_heads  = 8,
        .head_dim      = kHeadDim,
        .vocab_size    = 151936,

        .context_lengths = {4096},

        .eos_token_ids = {151645},
    };
}

inline LLMModel makeModel() {
    LLMModel m(makeSpec());
    m.addInputProvider(std::make_unique<EmbeddingInputProvider>());
    m.addInputProvider(std::make_unique<RoPEInputProvider>(kHeadDim, kRopeTheta));
    return m;
}

inline ChatTemplateFunc chatTemplate = chatMLTemplate;

inline std::optional<LLMPipeline> makePipeline(const QnnRuntimeConfig& runtime_cfg,
                                               const ModelConfig& model_cfg) {
    LLMPipeline pipe;
    if (!pipe.create(chatTemplate, makeModel(), runtime_cfg, model_cfg))
        return std::nullopt;
    return pipe;
}

} // namespace qwen3_8b
} // namespace geniex
