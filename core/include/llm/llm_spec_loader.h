// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <variant>
#include <vector>

#include "geniex_export.h"
#include "llm/input_provider.h"
#include "llm/llm_types.h"
#include "types.h"

// Reads a QAIRT distributed model bundle and produces an LLMSpec + matching
// CPU-side InputProviders. Every numerical hyperparameter is inferred from
// the compiled-graph tensor shapes recorded in metadata.json. Anything the
// tensors can't carry (RoPE base/scaling, EOS/BOS, dialog type) is read
// from genie_config.json. We do NOT consult HuggingFace config.json.
//
// Bundle layout we depend on:
//   metadata.json     — QAIRT export metadata: model_id, graph names,
//                       per-graph I/O tensor specs (dtype, shape, quant params).
//   genie_config.json — runtime config: dialog.type, context tokens, RoPE
//                       parameters, embedding LUT spec.
//   tokenizer.json    — sentencepiece/BPE tokenizer (read by LLMPipeline).
//   *.bin             — compiled context-binary shards.

namespace geniex {

// ── RoPE scaling variants ──────
struct StandardRope {};

struct Llama3RopeScaling {
    float  factor;
    float  low_freq_factor;
    float  high_freq_factor;
    size_t original_max_position_embeddings;
};

struct LongRopeScaling {
    std::vector<float> long_factor;
    std::vector<float> short_factor;
    size_t             original_max_position_embeddings;
};

struct PartialRopeScaling {
    float rope_fraction;
    float scale;
};

struct MRopeScaling {
    std::vector<int> mrope_section;  // sums to head_dim/2
    int              spatial_merge_size = 2;
    int              time_step          = 50;
};

using RopeScaling = std::variant<StandardRope, Llama3RopeScaling, LongRopeScaling, PartialRopeScaling, MRopeScaling>;

// ── Vision-preprocessing block (VLM only) ────────────────────────────────────
struct ParsedVisionPreprocessing {
    int                image_width         = 0;
    int                image_height        = 0;
    int                patch_size          = 0;
    int                temporal_patch_size = 0;
    int                spatial_merge_size  = 0;
    std::vector<float> normalize_mean;
    std::vector<float> normalize_std;
};

// ── Parsed metadata.json ─────────────────────────────────────────────────────
// Carries everything inferable from the compiled graphs' tensor shapes:
// shard wiring, per-shard KV ranges, AR/CL set, hidden_size, num_kv_heads,
// head_dim, vocab_size, num_hidden_layers.
struct ParsedQAIRTMetadata {
    // Top-level metadata.json fields.
    std::string model_id;  // e.g. "qwen3_4b", "llama_v3_2_3b_instruct_ssd"

    // Shard wiring.
    std::vector<ShardSpec>                 shards;
    std::vector<std::optional<LayerRange>> shard_layer_ranges;

    // Inferred from each graph's `attention_mask` last dim (or, for VLM, the
    // top-level `genie.context_lengths` array). Sorted ascending.
    std::vector<size_t> context_lengths;

    // Inferred from `attention_mask.shape[2]` (or AR-suffix in graph name).
    size_t seq_len_prefill = 0;
    size_t seq_len_decode  = 0;

    // Empty for VLM bundles (bare `partN_of_M.bin` keys).
    std::string graph_name_pattern;

    // Tensor-shape-inferred LLM hyperparameters.
    size_t hidden_size       = 0;  // inputs_embeds.shape[2] / hidden-state.shape[2]
    size_t num_kv_heads      = 0;  // past_key_*.shape[0]
    size_t head_dim          = 0;  // past_key_*.shape[2]
    size_t vocab_size        = 0;  // logits last dim
    size_t num_hidden_layers = 0;  // max past_key_<N>_in across all shards + 1

    // Optional VLM-only fields.
    std::optional<ParsedVisionPreprocessing> vision_preprocessing;
    std::string                              vision_encoder_graph;  // empty if absent

    // First non-special input tensor name on shard 0, as recorded in
    // metadata.json (typically "input_ids" or "inputs_embeds"). Used only by
    // the embedding-provider factory to decide between TokenIdInputProvider
    // and EmbeddingInputProvider. Hidden-state wiring tensor names are not
    // populated by the loader — see LLMModel::discoverShardTensorNames.
    std::string first_shard_input_hint;
};

// ── Parsed genie_config.json ─────────────────────────────────────────────────
// Subset of the runtime config the loader needs after metadata-driven
// inference covers the hardware shapes.
struct ParsedGenieConfig {
    // dialog.type — selects decoding strategy.
    //   "basic"        — standard LLM (default)
    //   "ssd-q1"       — Self-Speculative Decoding
    //   "spd"          — Speculative Decoding
    //   "lade"         — Lookahead Decoding
    //   "kv-share"     — KV-share multi-engine
    //   "multistream"  — Multi-stream
    //   "eaglet"       — EAGLE-style speculation
    std::string dialog_type = "basic";

    // dialog.context tokens.
    int32_t              bos_token_id = -1;
    std::vector<int32_t> eos_token_ids;  // accepts scalar or array
    int32_t              pad_token_id = -1;

    // dialog.engine.model.positional-encoding.{rope-theta, rope-scaling}.
    // Falls back to dialog.engine.backend.QnnHtp.rope-theta if the explicit
    // positional-encoding block is absent.
    float       rope_theta   = 10000.0f;
    RopeScaling rope_scaling = StandardRope{};

    // dialog.embedding.{lut-path} — set when an external embedding LUT ships
    // with the bundle (VLM, 8B-LLM with off-graph embedding).
    std::optional<std::string> embedding_lut_path;
};

// ── Loader entry points ──────────────────────────────────────────────────────

// Reads metadata.json. Throws std::runtime_error on missing / malformed file.
GENIEX_API ParsedQAIRTMetadata parseQAIRTMetadata(const std::filesystem::path& bundle_dir);

// Reads genie_config.json. Returns an all-defaults struct if the file is
// absent (most bundles ship one, but it's not strictly required).
GENIEX_API ParsedGenieConfig parseGenieConfig(const std::filesystem::path& bundle_dir);

// Composes the metadata-derived fields with the genie-config-derived fields
// into a fully-populated LLMSpec.
GENIEX_API LLMSpec buildSpec(const ParsedQAIRTMetadata& meta, const ParsedGenieConfig& gc);

// Picks the matching RoPE input provider implementation from gc.rope_scaling.
// head_dim comes from meta; rope_theta from gc; falls back to standard RoPE.
GENIEX_API std::unique_ptr<InputProvider> makeRoPEProvider(
    const ParsedQAIRTMetadata& meta, const ParsedGenieConfig& gc);

// Picks the embedding-input provider for shard 0 based on its expected input
// tensor name. Returns a TokenIdInputProvider for "input_ids" or an
// EmbeddingInputProvider for "input_embeds" / "inputs_embeds".
GENIEX_API std::unique_ptr<InputProvider> makeEmbeddingProvider(
    const ParsedQAIRTMetadata& meta, const ParsedGenieConfig& gc);

// Returns the directory that contains the modelfile bundle for `model_cfg`.
// Inferred as the parent directory of model_cfg.model_paths[0].
GENIEX_API std::filesystem::path bundleDirOf(const ModelConfig& model_cfg);

// Convenience: derive a ModelConfig from a bundle directory by reading
// genie_config.json (for ctx-bins ordering), tokenizer.json, and
// htp_backend_ext_config.json.
GENIEX_API ModelConfig modelConfigFromDirectory(const std::filesystem::path& bundle_dir);

}  // namespace geniex
