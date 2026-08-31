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

// Reads a QAIRT model bundle's JSON sidecars and produces the pieces of an
// LLMSpec that do NOT require the compiled graphs: the JSON-sourced fields
// (RoPE base/scaling, EOS/BOS, dialog type) via buildSpecSkeleton, plus the
// matching CPU-side InputProviders. Everything a tensor can carry (hidden
// size, KV heads, head dim, vocab, shard wiring, KV pairs) is filled later
// by LLMModel::inferSpecFromGraphs, once the HTP backend has loaded the
// context binaries. We do NOT consult HuggingFace config.json.
//
// Bundle layout we depend on:
//   genie_config.json — runtime config: dialog.type, context tokens, RoPE
//                       parameters, embedding LUT spec.
//   tokenizer.json    — sentencepiece/BPE tokenizer (read by LLMPipeline).
//   *.bin             — compiled context-binary shards.
//   metadata.json     — QAIRT export metadata. LLM path no longer reads it;
//                       retained only for VLM vision preprocessing and for
//                       model_id-based dispatch (see parseQAIRTMetadata).

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

// Fields read from metadata.json. Retained for the VLM path (vision
// preprocessing + vision-encoder hidden size). The LLM path infers all of
// these from graph tensors instead (see LLMModel::inferSpecFromGraphs).
struct ParsedQAIRTMetadata {
    std::string model_id;  // e.g. "qwen3_4b", "llama_v3_2_3b_instruct_ssd"

    std::vector<ShardSpec> shards;

    size_t hidden_size       = 0;  // inputs_embeds.shape[2] / hidden-state.shape[2]
    size_t num_kv_heads      = 0;  // past_key_*.shape[0]
    size_t head_dim          = 0;  // past_key_*.shape[2]
    size_t vocab_size        = 0;  // logits last dim
    size_t num_hidden_layers = 0;  // max past_key_<N>_in across all shards + 1

    std::optional<ParsedVisionPreprocessing> vision_preprocessing;
    std::string                              vision_encoder_graph;  // empty if absent

    // Raw JSON key of shard 0's first non-special input ("input_ids" or
    // "inputs_embeds"). Drives the embedding-provider factory choice between
    // TokenIdInputProvider and EmbeddingInputProvider.
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

    // dialog.embedding.{datatype,quant-param} — set when that LUT is stored
    // quantized rather than float32. Leave default for float32 tables.
    QuantizedLutSpec embedding_quant;

    // ── Gemma3/4 extensions ──────────────────────────────────────────────────
    // dialog.engine.model.local-positional-encoding.{rope-theta,rope-scaling}
    // — the sliding-window (local-attention) layers' RoPE. Present only for
    // Gemma-style dual-attention models; local_positional_encoding_present
    // stays false otherwise.
    bool        local_positional_encoding_present = false;
    float       local_rope_theta                  = 10000.0f;
    RopeScaling local_rope_scaling                = StandardRope{};

    // dialog.perlayer-embedding.{lut-path,size} — Gemma's per-layer embedding
    // stream (a second LUT feeding `per_layer_inputs`). size = num_layers *
    // per_layer_dim (E2B: 35*256 = 8960).
    std::optional<std::string> perlayer_embedding_lut_path;
    size_t                     perlayer_embedding_size = 0;
    QuantizedLutSpec           perlayer_embedding_quant;
};

// ── Parsed dialog.sampler block ──────────────────────────────────────────────
// Sampler defaults baked into genie_config.json. Each field is optional so
// callers can fall through to their own defaults when a bundle omits a key.
struct ParsedSamplerConfig {
    std::optional<uint32_t> seed;
    std::optional<float>    temperature;
    std::optional<int32_t>  top_k;
    std::optional<float>    top_p;
    std::optional<float>    repetition_penalty;
    std::optional<float>    presence_penalty;
    std::optional<float>    frequency_penalty;
    std::optional<int32_t>  penalty_last_n;
};

// ── Loader entry points ──────────────────────────────────────────────────────

// Reads genie_config.json. Returns an all-defaults struct if the file is
// absent (most bundles ship one, but it's not strictly required).
GENIEX_API ParsedGenieConfig parseGenieConfig(const std::filesystem::path& bundle_dir);

// Reads `dialog.sampler` from genie_config.json. Returns an all-nullopt struct
// if the file or block is missing.
GENIEX_API ParsedSamplerConfig parseGenieSamplerConfig(const std::filesystem::path& bundle_dir);

// Builds an LLMSpec with only the JSON-sourced fields (eos/bos tokens, a
// default KV state block). LLMModel::inferSpecFromGraphs fills the rest once
// the graphs load.
GENIEX_API LLMSpec buildSpecSkeleton(const ParsedGenieConfig& gc);

// Selects the RoPE provider variant from gc.rope_scaling. head_dim is resolved
// by the caller from the cos tensor. cos_name/sin_name name the graph inputs to
// write; they default to the classic position_ids_cos/sin, but newer exports
// rename the global-RoPE pair to position_ids_global_cos/sin, so the caller
// passes whichever pair the graph actually exposes.
GENIEX_API std::unique_ptr<InputProvider> makeRoPEProvider(size_t head_dim, const ParsedGenieConfig& gc,
    std::string cos_name = "position_ids_cos", std::string sin_name = "position_ids_sin");

// Selects the embedding provider from the first-shard input tensor name.
GENIEX_API std::unique_ptr<InputProvider> makeEmbeddingProvider(
    const std::string& first_shard_input, const ParsedGenieConfig& gc);

// Reads metadata.json. Retained for the VLM path, whose vision-encoder shapes
// are not carried by the LLM graph tensors.
GENIEX_API ParsedQAIRTMetadata parseQAIRTMetadata(const std::filesystem::path& bundle_dir);

// Returns the directory that contains the modelfile bundle for `model_cfg`.
// Inferred as the parent directory of model_cfg.model_paths[0].
GENIEX_API std::filesystem::path bundleDirOf(const ModelConfig& model_cfg);

// Convenience: derive a ModelConfig from a bundle directory by reading
// genie_config.json (for ctx-bins ordering), tokenizer.json, and
// htp_backend_ext_config.json.
GENIEX_API ModelConfig modelConfigFromDirectory(const std::filesystem::path& bundle_dir);

// Number of HTP cores an htp_backend_ext_config.json requests: the size of the
// largest `devices[].cores` list. Returns 0 (leave backend default) when the
// file is missing, unparsable, or carries no cores list — the JSON is otherwise
// consumed only by the closed-source QnnHtpNetRunExtensions library, so this is
// the one place GenieX reads it back for validation and multicore defaulting.
GENIEX_API uint32_t parseHtpCoreCount(const std::filesystem::path& htp_config_path);

// Reads `architectures[0]` from the bundle's HuggingFace-style config.json
// (e.g. "Phi3ForCausalLM", "Qwen3ForCausalLM"). This is the family signal
// ai-hub-models exports, so dispatch keys behavioural knobs (e.g. BOS) off it
// instead of pattern-matching the bundle's own model_id -- ai-hub-models is then
// free to name model_id however it likes without touching GenieX.
//
// Not a substitute for the bundle's own signals where those disagree: a VLM
// text tower can report the same architecture as a standalone LLM of that
// family (InternVL's Qwen3 tower reports "Qwen3ForCausalLM" with nothing to
// tell it apart), and some exports (Qwen3-VL, Gemma4) ship no `architectures`
// key at all. Multimodality is metadata.json's concern (vision_preprocessing /
// vision_encoder_graph), not this function's.
//
// Returns "" if config.json is missing, unparsable, or carries no
// `architectures` array -- callers must treat that as "unknown", not as a
// specific family.
GENIEX_API std::string parseModelArchitecture(const std::filesystem::path& bundle_dir);

}  // namespace geniex
