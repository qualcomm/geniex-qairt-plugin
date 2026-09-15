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
// context binaries. We do NOT consult HuggingFace config.json for anything
// but architecture (see parseModelArchitecture).
//
// Bundle layout we depend on:
//   metadata.json     — QAIRT export metadata: shard wiring (model_files),
//                       architecture, vision preprocessing, and the `geniex`
//                       block (dialog_type, ctx_bins, RoPE, tokens, embedding
//                       LUTs, sampler). Parsed by parseQAIRTMetadata; LLM/VLM
//                       factories convert it via runtimeConfigFromMetadata.
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

// ── Parsed dialog.sampler block ──────────────────────────────────────────────
// Sampler defaults. Each field is optional so callers can fall through to
// their own defaults when a bundle omits a key.
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

// Fields read from metadata.json. hidden_size / num_kv_heads / head_dim /
// vocab_size / num_hidden_layers are NOT read as fields -- they are inferred
// from graph tensors (see LLMModel::inferSpecFromGraphs).
struct ParsedQAIRTMetadata {
    std::string model_id;  // e.g. "qwen3_4b", "llama_v3_2_3b_instruct_ssd"

    // Top-level `architectures[0]`. Empty if the bundle predates this field;
    // callers fall back to parseModelArchitecture(bundle) in that case.
    std::string architecture;

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

    // ── Fields read from metadata.json's `geniex` block ───────────────────────

    // geniex.dialog_type. Empty when the bundle predates the `geniex` block --
    // callers should treat that as "basic".
    std::string dialog_type;

    // geniex.ctx_bins — context-binary shard filenames, in load order.
    std::vector<std::string> ctx_bins;

    // Context length this bundle's graphs were compiled for.
    size_t max_context_length = 0;

    // Special tokens.
    int32_t              bos_token_id = -1;
    std::vector<int32_t> eos_token_ids;
    int32_t              pad_token_id = -1;

    // Global RoPE base + scaling.
    float       rope_theta   = 10000.0f;
    RopeScaling rope_scaling = StandardRope{};

    // Gemma3/4 sliding-window (local-attention) RoPE. local_positional_encoding_present
    // stays false for models with only global RoPE.
    bool        local_positional_encoding_present = false;
    float       local_rope_theta                  = 10000.0f;
    RopeScaling local_rope_scaling                = StandardRope{};

    // External embedding LUT (VLM / off-graph-embedding LLM bundles).
    std::optional<std::string> embedding_lut_path;
    QuantizedLutSpec           embedding_quant;

    // Gemma3/4 per-layer embedding stream (feeds `per_layer_inputs`).
    std::optional<std::string> perlayer_embedding_lut_path;
    size_t                     perlayer_embedding_size = 0;
    QuantizedLutSpec           perlayer_embedding_quant;

    // Sampler defaults.
    ParsedSamplerConfig sampler;
};

// ── Runtime config ───────────────────────────────────────────────────────────
// Subset of the runtime config the loader needs after metadata-driven
// inference covers the hardware shapes. Populated via runtimeConfigFromMetadata.
struct ParsedGenieConfig {
    // geniex.dialog_type — selects decoding strategy.
    //   "basic"        — standard LLM (default)
    //   "ssd-q1"       — Self-Speculative Decoding
    //   "spd"          — Speculative Decoding
    //   "lade"         — Lookahead Decoding
    //   "kv-share"     — KV-share multi-engine
    //   "multistream"  — Multi-stream
    //   "eaglet"       — EAGLE-style speculation
    std::string dialog_type = "basic";

    // geniex.context tokens.
    int32_t              bos_token_id = -1;
    std::vector<int32_t> eos_token_ids;  // accepts scalar or array
    int32_t              pad_token_id = -1;

    // geniex.positional_encoding.{rope_theta, rope_scaling}.
    float       rope_theta   = 10000.0f;
    RopeScaling rope_scaling = StandardRope{};

    // geniex.embedding.{lut_path} — set when an external embedding LUT ships
    // with the bundle (VLM, 8B-LLM with off-graph embedding). Resolved against
    // bundle_dir.
    std::optional<std::string> embedding_lut_path;

    // geniex.embedding.{datatype,quant_param} — set when that LUT is stored
    // quantized rather than float32. Leave default for float32 tables.
    QuantizedLutSpec embedding_quant;

    // ── Gemma3/4 extensions ──────────────────────────────────────────────────
    // geniex.local_positional_encoding.{rope_theta,rope_scaling} — the
    // sliding-window (local-attention) layers' RoPE. Present only for
    // Gemma-style dual-attention models; local_positional_encoding_present
    // stays false otherwise.
    bool        local_positional_encoding_present = false;
    float       local_rope_theta                  = 10000.0f;
    RopeScaling local_rope_scaling                = StandardRope{};

    // geniex.perlayer_embedding.{lut_path,size} — Gemma's per-layer embedding
    // stream (a second LUT feeding `per_layer_inputs`). size = num_layers *
    // per_layer_dim (E2B: 35*256 = 8960). lut_path resolved against bundle_dir.
    std::optional<std::string> perlayer_embedding_lut_path;
    size_t                     perlayer_embedding_size = 0;
    QuantizedLutSpec           perlayer_embedding_quant;
};

// ── Loader entry points ──────────────────────────────────────────────────────

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

// Converts a ParsedQAIRTMetadata into the ParsedGenieConfig shape
// buildSpecSkeleton/makeRoPEProvider/makeEmbeddingProvider/LLMModel expect.
// dialog_type is left at its "basic" default; callers needing the real
// dialog type read meta.dialog_type directly.
GENIEX_API ParsedGenieConfig runtimeConfigFromMetadata(const ParsedQAIRTMetadata& meta);

// Returns the directory that contains the modelfile bundle for `model_cfg`.
// Inferred as the parent directory of model_cfg.model_paths[0].
GENIEX_API std::filesystem::path bundleDirOf(const ModelConfig& model_cfg);

// Convenience: derive a ModelConfig from a bundle directory by reading
// metadata.json (for ctx-bins ordering and the embedding LUT path),
// tokenizer.json, and htp_backend_ext_config.json.
GENIEX_API ModelConfig modelConfigFromDirectory(const std::filesystem::path& bundle_dir);

// Number of HTP cores an htp_backend_ext_config.json requests: the size of the largest
// `devices[].cores` list. Returns 0 (leave the backend default) when the file is missing,
// unparsable, or carries no cores list.
GENIEX_API uint32_t parseHtpCoreCount(const std::filesystem::path& htp_config_path);

// Fills the HTP knobs from an htp_backend_ext_config.json: load-time keys only, since
// offline-preparation keys are already baked into the context binary. A missing or
// malformed file leaves `cfg` untouched.
//
// Schema: <qairt-sdk>/docs/QAIRT-Docs/QNN/general/htp/htp_backend.html
GENIEX_API void parseHtpConfig(const std::filesystem::path& htp_config_path, HtpPerfConfig& cfg);

// Resolves the HTP knobs to apply at load time: starts from the bundle's
// htp_backend_ext_config.json (via parseHtpConfig), then lets any knob the
// caller set on model_cfg (perf_profile, or a nonzero *_us duration) win over it.
GENIEX_API HtpPerfConfig resolveHtpPerfConfig(const ModelConfig& model_cfg);

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
