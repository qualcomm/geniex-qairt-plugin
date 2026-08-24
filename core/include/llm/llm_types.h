// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace geniex {

// Quantization of an on-disk embedding lookup table.
//
// Large-vocab models ship their embedding LUTs quantized: dequantizing them
// into RAM is not an option (Gemma4-E2B's per-layer table is 2.35 GB as int8,
// 9.4 GB as float32). A table carrying one of these is memory-mapped and rows
// are converted on demand -- see EmbeddingInputProvider.
//
// Follows the QNN convention: real = scale * (stored + offset), with `offset`
// the negated zero-point (so it is normally negative).
struct QuantizedLutSpec {
    // "ufixed8" / "ufixed16" / "sfixed8" / "sfixed16"; empty or "float32"
    // means the table is plain float32 and needs no conversion.
    std::string datatype;
    float       scale  = 1.0f;
    int32_t     offset = 0;

    bool quantized() const { return !datatype.empty() && datatype != "float32"; }

    bool isSigned() const { return !datatype.empty() && datatype.front() == 's'; }

    // Bytes per stored element, from the trailing bit width in `datatype`.
    size_t elementBytes() const {
        if (!quantized()) return 4;
        return datatype.size() >= 2 && datatype.compare(datatype.size() - 2, 2, "16") == 0 ? 2 : 1;
    }
};

// Context describing a single forward-pass step in an LLM inference loop.
struct LLMRunContext {
    const std::vector<int32_t>& token_ids;  // token IDs for the current chunk/step
    size_t                      n_past;     // KV positions already filled
    size_t                      curr_len;   // number of tokens in this chunk/step
    size_t                      phase;      // 0 = prefill, 1 = decode
};

// Per-shard descriptor for hidden-state wiring between adjacent shards.
struct ShardSpec {
    std::string in_state_name;
    std::string out_state_name;
    bool        lm_head_only = false;
};

enum class StateBlockKind {
    KV,               // global key/value cache; grows with the context-length variant
    SlidingWindowKV,  // fixed-window (swa_*) cache; capacity is independent of CL
};

// True for any block that holds key/value cache state (global or sliding-window).
inline bool isKVStateBlock(StateBlockKind kind) {
    return kind == StateBlockKind::KV || kind == StateBlockKind::SlidingWindowKV;
}

// The four graph tensor names that carry one key/value cache entry.
struct KVTensorPair {
    std::string key_in;
    std::string key_out;
    std::string value_in;
    std::string value_out;
};

// Declares one key/value cache owned by a shard-partitioned decoder.
// The patterns name its tensors ({} = layer index); shard_pairs holds the
// tensors each shard owns, resolved during initialization.
struct StateBlockSpec {
    std::string    name = "kv_default";
    StateBlockKind kind = StateBlockKind::KV;

    std::string key_in_pattern    = "past_key_{}_in";
    std::string value_in_pattern  = "past_value_{}_in";
    std::string key_out_pattern   = "past_key_{}_out";
    std::string value_out_pattern = "past_value_{}_out";

    std::vector<std::vector<KVTensorPair>> shard_pairs;
};

inline StateBlockSpec makeKVStateBlock(std::string name = "kv_default") {
    StateBlockSpec block;
    block.name = std::move(name);
    block.kind = StateBlockKind::KV;
    return block;
}

// Gemma3/4 second KV cache: the sliding-window (local-attention) layers keep a
// separate `swa_*` key/value cache, distinct from the global `past_*` cache and
// with its own head dim. Declared as a second StateBlockSpec.
inline StateBlockSpec makeSwaKVStateBlock(std::string name = "kv_swa") {
    StateBlockSpec block;
    block.name              = std::move(name);
    block.kind              = StateBlockKind::SlidingWindowKV;
    block.key_in_pattern    = "swa_key_{}_in";
    block.value_in_pattern  = "swa_value_{}_in";
    block.key_out_pattern   = "swa_key_{}_out";
    block.value_out_pattern = "swa_value_{}_out";
    return block;
}

// Architecture and tensor naming parameters for a split-decoder LLM.
struct LLMSpec {
    std::vector<ShardSpec>      shards;
    std::vector<StateBlockSpec> state_blocks;

    // Inferred from the loaded graph tensors.
    size_t hidden_size = 0;

    // Physical KV heads per tensor (post-split: so a one-tensor-per-head export reports 1),
    // not the logical count.
    size_t num_kv_heads = 0;
    size_t head_dim     = 0;
    size_t vocab_size   = 0;

    // Inferred from the loaded graph names.
    size_t              seq_len_prefill = 0;
    size_t              seq_len_decode  = 0;
    std::vector<size_t> context_lengths;

    // Minimum token slots the decode phase needs. 0 (default) = use the smallest
    // AR variant the bundle ships.
    //
    // A bundle may ship MORE than two AR widths: the Llama-3.2-3B SSD w4a16 export
    // carries ar1 (plain single-token), ar32 (speculative tree) and ar128 (prefill).
    // The runtime addresses exactly two phases, so a driver whose decode pass needs
    // a specific width says so here and onInitialized() picks the smallest AR that
    // fits, ignoring the rest. SSD sets this to its tree size (30 -> ar32).
    size_t min_decode_seq_len = 0;

    std::string attention_mask_name = "attention_mask";

    // Scatter caches expose this int32 input: the runtime writes the cache column
    // where the current pass's fresh KV is to be placed, and the graph both reads
    // the CL-wide cache and drops its freshly computed KV in at that column.
    std::string cache_index_name = "cache_index";

    // Gemma3/4 sliding-window (local) attention: a second causal mask that is
    // additionally band-limited to the last `swa_window` key positions. Written
    // only when a shard graph actually exposes this input (Gemma), so it is
    // harmless for single-stream models.
    std::string swa_attention_mask_name = "swa_attention_mask";
    size_t      swa_window              = 512;

    std::vector<int32_t> eos_token_ids;

    // -1 = no BOS configured.
    int32_t bos_token_id = -1;
};

}  // namespace geniex
