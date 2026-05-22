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

// Context describing a single forward-pass step in an LLM inference loop.
struct LLMRunContext {
    const std::vector<int32_t>& token_ids;  // token IDs for the current chunk/step
    size_t                      n_past;     // KV positions already filled
    size_t                      curr_len;   // number of tokens in this chunk/step
    size_t                      phase;      // 0 = prefill, 1 = decode
};

// Inclusive KV layer range [begin, end] owned by a single shard.
struct LayerRange {
    size_t begin;
    size_t end;  // inclusive
};

// Per-shard descriptor for hidden-state wiring between adjacent shards.
struct ShardSpec {
    std::string in_state_name;
    std::string out_state_name;
    bool        lm_head_only = false;
};

enum class StateBlockKind {
    KV,
};

// State/cache declaration for a shard-partitioned decoder runtime.
// For KV blocks, layer ranges are defined per shard and patterns map to
// key/value in/out tensors with {} replaced by layer index.
struct StateBlockSpec {
    std::string    name = "kv_default";
    StateBlockKind kind = StateBlockKind::KV;

    // Per-shard ownership of this state block.
    // Each shard may own zero, one, or many disjoint layer ranges.
    std::vector<std::vector<LayerRange>> shard_layer_ranges;

    std::string key_in_pattern    = "past_key_{}_in";
    std::string value_in_pattern  = "past_value_{}_in";
    std::string key_out_pattern   = "past_key_{}_out";
    std::string value_out_pattern = "past_value_{}_out";
};

inline std::vector<std::vector<LayerRange>> makeShardLayerRanges(
    std::vector<std::optional<LayerRange>> shard_layer_ranges) {
    std::vector<std::vector<LayerRange>> ranges;
    ranges.reserve(shard_layer_ranges.size());
    for (auto&& range : shard_layer_ranges) {
        if (range) {
            ranges.push_back({*range});
        } else {
            ranges.push_back({});
        }
    }
    return ranges;
}

inline StateBlockSpec makeKVOnlyStateBlock(
    std::vector<std::optional<LayerRange>> shard_layer_ranges, std::string name = "kv_default") {
    StateBlockSpec block;
    block.name               = std::move(name);
    block.kind               = StateBlockKind::KV;
    block.shard_layer_ranges = makeShardLayerRanges(std::move(shard_layer_ranges));
    return block;
}

inline StateBlockSpec makeKVOnlyStateBlock(
    std::vector<std::vector<LayerRange>> shard_layer_ranges, std::string name = "kv_default") {
    StateBlockSpec block;
    block.name               = std::move(name);
    block.kind               = StateBlockKind::KV;
    block.shard_layer_ranges = std::move(shard_layer_ranges);
    return block;
}

// Architecture and tensor naming parameters for a split-decoder LLM.
struct LLMSpec {
    // Per-shard hidden-state descriptors, in order.
    std::vector<ShardSpec> shards;

    // State/cache blocks. For current models this is usually a single KV block.
    std::vector<StateBlockSpec> state_blocks;

    size_t seq_len_prefill = 0;
    size_t seq_len_decode  = 0;
    size_t hidden_size     = 0;
    size_t num_kv_heads    = 0;
    size_t head_dim        = 0;
    size_t vocab_size      = 0;

    // KV buffer size variants, sorted ascending. e.g. {640, 1152, 2176, 4096}.
    // Single entry = one fixed context length.
    std::vector<size_t> context_lengths = {4096};

    // Pattern for inferring (phase, shard, cl_idx) from graph names at load time.
    // Placeholders: {ar}    = AR sequence length (prefill or decode)
    //               {cl}    = context length
    //               {shard} = 1-based shard index
    //               {total} = total number of shards
    //               {phase} = phase string (e.g. "prefill" or "decode"), used by some
    //                         model export conventions; not used by sortKey() itself.
    // Additional placeholders beyond the ones above are allowed but are silently ignored
    // by sortKey() when parsing graph names.
    // Default matches Qualcomm export convention, e.g. "ar128_cl4096_1_of_2".
    std::string graph_name_pattern = "ar{ar}_cl{cl}_{shard}_of_{total}";

    // Name of the attention mask input tensor; written by runShard() on every shard.
    std::string attention_mask_name = "attention_mask";

    std::vector<int32_t> eos_token_ids;
};

}  // namespace geniex
