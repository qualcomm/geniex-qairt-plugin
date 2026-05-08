// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "model.h"
#include "types.h"
#include "llm/llm_types.h"
#include "llm/input_provider.h"
#include "geniex_export.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

namespace geniex {

// Thrown by LLMModel::generate when the prompt or the in-flight generation
class GENIEX_API ContextLengthExceededError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

class GENIEX_API LLMModel : public Model {
public:
    explicit LLMModel(LLMSpec spec);

    // Returns generated token IDs (excluding the prompt).
    // token_callback is called with each sampled token; return false to stop early.
    virtual std::vector<int32_t> generate(const std::vector<int32_t>& prompt_tokens,
                                  const GenerationConfig& gen_cfg = {},
                                  std::function<bool(int32_t)> token_callback = nullptr);

    virtual void resetKVCache();
    void saveKVCacheToFile(const std::string& path) const;
    void loadKVCacheFromFile(const std::string& path);

    size_t nPast() const;

    // Must be called before initialize().
    void addInputProvider(std::unique_ptr<InputProvider> provider);

protected:
    bool onInitialized() override;
    int32_t sampleNextToken(size_t phase, size_t token_offset = 0) const;

    static std::string fmtPattern(const std::string& pattern, size_t layer_idx);
    const StateBlockSpec& requireKVStateBlock() const;

    // phase * (shard_count_ * num_cl_) + shard * num_cl_ + cl_idx
    // phase: 0 = prefill, 1 = decode
    size_t graphIndex(size_t phase, size_t shard, size_t cl_idx) const;

    void runShard(size_t shard, size_t phase, size_t cl_idx, const LLMRunContext& ctx);

    // Strided copy of KV tokens between two distinct buffers (output→input after execution).
    // A flat memcpy would corrupt data because src/dst have different strides in the token dim.
    void copyKV(Graph& src_g, const std::string& src_name, bool src_is_output,
                Graph& dst_g, const std::string& dst_name,
                size_t src_off, size_t dst_off, size_t n_tok, bool is_key);
    void updateKV(size_t s, size_t phase, size_t dst_off, size_t n_tok);

    // Adjusts KV cache stride in-place when promoting to a larger context length.
    // Expanding iterates backward; contracting forward to handle overlapping regions safely.
    void reshapeKV(size_t shard, size_t old_kv_len, size_t new_kv_len, size_t n_valid);

    // Promotes active_cl_idx_ to the smallest CL where (CL - capacity_reserved_seq) >= required,
    // restriding all KV layers from the current CL to the new CL at stride
    bool promoteCL(size_t required,
                   size_t capacity_reserved_seq,
                   size_t stride_reserved_seq);

    LLMSpec spec_;
    std::vector<std::unique_ptr<InputProvider>> input_providers_;

    size_t shard_count_ = 0;  // total graphs = 2 × shard_count_ × num_cl_
    size_t num_cl_      = 0;
    size_t active_cl_idx_ = 0;  // index into spec_.context_lengths; advances during prefill

    std::vector<std::vector<Connection>> shard_hidden_state_;        // outer index = CL variant; hidden state across adjacent prefill shards
    std::vector<std::vector<Connection>> decode_shard_hidden_state_; // same for decode shards

    size_t kv_state_block_idx_ = std::numeric_limits<size_t>::max();
    size_t n_past_ = 0;

private:
    void buildConnections();

    // KV input tensor names across all shards, derived from the KV state block patterns.
    std::unordered_set<std::string> buildKVInputNameSet() const;
};

} // namespace geniex
