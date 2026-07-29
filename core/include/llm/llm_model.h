// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

#include "geniex-proc/sampler.h"
#include "geniex_export.h"
#include "llm/input_provider.h"
#include "llm/llm_spec_loader.h"  // ParsedGenieConfig
#include "llm/llm_types.h"
#include "model.h"
#include "threadpool.h"
#include "types.h"

namespace geniex {

// Thrown by LLMModel::generate when the prompt or the in-flight generation
class GENIEX_API ContextLengthExceededError : public std::runtime_error {
   public:
    using std::runtime_error::runtime_error;
};

class GENIEX_API LLMModel : public Model {
   public:
    explicit LLMModel(LLMSpec spec, ParsedGenieConfig gc = {});
    ~LLMModel() override;

    // Restored after the user-declared destructor suppressed the implicit moves.
    LLMModel(LLMModel&&) noexcept            = default;
    LLMModel& operator=(LLMModel&&) noexcept = default;

    // Returns generated token IDs (excluding the prompt).
    // token_callback is called with each sampled token; return false to stop early.
    virtual std::vector<int32_t> generate(const std::vector<int32_t>& prompt_tokens,
        const GenerationConfig& gen_cfg = {}, std::function<bool(int32_t)> token_callback = nullptr);

    // Raw logits from a single (non-autoregressive) forward pass over `tokens`.
    // No sampling, no decode loop -- runs the prefill path and reads the LM-head
    // output directly. Intended for on-target metrics (perplexity, MMLU, MMMU)
    // that score model outputs rather than generate text.
    //
    // all_positions == false (default): returns the last token's logits row,
    //   `vocab_size` floats. Sufficient for next-token / multiple-choice scoring.
    // all_positions == true: returns every position's logits, row-major
    //   [tokens.size(), vocab_size] (tokens.size() * vocab_size floats), so the
    //   caller can compute the per-token log-likelihoods perplexity needs.
    //
    // Runs against a fresh KV cache: it calls resetKVCache() on entry so the
    // result depends only on `tokens`, and again on exit so the model is left
    // clean for the next call. Throws ContextLengthExceededError if `tokens`
    // does not fit the largest context length, and std::invalid_argument if
    // `tokens` is empty. Not affected by, and does not touch, generate()'s
    // sampler state.
    virtual std::vector<float> forwardLogits(const std::vector<int32_t>& tokens, bool all_positions = false);

    virtual void resetKVCache();
    void         saveKVCacheToFile(const std::string& path) const;
    void         loadKVCacheFromFile(const std::string& path);

    size_t nPast() const;

    // Vocabulary size inferred from the LM-head graph's logits tensor (last
    // dim). 0 if the model has not been initialized yet.
    size_t vocabSize() const;

    // Must be called before initialize().
    void addInputProvider(std::unique_ptr<InputProvider> provider);

    // Returns the EmbeddingInputProvider that writes `tensor_name`, or nullptr.
    //
    // Resolved here, inside geniex_core, on purpose: the providers are
    // constructed in this library, and a dynamic_cast performed in a consumer
    // binary would have to match RTTI across the DLL boundary.
    EmbeddingInputProvider* findEmbeddingProvider(const std::string& tensor_name);

   protected:
    bool onInitialized() override;

    // Fills spec_'s tensor-derived fields (shapes, shard wiring, KV pairs) from
    // the loaded graphs_, which onInitialized has sorted by (phase, shard, cl).
    // Sole source of truth for hyperparameters; throws if a field can't resolve.
    void inferSpecFromGraphs();

    // Resolves the KV tensor pairs a shard graph owns. Layer indices are global
    // and may be non-zero-based and non-contiguous across shards, so they are
    // read from the matched tensor names rather than assumed.
    static std::vector<KVTensorPair> discoverKVPairs(const Graph& g, const StateBlockSpec& block);

    // Builds the CPU-side input providers after the spec is inferred.
    // Subclasses override to supply modality-specific providers.
    virtual void createInputProviders();

    // Reads the last logits row, then either runs the cached sampler chain
    // (advancing penalty / DRY state) or returns argmax when sampler_ is null.
    int32_t sampleNextToken(size_t phase, size_t token_offset = 0);

    // Stops on configured EOS ids and on any token the tokenizer flags as end-of-generation.
    bool isEndOfGeneration(int32_t token, const GenerationConfig& gen_cfg) const;

    // Reads the last logits row from the LM-head output. Shared by the
    // greedy fast path and the sampler-driven path.
    void readLastLogits(size_t phase, size_t token_offset, std::vector<float>& out) const;

    // (Re)build the cached sampler from `gen_cfg` and seed it with this
    // turn's prompt. Called once at the top of generate(). Reuses the
    // existing sampler when config is unchanged so penalty / DRY history
    // persists across multi-turn calls. No-op when sampling is disabled.
    void prepareSampler(const GenerationConfig& gen_cfg, const std::vector<int32_t>& prompt_tokens);

    const StateBlockSpec& requireKVStateBlock() const;

    // phase * (shard_count_ * num_cl_) + shard * num_cl_ + cl_idx
    // phase: 0 = prefill, 1 = decode
    size_t graphIndex(size_t phase, size_t shard, size_t cl_idx) const;

    void runShard(size_t shard, size_t phase, size_t cl_idx, const LLMRunContext& ctx);

    // Strided copy of KV tokens between two distinct buffers (output→input after execution).
    // A flat memcpy would corrupt data because src/dst have different strides in the token dim.
    void copyKV(Graph& src_g, const std::string& src_name, bool src_is_output, Graph& dst_g,
        const std::string& dst_name, size_t src_off, size_t dst_off, size_t n_tok, bool is_key);
    void updateKV(size_t s, size_t phase, size_t dst_off, size_t n_tok);

    // Token capacity (kv_len) of a KV input tensor, read from its shape.
    size_t kvCapacityOf(Graph& g, const std::string& name, bool is_key) const;
    // Shift a fixed-window KV input buffer left by `shift` tokens (drop oldest),
    // making room to append at the tail. Used by sliding-window (swa_*) caches.
    void shiftKVLeft(Graph& g, const std::string& name, size_t shift, bool is_key);

    // Adjusts KV cache stride in-place when promoting to a larger context length.
    // Expanding iterates backward; contracting forward to handle overlapping regions safely.
    void reshapeKV(size_t shard, size_t old_kv_len, size_t new_kv_len, size_t n_valid);

    // Promotes active_cl_idx_ to the smallest CL where (CL - capacity_reserved_seq) >= required,
    // restriding all KV layers from the current CL to the new CL at stride
    bool promoteCL(size_t required, size_t capacity_reserved_seq, size_t stride_reserved_seq);

    // Number of oldest tokens (above n_keep) to discard so `n_fit` more fit within max_cl.
    // Mirrors llama.cpp's context-shift heuristic (~half of n_past - n_keep, or more if
    // needed to fit n_fit). Returns 0 when n_past <= n_keep.
    static size_t computeSlideDiscard(size_t n_past, size_t n_fit, size_t max_cl, size_t n_keep);

    // Evicts the oldest `n_discard` tokens above the anchored `n_keep` prefix, then re-prefills
    // the surviving tail (recovered from token_history_) instead of relocating its cached KV --
    // QAIRT's compiled graphs cache post-RoPE K/V with no facility to re-rotate cached history,
    // so a byte relocation would leave survivors' RoPE rotation at an out-of-distribution
    // position. `at_decode_stride` must be true when called mid-decode-loop. Updates n_past_ and
    // token_history_.
    void slideWindowEvict(size_t n_discard, size_t n_keep, bool at_decode_stride);

    // Runs a chunked prefill pass over `tokens`, writing fresh KV starting at the current n_past_
    // and advancing n_past_ (and token_history_) as each chunk completes. Assumes the KV buffer is
    // currently strided for prefill; callers coming from decode stride must reshapeKV first (see
    // slideWindowEvict). If `last_chunk_size_out` is non-null, receives the final chunk's token
    // count -- needed by generate() to sample the first token, but not by slideWindowEvict's
    // re-prefill, which re-derives already-generated history and passes nullptr.
    //
    // If `all_logits_out` is non-null, every chunk's LM-head logits are appended to it row-major
    // ([chunk_size, vocab_size] per chunk), so the full pass yields [tokens.size(), vocab_size].
    // This also forces the LM-head shard to run on every chunk (the non-final-chunk skip that
    // generate()/slideWindowEvict rely on is suppressed), which costs extra compute -- pass nullptr
    // (the default) unless per-position logits are actually needed. Used by forwardLogits().
    void prefillChunks(
        const std::vector<int32_t>& tokens, size_t* last_chunk_size_out, std::vector<float>* all_logits_out = nullptr);

    LLMSpec                                     spec_;
    ParsedGenieConfig                           gc_;  // JSON-sourced RoPE / token config
    std::vector<std::unique_ptr<InputProvider>> input_providers_;
    size_t                                      shard_count_ = 0;  // total graphs = 2 × shard_count_ × num_cl_
    size_t                                      num_cl_      = 0;
    size_t active_cl_idx_ = 0;  // index into spec_.context_lengths; advances during prefill

    std::vector<std::vector<Connection>>
        shard_hidden_state_;  // outer index = CL variant; hidden state across adjacent prefill shards
    std::vector<std::vector<Connection>> decode_shard_hidden_state_;  // same for decode shards

    size_t kv_state_block_idx_ = std::numeric_limits<size_t>::max();
    size_t n_past_             = 0;

    // Token IDs resident in the KV cache: token_history_[i] == the token at KV position i.
    // token_history_.size() == n_past_ always. Populated by prefillChunks() and generate()'s
    // decode loop; truncated by slideWindowEvict() and cleared by resetKVCache(). Exists so
    // slideWindowEvict can recover the surviving tail's token IDs to re-prefill them.
    std::vector<int32_t> token_history_;

    // Cached sampler state. `sampler_` is null when sampling is disabled
    // (greedy fast path); otherwise it persists across multi-turn calls so
    // penalty / DRY history spans the conversation. `sampler_cfg_` records
    // the config used to build it; prepareSampler() rebuilds when it changes.
    std::unique_ptr<Sampler> sampler_;
    GenerationConfig         sampler_cfg_;
    bool                     sampler_cfg_valid_ = false;

    // Background workers overlapping KV write-back with HTP execute during decode.
    // Also hosts the clock-keeper spinners (active across the decode window).
    std::unique_ptr<ThreadPool> decode_pool_;
    unsigned                    clock_keeper_threads_ = 1;  // GENIEX_CLOCK_KEEPER_THREADS overrides (0 = off)
    uint64_t                    decode_cpu_mask_      = 0;  // shared by KV workers and clock keeper

   private:
    void buildConnections();

    // KV input tensor names across all shards, taken from the resolved KV pairs.
    std::unordered_set<std::string> buildKVInputNameSet() const;

    // One-shot fill of every KV input buffer with the encoded-zero pattern
    // for its dtype
    void initKVBuffers();
};

}  // namespace geniex
