// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "cb/input_provider.h"
#include "cb/kv_cache_manager.h"
#include "cb/scheduler.h"
#include "cb/session.h"
#include "llm/llm_model.h"
#include "llm/llm_types.h"
#include "logging.h"
#include "utils.h"

namespace geniex {
namespace cb {

// Continuous-batching LLMModel. Concatenates inputs from multiple sessions
// along the sequence dimension and always runs the prefill graph (phase=0);
// during decode each active session contributes a single token.
//
// Owns the model-agnostic loop — scheduling, KV bookkeeping, attention
// mask, buffer moves, per-session KV copy, CL promotion, sampling.
// Model-specific tensor writes (token-ids / embeddings / RoPE cos/sin)
// are injected via CBInputProvider implementations registered with
// addCBProvider().
//
// Usage:
//   auto model = mymodel::makeCBModel();
//   model.initialize(runtime_cfg, model_cfg);
//
//   Scheduler       sched;
//   KVCacheManager  kv_mgr;
//   sched.addSession("s0", prompt0, max_tokens);
//   sched.addSession("s1", prompt1, max_tokens);
//
//   model.generateBatch(sched, kv_mgr,
//       [&](const std::string& sid, int32_t tok) { /* stream */ });
class CBLLMModel : public LLMModel {
   public:
    explicit CBLLMModel(LLMSpec spec, ParsedGenieConfig gc = {}) : LLMModel(std::move(spec), std::move(gc)) {}

    void addCBProvider(std::unique_ptr<CBInputProvider> provider) { cb_providers_.push_back(std::move(provider)); }

    // Drives all scheduled sessions until each hits EOS or max_tokens.
    void generateBatch(Scheduler& scheduler, KVCacheManager& kv_mgr,
        std::function<void(const std::string& session_id, int32_t token)> token_callback = nullptr) {
        const int seq_len = static_cast<int>(spec_.seq_len_prefill);

        while (scheduler.hasActiveSessions()) {
            // 1. Pick sessions for this step.
            std::vector<Session*> batch;
            const int             count = scheduler.getNextBatch(batch, seq_len);
            if (count == 0) break;

            std::vector<std::vector<int32_t>> per_session_tokens(batch.size());
            std::vector<int>                  seg_lengths(batch.size());
            for (size_t i = 0; i < batch.size(); ++i) {
                Session& s = *batch[i];
                if (s.processed_length < s.query_len) {
                    // Prefill: take as much of the remaining prompt as fits
                    // in the sequence budget left by earlier sessions.
                    const int rem          = s.query_len - s.processed_length;
                    int       already_used = 0;
                    for (size_t j = 0; j < i; ++j) already_used += seg_lengths[j];
                    const int chunk = std::min(rem, seq_len - already_used);
                    per_session_tokens[i].assign(s.query_tokens.begin() + s.processed_length,
                        s.query_tokens.begin() + s.processed_length + chunk);
                    seg_lengths[i] = chunk;
                } else {
                    per_session_tokens[i] = {s.pending_token};
                    seg_lengths[i]        = 1;
                }
            }

            // 2. Allocate new sessions, promote CL if needed, shift for growth.
            for (size_t i = 0; i < batch.size(); ++i) {
                if (!kv_mgr.getSegment(batch[i]->id)) kv_mgr.allocate(batch[i]->id, 0);
            }

            int                                      total_growth = 0;
            std::vector<std::pair<std::string, int>> growth_list;
            for (size_t i = 0; i < batch.size(); ++i) {
                growth_list.push_back({batch[i]->id, seg_lengths[i]});
                total_growth += seg_lengths[i];
            }

            const int kv_used     = kv_mgr.totalUsed();
            const int required_kv = kv_used + total_growth;
            selectCL(required_kv, seq_len, kv_used);

            const int kv_len = static_cast<int>(spec_.context_lengths[active_cl_idx_]) - seq_len;

            auto shift_moves = kv_mgr.shiftForGrowth(growth_list);
            for (const auto& m : shift_moves) moveKVBuffer(m.src, m.dst, m.len);

            // 3. Build step context for the providers.
            CBStepContext ctx;
            ctx.sessions = batch;
            ctx.seq_len  = seq_len;
            ctx.kv_len   = kv_len;
            ctx.concat_tokens.reserve(seq_len);
            int off = 0;
            for (size_t i = 0; i < batch.size(); ++i) {
                ctx.in_segs.push_back({off, seg_lengths[i]});
                ctx.concat_tokens.insert(
                    ctx.concat_tokens.end(), per_session_tokens[i].begin(), per_session_tokens[i].end());
                off += seg_lengths[i];
            }
            ctx.concat_tokens.resize(seq_len, 0);
            for (auto* s : batch) ctx.kv_segs.push_back(*kv_mgr.getSegment(s->id));

            // 4. Attention mask — block-diagonal causal, shared across shards.
            std::vector<float> mask;
            KVCacheManager::getAttentionMask(ctx.kv_segs, ctx.in_segs, seq_len, kv_len, mask);

            // 5. Run all shards.
            for (size_t s = 0; s < shard_count_; ++s) {
                Graph& g = graph(graphIndex(/*phase=*/0, s, active_cl_idx_));

                if (g.hasInput(spec_.attention_mask_name)) g.write(spec_.attention_mask_name, mask.data(), mask.size());

                for (auto& p : cb_providers_) p->write(g, ctx);

                TimeLog tl;
                if (!g.execute(tl)) {
                    throw std::runtime_error("CB graph execute failed: shard=" + std::to_string(s) +
                                             " cl_idx=" + std::to_string(active_cl_idx_));
                }

                copyNewKV(s, batch, ctx.kv_segs, ctx.in_segs, seg_lengths);

                if (s + 1 < shard_count_) applyConnections({shard_hidden_state_[active_cl_idx_][s]});
            }

            // 6. Update scheduler + KV lengths.
            for (size_t i = 0; i < batch.size(); ++i) {
                scheduler.updateSession(batch[i]->id, seg_lengths[i]);
                kv_mgr.extend(batch[i]->id, seg_lengths[i]);
            }

            // 7. Sample for each session whose prefill just finished.
            int logit_off = 0;
            for (size_t i = 0; i < batch.size(); ++i) {
                Session&   s            = *batch[i];
                const bool done_prefill = (s.processed_length >= s.query_len);

                if (done_prefill) {
                    const int     last_pos = logit_off + seg_lengths[i] - 1;
                    const int32_t tok      = sampleNextToken(/*phase=*/0, static_cast<size_t>(last_pos));

                    bool eos = false;
                    for (int32_t e : spec_.eos_token_ids)
                        if (tok == e) {
                            eos = true;
                            break;
                        }

                    if (eos || s.generated_len >= s.max_tokens) {
                        scheduler.completeSession(s.id);
                    } else {
                        s.generated_tokens.push_back(tok);
                        s.generated_len++;
                        s.pending_token = tok;
                        if (token_callback) token_callback(s.id, tok);
                    }
                }

                logit_off += seg_lengths[i];
            }

            // 8. Release completed sessions and defragment the KV buffer.
            for (auto& s : scheduler.sessions()) {
                if (s.status == SessionStatus::COMPLETED && kv_mgr.getSegment(s.id)) {
                    kv_mgr.release(s.id);
                }
            }
            auto compact_moves = kv_mgr.compact();
            for (const auto& m : compact_moves) moveKVBuffer(m.src, m.dst, m.len);
        }
    }

    // Drop-in LLMModel::generate override — wraps a 1-session batch so
    // CBLLMModel can be used anywhere an LLMModel is expected.
    std::vector<int32_t> generate(const std::vector<int32_t>& prompt_tokens, const GenerationConfig& gen_cfg = {},
        std::function<bool(int32_t)> token_callback = nullptr) override {
        Scheduler      sched;
        KVCacheManager kv_mgr;
        sched.addSession("s0", prompt_tokens, gen_cfg.max_tokens);
        generateBatch(sched, kv_mgr, [&](const std::string&, int32_t t) {
            if (token_callback) token_callback(t);
        });
        auto* s = sched.getSession("s0");
        return s ? s->generated_tokens : std::vector<int32_t>{};
    }

    void resetKVCache() override { LLMModel::resetKVCache(); }

   protected:
    // CB drives its own CBInputProvider chain; suppress the base LLM providers.
    void createInputProviders() override {}

    bool onInitialized() override {
        if (!LLMModel::onInitialized()) return false;
        for (auto& p : cb_providers_) p->onInitialized(model_cfg_);
        return true;
    }

   private:
    // Same promotion policy as single-session prefill: pick the smallest CL
    // that fits required_kv, then reshape every shard's KV tensors in place.
    void selectCL(int required_kv, int seq_len, int n_valid) {
        if (num_cl_ <= 1) return;

        size_t new_cl = active_cl_idx_;
        while (new_cl + 1 < num_cl_ && static_cast<int>(spec_.context_lengths[new_cl]) - seq_len < required_kv)
            ++new_cl;

        if (new_cl > active_cl_idx_) {
            GENIEX_LOG_INFO("CB: upgrading CL {} -> {} (required_kv={})", active_cl_idx_, new_cl, required_kv);
            const size_t old_kv = spec_.context_lengths[active_cl_idx_] - spec_.seq_len_prefill;
            const size_t new_kv = spec_.context_lengths[new_cl] - spec_.seq_len_prefill;
            for (size_t s = 0; s < shard_count_; ++s) reshapeKV(s, old_kv, new_kv, static_cast<size_t>(n_valid));
            active_cl_idx_ = new_cl;
        }
    }

    // Memmove-safe relocation of KV data inside the shared buffer, per
    // layer and both K/V. Tensor layouts:
    //   Key   [num_kv_heads, 1, head_dim, kv_len]
    //   Value [num_kv_heads, 1, kv_len,   head_dim]
    void moveKVBuffer(int src_start, int dst_start, int length) {
        if (src_start == dst_start || length == 0) return;

        const auto& kv_block = requireKVStateBlock();
        for (size_t s = 0; s < shard_count_; ++s) {
            const auto& pairs = kv_block.shard_pairs[s];
            if (pairs.empty()) continue;
            Graph& g = graph(graphIndex(0, s, active_cl_idx_));

            for (const auto& pair : pairs) {
                {  // Key
                    const auto&  name   = pair.key_in;
                    auto*        buf    = static_cast<uint8_t*>(g.inputPtr(name));
                    const auto&  sp     = g.inputSpec(name);
                    const size_t es     = sp.elementSize();
                    const size_t n_rows = spec_.num_kv_heads * spec_.head_dim;
                    const size_t stride = sp.shape[3];
                    for (size_t r = 0; r < n_rows; ++r)
                        std::memmove(buf + (r * stride + dst_start) * es,
                            buf + (r * stride + src_start) * es,
                            static_cast<size_t>(length) * es);
                }
                {  // Value
                    const auto&  name    = pair.value_in;
                    auto*        buf     = static_cast<uint8_t*>(g.inputPtr(name));
                    const auto&  sp      = g.inputSpec(name);
                    const size_t es      = sp.elementSize();
                    const size_t n_heads = spec_.num_kv_heads;
                    const size_t stride  = sp.shape[2];
                    const size_t tok_sz  = spec_.head_dim * es;
                    for (size_t h = 0; h < n_heads; ++h)
                        std::memmove(buf + (h * stride + dst_start) * tok_sz,
                            buf + (h * stride + src_start) * tok_sz,
                            static_cast<size_t>(length) * tok_sz);
                }
            }
        }
    }

    // Append this step's new KV tokens to each session's segment in the
    // shared input buffer, reading from the shard's K/V output tensors.
    void copyNewKV(size_t shard, const std::vector<Session*>& batch, const std::vector<KVCacheSegment>& batch_kv_segs,
        const std::vector<std::pair<int, int>>& in_segs, const std::vector<int>& /*seg_lengths*/) {
        const auto& kv_block = requireKVStateBlock();
        const auto& pairs    = kv_block.shard_pairs[shard];
        if (pairs.empty()) return;
        Graph& g = graph(graphIndex(0, shard, active_cl_idx_));

        for (size_t si = 0; si < batch.size(); ++si) {
            const auto [in_s, in_l] = in_segs[si];
            const size_t dst        = static_cast<size_t>(batch_kv_segs[si].start_pos + batch_kv_segs[si].length);

            for (const auto& pair : pairs) {
                copyKV(g,
                    pair.key_out,
                    true,
                    g,
                    pair.key_in,
                    static_cast<size_t>(in_s),
                    dst,
                    static_cast<size_t>(in_l),
                    true);
                copyKV(g,
                    pair.value_out,
                    true,
                    g,
                    pair.value_in,
                    static_cast<size_t>(in_s),
                    dst,
                    static_cast<size_t>(in_l),
                    false);
            }
        }
    }

    std::vector<std::unique_ptr<CBInputProvider>> cb_providers_;
};

}  // namespace cb
}  // namespace geniex
