// Copyright (c) 2026 Nexa AI. SPDX-License-Identifier: BSD-3-Clause
//
// Gemma4 E2B AR=1 decode support for the prefill(ar128)+decode(ar1) driver.
//
// Extends the prefill InputBuilder (gemma4.h) with the AR=1 (curr_len=1,
// n_past>0) variants of the embed / PLE / RoPE / mask builders, plus a KV-cache
// manager that seeds from the prefill graph's KV outputs and appends one column
// (K) / row (V) per decode step. All layouts match the deploy graph:
//   input_embeds        BD1L [1, hidden, 1, 1]   (S=1)   -> channel-major = [c]
//   per_layer_input[l]  BD1L [1, ple,    1, 1]           -> [c]
//   cos/sin (regime)    [1,1,1,half]                      -> [i]   (single pos)
//   attention_mask      [1,1,1, KV+1]  additive           -> [col]
//   past_key_{L}_in     [1,1, hd, KV]   K column-per-pos  -> write col n_past
//   past_value_{L}_in   [1,1, KV, hd]   V row-per-pos     -> write row n_past
// KV lengths at ar1-cl4096: full=4095, sliding=511 (=CL-1).
//
// KV-sharing: only the first (num_layers - num_kv_shared) = 15 layers own a KV
// pair; the graph internally reuses the last non-shared layer's KV for the
// shared tail, so the driver only tracks/plumbs 15 KV pairs.
//
// TODO(gemma4): this whole bespoke driver (KvCache, DecodeInputBuilder,
// writeKVIn/writeKVColumn/quantizeRoundU16/U8, and the hand-rolled prefill+decode
// loop in gemma4_e2b_decode_example.cpp) bypasses the generic core/ runtime and
// should be ported to LLMModel + InputProvider composition (per .claude/rules/
// engineering-principles.md: "leaf models = LLMSpec + InputProvider, not a
// hand-rolled loop"). It was driven directly because gemma4's exported graph
// didn't match LLMModel's stock role-tensor scheme (per-head past_*_{L}_h0_in
// names, dual full/sliding masks, PLE) — but branch zack/gemma4-v73-ce already
// runs gemma4 on the generic LLMModel, so it's achievable, not blocked.
// Porting would DELETE most of this file's KV/requant machinery for free:
// core's LLMModel::copyKV (core/src/llm/llm_model.cpp) is a raw-byte,
// dtype-driven memcpy of already-quantized KV-out->KV-in — it never requantizes
// from float, so it structurally avoids the trunc-vs-round bug that
// quantizeRoundU16/U8 exist to fix AND already handles int8/uint8 KV I/O with
// zero extra code. Remaining model-specific bits (PLE, dual mask, partial RoPE)
// become InputProvider subclasses in models/gemma4. Keep this driver as the
// working/fast example until that port lands.

#pragma once

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "gemma4/gemma4.h"

namespace geniex {
namespace gemma4 {

// ---- AR=1 input builder (single-token decode step) -----------------------
class DecodeInputBuilder {
   public:
    DecodeInputBuilder(const Gemma4Arch& arch, int kv_full, int kv_slide, Bf16Table* embed, Bf16Table* ple)
        : arch_(arch), kv_full_(kv_full), kv_slide_(kv_slide), embed_(embed), ple_(ple) {}

    // input_embeds BD1L [1, hidden, 1, 1] = embed[id] * sqrt(hidden). S=1 so the
    // channel-major layout is simply out[c].
    std::vector<float> buildEmbeds(int32_t id) const {
        std::vector<float> out(arch_.hidden, 0.0f);
        const float        scale = std::sqrt((float)arch_.hidden);
        std::vector<float> row(arch_.hidden);
        embed_->gatherRow(id, row.data());
        for (int c = 0; c < arch_.hidden; ++c) out[c] = row[c] * scale;
        return out;
    }

    // per_layer_input[l] BD1L [1, ple, 1, 1] = ple[id][l] * sqrt(ple_dim).
    std::vector<std::vector<float>> buildPerLayer(int32_t id) const {
        const int   L = arch_.num_layers, D = arch_.ple_dim;
        const float scl = std::sqrt((float)D);
        std::vector<std::vector<float>> outs(L, std::vector<float>(D, 0.0f));
        int                             gid = (id < 0 || id >= ple_->rows()) ? 0 : id;
        std::vector<float>              packed(ple_->cols());  // L*D
        ple_->gatherRow(gid, packed.data());
        for (int l = 0; l < L; ++l)
            for (int c = 0; c < D; ++c) outs[l][c] = packed[(size_t)l * D + c] * scl;
        return outs;
    }

    // cos/sin at a single absolute position n_past. Half-width [half]. full uses
    // proportional rope (0.25*global_head_dim/2 real freqs, rest zero-freq→cos1/sin0).
    void buildRope(bool sliding, int n_past, std::vector<float>& cos_out, std::vector<float>& sin_out) const {
        const int   hd = sliding ? arch_.head_dim : arch_.global_head_dim;
        const int   half = hd / 2;
        const float theta = sliding ? arch_.rope_theta_local : arch_.rope_theta_full;
        int         real = half;
        if (!sliding) real = (int)(arch_.partial_rotary * hd / 2);  // 64
        cos_out.assign(half, 1.0f);
        sin_out.assign(half, 0.0f);
        const float pos = (float)n_past;
        for (int i = 0; i < real; ++i) {
            float inv = 1.0f / std::pow(theta, (float)(2 * i) / (float)hd);
            float ang = pos * inv;
            cos_out[i] = std::cos(ang);
            sin_out[i] = std::sin(ang);
        }
    }

    // Additive mask [1,1,1, KV+1]: for the single query at absolute pos n_past,
    // attend past columns [0, n_past) and the current column at index KV.
    // sliding: additionally require (n_past - j) < sliding_window for past cols.
    std::vector<float> buildMask(bool sliding, int n_past) const {
        const int kv = sliding ? kv_slide_ : kv_full_;
        const int total = kv + 1;
        // Masked-out fill = -1e9 (matches the proven LLMModel reference; -1e4 was
        // NOT negative enough to fully zero out after the HTP int16 softmax, so
        // masked columns leaked in and derailed AR=1 decode).
        std::vector<float> m((size_t)total, -1e9f);
        for (int j = 0; j < n_past && j < kv; ++j) {
            if (!sliding || (n_past - j) < arch_.sliding_window) m[j] = 0.0f;
        }
        m[kv] = 0.0f;  // current token attends itself
        return m;
    }

    int kvFull() const { return kv_full_; }
    int kvSlide() const { return kv_slide_; }

   private:
    Gemma4Arch arch_;
    int        kv_full_, kv_slide_;
    Bf16Table* embed_;
    Bf16Table* ple_;
};

// ---- running KV cache for the 15 non-shared layers -----------------------
// Stores each layer's K as [hd, KV] (column-major over positions: new key at
// column n_past) and V as [KV, hd] (new value at row n_past). head_dim + KV
// length depend on the layer's attention regime (full vs sliding).
class KvCache {
   public:
    KvCache(const Gemma4Arch& arch, int kv_full, int kv_slide)
        : arch_(arch), kv_full_(kv_full), kv_slide_(kv_slide) {
        n_pairs_ = arch.firstShared();  // 15
        k_.resize(n_pairs_);
        v_.resize(n_pairs_);
        for (int L = 0; L < n_pairs_; ++L) {
            int hd = arch_.headDimOf(L);
            int kv = kvOf(L);
            k_[L].assign((size_t)hd * kv, 0.0f);
            v_[L].assign((size_t)kv * hd, 0.0f);
        }
    }

    int  numPairs() const { return n_pairs_; }
    int  kvOf(int L) const { return arch_.isSliding(L) ? kv_slide_ : kv_full_; }
    int  hdOf(int L) const { return arch_.headDimOf(L); }
    const std::vector<float>& key(int L) const { return k_[L]; }
    const std::vector<float>& val(int L) const { return v_[L]; }

    // Seed the cache from prefill KV outputs. Prefill K-out is [hd, S_prefill]
    // (column per prefill position); V-out is [S_prefill, hd]. We copy the first
    // curr_len real positions into cache columns/rows [0, curr_len).
    // prefill_k: pointer to [hd * s_pref], prefill_v: [s_pref * hd].
    void seedLayer(int L, const float* prefill_k, int s_pref, const float* prefill_v, int curr_len) {
        int hd = hdOf(L), kv = kvOf(L);
        for (int d = 0; d < hd; ++d)
            for (int t = 0; t < curr_len; ++t) k_[L][(size_t)d * kv + t] = prefill_k[(size_t)d * s_pref + t];
        for (int t = 0; t < curr_len; ++t)
            for (int d = 0; d < hd; ++d) v_[L][(size_t)t * hd + d] = prefill_v[(size_t)t * hd + d];
    }

    // Append a single decode-step KV at column/row n_past. decode_k is [hd, 1],
    // decode_v is [1, hd].
    void appendLayer(int L, const float* decode_k, const float* decode_v, int n_past) {
        int hd = hdOf(L), kv = kvOf(L);
        if (n_past >= kv) return;  // cache full (shouldn't happen within CL)
        for (int d = 0; d < hd; ++d) k_[L][(size_t)d * kv + n_past] = decode_k[d];
        for (int d = 0; d < hd; ++d) v_[L][(size_t)n_past * hd + d] = decode_v[d];
    }

   private:
    Gemma4Arch         arch_;
    int                kv_full_, kv_slide_, n_pairs_;
    std::vector<std::vector<float>> k_, v_;
};

}  // namespace gemma4
}  // namespace geniex
