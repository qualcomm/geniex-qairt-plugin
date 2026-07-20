// Copyright (c) 2026 Nexa AI. SPDX-License-Identifier: BSD-3-Clause
//
// Gemma4 (E2B text) driver for geniex-qairt-plugin.
//
// Gemma4's QNN graph does NOT match the stock LLMModel role-tensor scheme
// (input_ids / attention_mask / position_ids_cos / past_key_N). It has 72
// numbered inputs (input_0..71) and 31 outputs (output_0..30):
//   input_0                    : inputs_embeds        [1, S, 1536]  (uFixed16)
//   input_1..35                : per_layer_input[i]   [1, S, 256]   (PLE, 35 layers)
//   input_36/37                : cos_global / sin_global   [1,1,S,256]  (full-attn RoPE, half global_head_dim)
//   input_38/39                : cos_local  / sin_local    [1,1,S,128]  (sliding-attn RoPE, half head_dim)
//   input_40                   : attention_mask_full  [1,1,S, S+KVfull]
//   input_41                   : attention_mask_slide [1,1,S, S+KVslide]
//   input_42..71               : kv caches for the 15 non-shared layers (k,v interleaved)
//                                k: [1,1,head_dim,KV]  v: [1,1,KV,head_dim]  (per layer type)
//   output_0                   : logits [1, S, 262144]
//   output_1..30               : new k/v for the 15 non-shared layers
//
// So we drive it directly: a thin Model subclass loads the context binary, and
// this header ports QNNGemma4LLMUtils (embeds+PLE lookup, dual RoPE, dual masks,
// KV) to C++. Streaming/tokenizer/sampler are reused from geniex-proc.
//
// This header holds the shared AR=S input builder used by both drivers:
//   - gemma4_e2b_example.cpp: single AR=S prefill graph, re-run each decode step
//     with the running token sequence (correct, but O(S) per token).
//   - gemma4_e2b_decode_example.cpp: weight-shared prefill(ar128)+decode(ar1)
//     binary with a real AR=1 decode graph (see gemma4_decode.h). ~5x faster.

#pragma once

#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "graph.h"
#include "model.h"
#include "types.h"

namespace geniex {
namespace gemma4 {

// ---- gemma4 E2B architecture constants (from config.json) ----------------
struct Gemma4Arch {
    int hidden          = 1536;
    int num_layers      = 35;
    int num_heads       = 8;
    int num_kv_heads    = 1;
    int head_dim        = 256;   // sliding-attn layers
    int global_head_dim = 512;   // full-attn layers
    int ple_dim         = 256;   // hidden_size_per_layer_input
    int vocab           = 262144;
    int sliding_window  = 512;
    int num_kv_shared   = 20;    // last 20 layers reuse KV
    float rms_eps       = 1e-6f;
    float rope_theta_full  = 1000000.0f;
    float rope_theta_local = 10000.0f;
    float partial_rotary   = 0.25f;   // full-attn proportional rope
    float final_softcap    = 30.0f;

    int firstShared() const { return num_layers - num_kv_shared; }  // 15
    bool isSliding(int layer) const {
        // full_attention at layer idx where (i+1)%6==0 → [5,11,17,23,29,34]? gemma4 E2B
        // layer_types: full at [4,9,14,19,24,29,34]. Encode that directly.
        static const int full[] = {4, 9, 14, 19, 24, 29, 34};
        for (int f : full)
            if (layer == f) return false;
        return true;
    }
    int headDimOf(int layer) const { return isSliding(layer) ? head_dim : global_head_dim; }
};

// ---- bf16 embedding table (mmap-free simple loader) ----------------------
// Raw little-endian bfloat16, row-major [rows, cols].
class Bf16Table {
   public:
    void load(const std::string& path, int rows, int cols) {
        rows_ = rows;
        cols_ = cols;
        std::ifstream f(path, std::ios::binary);
        if (!f) throw std::runtime_error("Bf16Table: cannot open " + path);
        data_.resize(static_cast<size_t>(rows) * cols);
        f.read(reinterpret_cast<char*>(data_.data()), static_cast<std::streamsize>(data_.size() * 2));
        if (!f) throw std::runtime_error("Bf16Table: short read " + path);
    }
    // Gather row `id` into out[0..cols_) as float32.
    void gatherRow(int id, float* out) const {
        const uint16_t* row = data_.data() + static_cast<size_t>(id) * cols_;
        for (int c = 0; c < cols_; ++c) out[c] = bf16_to_f32(row[c]);
    }
    int rows() const { return rows_; }
    int cols() const { return cols_; }

   private:
    static float bf16_to_f32(uint16_t b) {
        uint32_t bits = static_cast<uint32_t>(b) << 16;
        float f;
        std::memcpy(&f, &bits, 4);
        return f;
    }
    std::vector<uint16_t> data_;
    int                   rows_ = 0, cols_ = 0;
};

// ---- CPU-side input builder (ports QNNGemma4LLMUtils) --------------------
// Produces the float payloads for every graph input given the running token
// sequence. All tensors are built at the graph's fixed S (=seq_len_prefill);
// positions >= curr_len are padded and masked out.
class InputBuilder {
   public:
    InputBuilder(const Gemma4Arch& arch, int seq_len, int kv_full, int kv_slide,
                 Bf16Table* embed, Bf16Table* ple)
        : arch_(arch), S_(seq_len), kv_full_(kv_full), kv_slide_(kv_slide), embed_(embed), ple_(ple) {}

    // input_embeds is BD1L: graph tensor shape [1, hidden, 1, S], i.e. the
    // memory layout is CHANNEL-major (index = c*S + t), NOT token-major. The
    // python export does reshape(B,S,1,hidden).transpose(1,3) → [B,hidden,1,S].
    // (Verified byte-exact vs the python reference raws for real positions.)
    //   input_embeds = embed[id] * sqrt(hidden)
    std::vector<float> buildEmbeds(const std::vector<int32_t>& ids) const {
        std::vector<float> out(static_cast<size_t>(S_) * arch_.hidden, 0.0f);
        const float scale = std::sqrt((float)arch_.hidden);
        std::vector<float> row(arch_.hidden);
        for (int t = 0; t < (int)ids.size() && t < S_; ++t) {
            embed_->gatherRow(ids[t], row.data());
            for (int c = 0; c < arch_.hidden; ++c) out[(size_t)c * S_ + t] = row[c] * scale;  // [c, t]
        }
        return out;
    }

    // per_layer_input[layer] is BD1L too: graph tensor [1, ple_dim, 1, S] →
    // channel-major (index = c*S + t). ple packed table is [vocab, L*ple_dim].
    //   per_layer_input[l] = ple[id][l] * sqrt(ple_dim)
    std::vector<std::vector<float>> buildPerLayer(const std::vector<int32_t>& ids) const {
        const int   L   = arch_.num_layers, D = arch_.ple_dim;
        const float scl = std::sqrt((float)D);
        std::vector<std::vector<float>> outs(L, std::vector<float>((size_t)S_ * D, 0.0f));
        std::vector<float> packed(ple_->cols());  // L*D
        for (int t = 0; t < (int)ids.size() && t < S_; ++t) {
            int id = ids[t];
            if (id < 0 || id >= ple_->rows()) id = 0;
            ple_->gatherRow(id, packed.data());
            for (int l = 0; l < L; ++l)
                for (int c = 0; c < D; ++c) outs[l][(size_t)c * S_ + t] = packed[(size_t)l * D + c] * scl;  // [c, t]
        }
        return outs;
    }

    // Half-width cos/sin for one attention regime, [S, half] flattened.
    // full: proportional rope over global_head_dim (0.25 → 64 real + rest zero).
    // local: default rope over head_dim (all half nonzero).
    void buildRope(bool sliding, const std::vector<int32_t>& /*ids*/, int curr_len,
                   std::vector<float>& cos_out, std::vector<float>& sin_out) const {
        const int   hd    = sliding ? arch_.head_dim : arch_.global_head_dim;
        const int   half  = hd / 2;
        const float theta = sliding ? arch_.rope_theta_local : arch_.rope_theta_full;
        int         real  = half;
        if (!sliding) real = (int)(arch_.partial_rotary * hd / 2);  // 64
        std::vector<float> inv_freq(half, 0.0f);
        for (int i = 0; i < real; ++i) inv_freq[i] = 1.0f / std::pow(theta, (float)(2 * i) / (float)hd);
        cos_out.assign((size_t)S_ * half, 1.0f);  // zero-freq → cos=1
        sin_out.assign((size_t)S_ * half, 0.0f);  // zero-freq → sin=0
        for (int p = 0; p < S_; ++p) {
            float pos = (float)p;  // positions 0..S-1 (prefill from n_past=0)
            for (int i = 0; i < half; ++i) {
                float ang                          = pos * inv_freq[i];
                cos_out[(size_t)p * half + i]      = std::cos(ang);
                sin_out[(size_t)p * half + i]      = std::sin(ang);
            }
        }
        (void)curr_len;
    }

    // Additive 4D mask [S, total] flattened; 0 where attend, -1e4 where masked.
    // n_past=0 prefill: causal over the current tokens placed at columns
    // [kv_len .. kv_len+curr_len). Left kv_len columns are all-masked (empty cache).
    std::vector<float> buildMask(bool sliding, int curr_len) const {
        const int kv    = sliding ? kv_slide_ : kv_full_;
        const int total = kv + S_;
        std::vector<float> m((size_t)S_ * total, -1e9f);  // -1e9 (not -1e4): HTP int16 softmax
        for (int i = 0; i < curr_len; ++i) {
            for (int j = 0; j < curr_len; ++j) {
                int col = kv + j;
                if (j <= i && (!sliding || (i - j) < arch_.sliding_window)) m[(size_t)i * total + col] = 0.0f;
            }
        }
        return m;
    }

    int seqLen() const { return S_; }
    int kvFull() const { return kv_full_; }
    int kvSlide() const { return kv_slide_; }

   private:
    Gemma4Arch arch_;
    int        S_, kv_full_, kv_slide_;
    Bf16Table* embed_;
    Bf16Table* ple_;
};

// Thin Model subclass: just loads the context binary and exposes graph(0).
class Gemma4Model : public Model {
   public:
    Gemma4Model() = default;
};

}  // namespace gemma4
}  // namespace geniex
