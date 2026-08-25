// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause
//
// Builds a coherent set of GraphInfo_t / Graph fixtures plus a matching LLMSpec
// for LLMModel orchestration tests, with no QNN runtime. Each graph is backed
// by a ClientBuffer IOTensor and the link-time QnnApi stub. The builder owns
// all backing storage (GraphInfoBuilders, IOTensor, QnnApi, Graphs) so it must
// outlive the model that consumes it.

#pragma once

#include <cstdint>
#include <deque>
#include <memory>
#include <string>
#include <vector>

#include "IOTensor.hpp"
#include "QnnApi.hpp"
#include "graph.h"
#include "llm/llm_types.h"
#include "testing/graph_info_builder.hpp"

namespace geniex::testing {

// Small, deterministic single-shard / single-CL / single-KV-layer LLM. The
// graph names encode prefill ar=4 and decode ar=1 at cl=16, which is what
// LLMModel::onInitialized parses to derive seq_len_prefill / seq_len_decode /
// context_lengths.
struct LLMFixture {
    static constexpr uint32_t kVocab      = 8;
    static constexpr uint32_t kHidden     = 4;
    static constexpr uint32_t kKVHeads    = 1;
    static constexpr uint32_t kHeadDim    = 2;
    static constexpr uint32_t kContextLen = 16;
    static constexpr uint32_t kArPrefill  = 4;
    static constexpr uint32_t kArDecode   = 1;
    static constexpr uint32_t kKVLayers   = 1;

    QnnApi   api;
    IOTensor io{BufferAlloc::DEFAULT};

    // Stable storage: builders own tensor buffers; graphs hold non-owning
    // pointers into the builders and into io.
    std::deque<GraphInfoBuilder> builders;
    std::vector<Graph>           graphs;

    LLMFixture() {
        // KV input buffers must hold the largest stride any phase reshapes them
        // to. reshapeKV grows the (prefill) graph's KV buffer to the decode
        // stride (kContextLen - kArDecode), so size every KV input to that max;
        // a per-phase size would let reshapeKV overflow the prefill buffer.
        const uint32_t kv_capacity = kContextLen - kArDecode;
        addGraph("prefill_ar4_cl16_1_of_1", kArPrefill, kv_capacity);
        addGraph("token_ar1_cl16_1_of_1", kArDecode, kv_capacity);
    }

    LLMFixture(const LLMFixture&)            = delete;
    LLMFixture& operator=(const LLMFixture&) = delete;

    // Skeleton spec; shapes, shard wiring, and KV pairs are inferred from the
    // fixture graphs by LLMModel::onInitialized.
    static LLMSpec makeSpec() {
        LLMSpec spec;
        spec.state_blocks.push_back(makeKVStateBlock());
        return spec;
    }

   private:
    void addGraph(const std::string& name, uint32_t ar, uint32_t kv_capacity) {
        std::vector<TensorDesc> inputs{
            {"input_embeds", QNN_DATATYPE_FLOAT_32, {ar, kHidden}},
            {"attention_mask", QNN_DATATYPE_FLOAT_32, {ar, kContextLen}},
        };
        std::vector<TensorDesc> outputs{
            {"logits", QNN_DATATYPE_FLOAT_32, {ar, kVocab}},
        };
        for (uint32_t l = 0; l < kKVLayers; ++l) {
            const std::string suffix = std::to_string(l);
            inputs.push_back(
                {"past_key_" + suffix + "_in", QNN_DATATYPE_FLOAT_32, {kKVHeads, 1, kHeadDim, kv_capacity}});
            inputs.push_back(
                {"past_value_" + suffix + "_in", QNN_DATATYPE_FLOAT_32, {kKVHeads, 1, kv_capacity, kHeadDim}});
            outputs.push_back({"past_key_" + suffix + "_out", QNN_DATATYPE_FLOAT_32, {kKVHeads, 1, kHeadDim, ar}});
            outputs.push_back({"past_value_" + suffix + "_out", QNN_DATATYPE_FLOAT_32, {kKVHeads, 1, ar, kHeadDim}});
        }
        builders.emplace_back(name, inputs, outputs);
        Graph g(&builders.back().graphInfo(), &api, &io);
        g.setup(/*context=*/nullptr);
        graphs.push_back(std::move(g));
    }
};

// Single-shard, TWO-context-length variant ([cl8, cl16]) used to exercise the
// promoteCL / reshapeKV paths that a single-CL fixture never reaches. Same
// dims as LLMFixture; the mask is sized to the MAX CL (per-chunk mask write is
// ar * context_lengths[active]) and KV inputs to max_cl - ar_decode.
struct MultiCLFixture {
    static constexpr uint32_t kVocab     = 8;
    static constexpr uint32_t kHidden    = 4;
    static constexpr uint32_t kKVHeads   = 1;
    static constexpr uint32_t kHeadDim   = 2;
    static constexpr uint32_t kCL0       = 8;
    static constexpr uint32_t kCL1       = 16;  // max CL
    static constexpr uint32_t kArPrefill = 4;
    static constexpr uint32_t kArDecode  = 1;
    static constexpr uint32_t kKVLayers  = 1;

    QnnApi   api;
    IOTensor io{BufferAlloc::DEFAULT};

    std::deque<GraphInfoBuilder> builders;
    std::vector<Graph>           graphs;

    MultiCLFixture() {
        // Graph order must match graphIndex = phase*(1*num_cl) + cl_idx, i.e.
        // prefill[cl0], prefill[cl1], decode[cl0], decode[cl1].
        const uint32_t kv_capacity = kCL1 - kArDecode;
        addGraph("prefill_ar4_cl8_1_of_1", kArPrefill, kv_capacity);
        addGraph("prefill_ar4_cl16_1_of_1", kArPrefill, kv_capacity);
        addGraph("token_ar1_cl8_1_of_1", kArDecode, kv_capacity);
        addGraph("token_ar1_cl16_1_of_1", kArDecode, kv_capacity);
    }

    MultiCLFixture(const MultiCLFixture&)            = delete;
    MultiCLFixture& operator=(const MultiCLFixture&) = delete;

    static LLMSpec makeSpec() {
        LLMSpec spec;
        spec.state_blocks.push_back(makeKVStateBlock());
        return spec;
    }

   private:
    void addGraph(const std::string& name, uint32_t ar, uint32_t kv_capacity) {
        std::vector<TensorDesc> inputs{
            {"input_embeds", QNN_DATATYPE_FLOAT_32, {ar, kHidden}},
            {"attention_mask", QNN_DATATYPE_FLOAT_32, {ar, kCL1}},  // sized to max CL
        };
        std::vector<TensorDesc> outputs{
            {"logits", QNN_DATATYPE_FLOAT_32, {ar, kVocab}},
        };
        for (uint32_t l = 0; l < kKVLayers; ++l) {
            const std::string s = std::to_string(l);
            inputs.push_back({"past_key_" + s + "_in", QNN_DATATYPE_FLOAT_32, {kKVHeads, 1, kHeadDim, kv_capacity}});
            inputs.push_back({"past_value_" + s + "_in", QNN_DATATYPE_FLOAT_32, {kKVHeads, 1, kv_capacity, kHeadDim}});
            outputs.push_back({"past_key_" + s + "_out", QNN_DATATYPE_FLOAT_32, {kKVHeads, 1, kHeadDim, ar}});
            outputs.push_back({"past_value_" + s + "_out", QNN_DATATYPE_FLOAT_32, {kKVHeads, 1, ar, kHeadDim}});
        }
        builders.emplace_back(name, inputs, outputs);
        Graph g(&builders.back().graphInfo(), &api, &io);
        g.setup(/*context=*/nullptr);
        graphs.push_back(std::move(g));
    }
};

// TWO-shard, single-CL variant exercising the multi-shard paths a 1-shard
// fixture never reaches: discoverShardTensorNames (lm_head_only), the inter-
// shard hidden-state connections (buildConnections / applyConnections), and the
// LM-head-skip on non-final prefill chunks.
//
// Shard 0 owns the KV layer and outputs `hidden_states`; shard 1 is LM-head-
// only (no KV, consumes `hidden_states`, outputs `logits`).
struct MultiShardFixture {
    static constexpr uint32_t kVocab      = 8;
    static constexpr uint32_t kHidden     = 4;
    static constexpr uint32_t kKVHeads    = 1;
    static constexpr uint32_t kHeadDim    = 2;
    static constexpr uint32_t kContextLen = 16;
    static constexpr uint32_t kArPrefill  = 4;
    static constexpr uint32_t kArDecode   = 1;
    static constexpr uint32_t kKVLayers   = 1;
    static constexpr uint32_t kShards     = 2;

    QnnApi   api;
    IOTensor io{BufferAlloc::DEFAULT};

    std::deque<GraphInfoBuilder> builders;
    std::vector<Graph>           graphs;

    MultiShardFixture() {
        // graphIndex = phase*(shard_count*num_cl) + shard*num_cl + cl_idx, with
        // num_cl=1: prefill[s0], prefill[s1], decode[s0], decode[s1].
        const uint32_t kv_capacity = kContextLen - kArDecode;
        addShard0("prefill_ar4_cl16_1_of_2", kArPrefill, kv_capacity);
        addShard1("prefill_ar4_cl16_2_of_2", kArPrefill);
        addShard0("token_ar1_cl16_1_of_2", kArDecode, kv_capacity);
        addShard1("token_ar1_cl16_2_of_2", kArDecode);
    }

    MultiShardFixture(const MultiShardFixture&)            = delete;
    MultiShardFixture& operator=(const MultiShardFixture&) = delete;

    static LLMSpec makeSpec() {
        LLMSpec spec;
        spec.state_blocks.push_back(makeKVStateBlock());
        return spec;
    }

   private:
    // Shard 0: input_embeds -> hidden_states, owns KV. Output `hidden_states`
    // is the first non-special output, so discoverShardTensorNames picks it.
    void addShard0(const std::string& name, uint32_t ar, uint32_t kv_capacity) {
        std::vector<TensorDesc> inputs{
            {"input_embeds", QNN_DATATYPE_FLOAT_32, {ar, kHidden}},
            {"attention_mask", QNN_DATATYPE_FLOAT_32, {ar, kContextLen}},
        };
        std::vector<TensorDesc> outputs{
            {"hidden_states", QNN_DATATYPE_FLOAT_32, {ar, kHidden}},
        };
        for (uint32_t l = 0; l < kKVLayers; ++l) {
            const std::string s = std::to_string(l);
            inputs.push_back({"past_key_" + s + "_in", QNN_DATATYPE_FLOAT_32, {kKVHeads, 1, kHeadDim, kv_capacity}});
            inputs.push_back({"past_value_" + s + "_in", QNN_DATATYPE_FLOAT_32, {kKVHeads, 1, kv_capacity, kHeadDim}});
            outputs.push_back({"past_key_" + s + "_out", QNN_DATATYPE_FLOAT_32, {kKVHeads, 1, kHeadDim, ar}});
            outputs.push_back({"past_value_" + s + "_out", QNN_DATATYPE_FLOAT_32, {kKVHeads, 1, ar, kHeadDim}});
        }
        emplace(name, inputs, outputs);
    }

    // Shard 1: hidden_states -> logits, no KV (lm_head_only).
    void addShard1(const std::string& name, uint32_t ar) {
        std::vector<TensorDesc> inputs{
            {"hidden_states", QNN_DATATYPE_FLOAT_32, {ar, kHidden}},
        };
        std::vector<TensorDesc> outputs{
            {"logits", QNN_DATATYPE_FLOAT_32, {ar, kVocab}},
        };
        emplace(name, inputs, outputs);
    }

    void emplace(
        const std::string& name, const std::vector<TensorDesc>& inputs, const std::vector<TensorDesc>& outputs) {
        builders.emplace_back(name, inputs, outputs);
        Graph g(&builders.back().graphInfo(), &api, &io);
        g.setup(/*context=*/nullptr);
        graphs.push_back(std::move(g));
    }
};

// TWO-shard, single-CL variant whose shards are labelled `2_of_3` / `3_of_3`
// with NO `1_of_3` graph — mirroring an EAGLE/eaglet bundle whose leading shard
// (a CPU-side quantized embedding LUT) runs off-graph. Exercises onInitialized's
// non-contiguous shard handling: shard_count must be the number of LOADED shards
// (2), not the `_of_T` total (3), and the shard numbers 2,3 must rank to dense
// slots 0,1. A stale `max(_of_T)` would size the KV/graph tables to 3 and index
// past the 4 loaded graphs. Body shard exposes `last_hidden_states` (the EAGLE
// feature); the lm-head shard is KV-less.
struct ExternalLeadingShardFixture {
    static constexpr uint32_t kVocab      = 8;
    static constexpr uint32_t kHidden     = 4;
    static constexpr uint32_t kKVHeads    = 1;
    static constexpr uint32_t kHeadDim    = 2;
    static constexpr uint32_t kContextLen = 16;
    static constexpr uint32_t kArPrefill  = 4;
    static constexpr uint32_t kArDecode   = 1;
    static constexpr uint32_t kKVLayers   = 1;

    QnnApi   api;
    IOTensor io{BufferAlloc::DEFAULT};

    std::deque<GraphInfoBuilder> builders;
    std::vector<Graph>           graphs;

    ExternalLeadingShardFixture() {
        const uint32_t kv_capacity = kContextLen - kArDecode;
        addBody("prefill_ar4_cl16_2_of_3", kArPrefill, kv_capacity);
        addLMHead("prefill_ar4_cl16_3_of_3", kArPrefill);
        addBody("token_ar1_cl16_2_of_3", kArDecode, kv_capacity);
        addLMHead("token_ar1_cl16_3_of_3", kArDecode);
    }

    ExternalLeadingShardFixture(const ExternalLeadingShardFixture&)            = delete;
    ExternalLeadingShardFixture& operator=(const ExternalLeadingShardFixture&) = delete;

    static LLMSpec makeSpec() {
        LLMSpec spec;
        spec.state_blocks.push_back(makeKVStateBlock());
        return spec;
    }

   private:
    void addBody(const std::string& name, uint32_t ar, uint32_t kv_capacity) {
        std::vector<TensorDesc> inputs{
            {"input_embeds", QNN_DATATYPE_FLOAT_32, {ar, kHidden}},
            {"attention_mask", QNN_DATATYPE_FLOAT_32, {ar, kContextLen}},
        };
        std::vector<TensorDesc> outputs{
            {"last_hidden_states", QNN_DATATYPE_FLOAT_32, {ar, kHidden}},
        };
        for (uint32_t l = 0; l < kKVLayers; ++l) {
            const std::string s = std::to_string(l);
            inputs.push_back({"past_key_" + s + "_in", QNN_DATATYPE_FLOAT_32, {kKVHeads, 1, kHeadDim, kv_capacity}});
            inputs.push_back({"past_value_" + s + "_in", QNN_DATATYPE_FLOAT_32, {kKVHeads, 1, kv_capacity, kHeadDim}});
            outputs.push_back({"past_key_" + s + "_out", QNN_DATATYPE_FLOAT_32, {kKVHeads, 1, kHeadDim, ar}});
            outputs.push_back({"past_value_" + s + "_out", QNN_DATATYPE_FLOAT_32, {kKVHeads, 1, ar, kHeadDim}});
        }
        emplace(name, inputs, outputs);
    }

    void addLMHead(const std::string& name, uint32_t ar) {
        std::vector<TensorDesc> inputs{
            {"last_hidden_states", QNN_DATATYPE_FLOAT_32, {ar, kHidden}},
        };
        std::vector<TensorDesc> outputs{
            {"logits", QNN_DATATYPE_FLOAT_32, {ar, kVocab}},
        };
        emplace(name, inputs, outputs);
    }

    void emplace(
        const std::string& name, const std::vector<TensorDesc>& inputs, const std::vector<TensorDesc>& outputs) {
        builders.emplace_back(name, inputs, outputs);
        Graph g(&builders.back().graphInfo(), &api, &io);
        g.setup(/*context=*/nullptr);
        graphs.push_back(std::move(g));
    }
};

// Target engine for EagleModel: a two-shard model whose DECODE graph is batched
// (ar = draft_len + 1) so the whole speculative chain is verified in one pass.
// Shard 0 (body, owns KV) emits `last_hidden_states` (the EAGLE feature the
// draft is seeded from); shard 1 is the KV-less LM head. kArDecode must equal
// the test's verify width (draft_len + 1); it is kept distinct from kArPrefill
// so onInitialized's max-ar/min-ar phase split still separates prefill/decode.
struct EagleTargetFixture {
    static constexpr uint32_t kVocab      = 8;
    static constexpr uint32_t kHidden     = 4;
    static constexpr uint32_t kKVHeads    = 1;
    static constexpr uint32_t kHeadDim    = 2;
    static constexpr uint32_t kContextLen = 16;
    static constexpr uint32_t kArPrefill  = 8;
    static constexpr uint32_t kArDecode   = 4;  // batched verify width (draft_len + 1)
    static constexpr uint32_t kKVLayers   = 1;

    QnnApi   api;
    IOTensor io{BufferAlloc::DEFAULT};

    std::deque<GraphInfoBuilder> builders;
    std::vector<Graph>           graphs;

    EagleTargetFixture() {
        const uint32_t kv_capacity = kContextLen - kArDecode;
        addBody("prefill_ar8_cl16_1_of_2", kArPrefill, kv_capacity);
        addLMHead("prefill_ar8_cl16_2_of_2", kArPrefill);
        addBody("token_ar4_cl16_1_of_2", kArDecode, kv_capacity);
        addLMHead("token_ar4_cl16_2_of_2", kArDecode);
    }

    EagleTargetFixture(const EagleTargetFixture&)            = delete;
    EagleTargetFixture& operator=(const EagleTargetFixture&) = delete;

    static LLMSpec makeSpec() {
        LLMSpec spec;
        spec.state_blocks.push_back(makeKVStateBlock());
        return spec;
    }

   private:
    void addBody(const std::string& name, uint32_t ar, uint32_t kv_capacity) {
        std::vector<TensorDesc> inputs{
            {"input_embeds", QNN_DATATYPE_FLOAT_32, {ar, kHidden}},
            {"attention_mask", QNN_DATATYPE_FLOAT_32, {ar, kContextLen}},
        };
        std::vector<TensorDesc> outputs{
            {"last_hidden_states", QNN_DATATYPE_FLOAT_32, {ar, kHidden}},
        };
        for (uint32_t l = 0; l < kKVLayers; ++l) {
            const std::string s = std::to_string(l);
            inputs.push_back({"past_key_" + s + "_in", QNN_DATATYPE_FLOAT_32, {kKVHeads, 1, kHeadDim, kv_capacity}});
            inputs.push_back({"past_value_" + s + "_in", QNN_DATATYPE_FLOAT_32, {kKVHeads, 1, kv_capacity, kHeadDim}});
            outputs.push_back({"past_key_" + s + "_out", QNN_DATATYPE_FLOAT_32, {kKVHeads, 1, kHeadDim, ar}});
            outputs.push_back({"past_value_" + s + "_out", QNN_DATATYPE_FLOAT_32, {kKVHeads, 1, ar, kHeadDim}});
        }
        emplace(name, inputs, outputs);
    }

    void addLMHead(const std::string& name, uint32_t ar) {
        std::vector<TensorDesc> inputs{
            {"last_hidden_states", QNN_DATATYPE_FLOAT_32, {ar, kHidden}},
        };
        std::vector<TensorDesc> outputs{
            {"logits", QNN_DATATYPE_FLOAT_32, {ar, kVocab}},
        };
        emplace(name, inputs, outputs);
    }

    void emplace(
        const std::string& name, const std::vector<TensorDesc>& inputs, const std::vector<TensorDesc>& outputs) {
        builders.emplace_back(name, inputs, outputs);
        Graph g(&builders.back().graphInfo(), &api, &io);
        g.setup(/*context=*/nullptr);
        graphs.push_back(std::move(g));
    }
};

// Draft engine for EagleModel: two-shard, single-token (ar=1) decode. The body
// takes an extra `hidden_states` input — the feature the driver overrides each
// step with the target's last hidden state — and emits its own
// `last_hidden_states` for the next draft step.
struct EagleDraftFixture {
    static constexpr uint32_t kVocab      = 8;
    static constexpr uint32_t kHidden     = 4;
    static constexpr uint32_t kKVHeads    = 1;
    static constexpr uint32_t kHeadDim    = 2;
    static constexpr uint32_t kContextLen = 16;
    static constexpr uint32_t kArPrefill  = 4;
    static constexpr uint32_t kArDecode   = 1;
    static constexpr uint32_t kKVLayers   = 1;

    QnnApi   api;
    IOTensor io{BufferAlloc::DEFAULT};

    std::deque<GraphInfoBuilder> builders;
    std::vector<Graph>           graphs;

    EagleDraftFixture() {
        const uint32_t kv_capacity = kContextLen - kArDecode;
        addBody("prefill_ar4_cl16_1_of_2", kArPrefill, kv_capacity);
        addLMHead("prefill_ar4_cl16_2_of_2", kArPrefill);
        addBody("token_ar1_cl16_1_of_2", kArDecode, kv_capacity);
        addLMHead("token_ar1_cl16_2_of_2", kArDecode);
    }

    EagleDraftFixture(const EagleDraftFixture&)            = delete;
    EagleDraftFixture& operator=(const EagleDraftFixture&) = delete;

    static LLMSpec makeSpec() {
        LLMSpec spec;
        spec.state_blocks.push_back(makeKVStateBlock());
        return spec;
    }

   private:
    void addBody(const std::string& name, uint32_t ar, uint32_t kv_capacity) {
        std::vector<TensorDesc> inputs{
            {"input_embeds", QNN_DATATYPE_FLOAT_32, {ar, kHidden}},
            {"attention_mask", QNN_DATATYPE_FLOAT_32, {ar, kContextLen}},
            {"hidden_states", QNN_DATATYPE_FLOAT_32, {ar, kHidden}},
        };
        std::vector<TensorDesc> outputs{
            {"last_hidden_states", QNN_DATATYPE_FLOAT_32, {ar, kHidden}},
        };
        for (uint32_t l = 0; l < kKVLayers; ++l) {
            const std::string s = std::to_string(l);
            inputs.push_back({"past_key_" + s + "_in", QNN_DATATYPE_FLOAT_32, {kKVHeads, 1, kHeadDim, kv_capacity}});
            inputs.push_back({"past_value_" + s + "_in", QNN_DATATYPE_FLOAT_32, {kKVHeads, 1, kv_capacity, kHeadDim}});
            outputs.push_back({"past_key_" + s + "_out", QNN_DATATYPE_FLOAT_32, {kKVHeads, 1, kHeadDim, ar}});
            outputs.push_back({"past_value_" + s + "_out", QNN_DATATYPE_FLOAT_32, {kKVHeads, 1, ar, kHeadDim}});
        }
        emplace(name, inputs, outputs);
    }

    void addLMHead(const std::string& name, uint32_t ar) {
        std::vector<TensorDesc> inputs{
            {"last_hidden_states", QNN_DATATYPE_FLOAT_32, {ar, kHidden}},
        };
        std::vector<TensorDesc> outputs{
            {"logits", QNN_DATATYPE_FLOAT_32, {ar, kVocab}},
        };
        emplace(name, inputs, outputs);
    }

    void emplace(
        const std::string& name, const std::vector<TensorDesc>& inputs, const std::vector<TensorDesc>& outputs) {
        builders.emplace_back(name, inputs, outputs);
        Graph g(&builders.back().graphInfo(), &api, &io);
        g.setup(/*context=*/nullptr);
        graphs.push_back(std::move(g));
    }
};

// Draft engine with a batched (ar=2) decode graph so buildDraftTree's frontier
// is wider than one row. Only this width lets n_branches > 1 survive the
// draft_w clamp, reaching the level-synchronous expansion, global candidate
// pool, per-level KV commit, kv_ancestors threading and cumulative-probability
// prune that the ar=1 EagleDraftFixture collapses away. Same bindings as
// EagleDraftFixture; only kArDecode differs.
struct EagleWideDraftFixture {
    static constexpr uint32_t kVocab      = 8;
    static constexpr uint32_t kHidden     = 4;
    static constexpr uint32_t kKVHeads    = 1;
    static constexpr uint32_t kHeadDim    = 2;
    static constexpr uint32_t kContextLen = 16;
    static constexpr uint32_t kArPrefill  = 4;
    static constexpr uint32_t kArDecode   = 2;  // batched draft frontier width
    static constexpr uint32_t kKVLayers   = 1;

    QnnApi   api;
    IOTensor io{BufferAlloc::DEFAULT};

    std::deque<GraphInfoBuilder> builders;
    std::vector<Graph>           graphs;

    EagleWideDraftFixture() {
        const uint32_t kv_capacity = kContextLen - kArDecode;
        addBody("prefill_ar4_cl16_1_of_2", kArPrefill, kv_capacity);
        addLMHead("prefill_ar4_cl16_2_of_2", kArPrefill);
        addBody("token_ar2_cl16_1_of_2", kArDecode, kv_capacity);
        addLMHead("token_ar2_cl16_2_of_2", kArDecode);
    }

    EagleWideDraftFixture(const EagleWideDraftFixture&)            = delete;
    EagleWideDraftFixture& operator=(const EagleWideDraftFixture&) = delete;

    static LLMSpec makeSpec() {
        LLMSpec spec;
        spec.state_blocks.push_back(makeKVStateBlock());
        return spec;
    }

   private:
    void addBody(const std::string& name, uint32_t ar, uint32_t kv_capacity) {
        std::vector<TensorDesc> inputs{
            {"input_embeds", QNN_DATATYPE_FLOAT_32, {ar, kHidden}},
            {"attention_mask", QNN_DATATYPE_FLOAT_32, {ar, kContextLen}},
            {"hidden_states", QNN_DATATYPE_FLOAT_32, {ar, kHidden}},
        };
        std::vector<TensorDesc> outputs{
            {"last_hidden_states", QNN_DATATYPE_FLOAT_32, {ar, kHidden}},
        };
        for (uint32_t l = 0; l < kKVLayers; ++l) {
            const std::string s = std::to_string(l);
            inputs.push_back({"past_key_" + s + "_in", QNN_DATATYPE_FLOAT_32, {kKVHeads, 1, kHeadDim, kv_capacity}});
            inputs.push_back({"past_value_" + s + "_in", QNN_DATATYPE_FLOAT_32, {kKVHeads, 1, kv_capacity, kHeadDim}});
            outputs.push_back({"past_key_" + s + "_out", QNN_DATATYPE_FLOAT_32, {kKVHeads, 1, kHeadDim, ar}});
            outputs.push_back({"past_value_" + s + "_out", QNN_DATATYPE_FLOAT_32, {kKVHeads, 1, ar, kHeadDim}});
        }
        emplace(name, inputs, outputs);
    }

    void addLMHead(const std::string& name, uint32_t ar) {
        std::vector<TensorDesc> inputs{
            {"last_hidden_states", QNN_DATATYPE_FLOAT_32, {ar, kHidden}},
        };
        std::vector<TensorDesc> outputs{
            {"logits", QNN_DATATYPE_FLOAT_32, {ar, kVocab}},
        };
        emplace(name, inputs, outputs);
    }

    void emplace(
        const std::string& name, const std::vector<TensorDesc>& inputs, const std::vector<TensorDesc>& outputs) {
        builders.emplace_back(name, inputs, outputs);
        Graph g(&builders.back().graphInfo(), &api, &io);
        g.setup(/*context=*/nullptr);
        graphs.push_back(std::move(g));
    }
};

// Two-CL ([cl8, cl16]) EAGLE target: the multi-CL analog of EagleTargetFixture,
// used to drive CL promotion through the speculative verify loop. Two shards
// (body + LM head), batched decode (ar = draft_len + 1). KV inputs are strided
// to (max_cl - ar_decode) and the mask to the max CL, matching how a real
// weight-shared multi-CL export lays out its buffers.
struct MultiCLEagleTargetFixture {
    static constexpr uint32_t kVocab     = 8;
    static constexpr uint32_t kHidden    = 4;
    static constexpr uint32_t kKVHeads   = 1;
    static constexpr uint32_t kHeadDim   = 2;
    static constexpr uint32_t kCL0       = 8;
    static constexpr uint32_t kCL1       = 16;  // max CL
    static constexpr uint32_t kArPrefill = 8;
    static constexpr uint32_t kArDecode  = 4;  // batched verify width (draft_len + 1)
    static constexpr uint32_t kKVLayers  = 1;

    QnnApi   api;
    IOTensor io{BufferAlloc::DEFAULT};

    std::deque<GraphInfoBuilder> builders;
    std::vector<Graph>           graphs;

    MultiCLEagleTargetFixture() {
        // graphIndex = phase*(shard_count*num_cl) + shard*num_cl + cl_idx, with
        // shard_count=2, num_cl=2: prefill[s0/cl0, s0/cl1, s1/cl0, s1/cl1],
        // then the same layout for decode.
        const uint32_t kv_capacity = kCL1 - kArDecode;
        addBody("prefill_ar8_cl8_1_of_2", kArPrefill, kv_capacity);
        addBody("prefill_ar8_cl16_1_of_2", kArPrefill, kv_capacity);
        addLMHead("prefill_ar8_cl8_2_of_2", kArPrefill);
        addLMHead("prefill_ar8_cl16_2_of_2", kArPrefill);
        addBody("token_ar4_cl8_1_of_2", kArDecode, kv_capacity);
        addBody("token_ar4_cl16_1_of_2", kArDecode, kv_capacity);
        addLMHead("token_ar4_cl8_2_of_2", kArDecode);
        addLMHead("token_ar4_cl16_2_of_2", kArDecode);
    }

    MultiCLEagleTargetFixture(const MultiCLEagleTargetFixture&)            = delete;
    MultiCLEagleTargetFixture& operator=(const MultiCLEagleTargetFixture&) = delete;

    static LLMSpec makeSpec() {
        LLMSpec spec;
        spec.state_blocks.push_back(makeKVStateBlock());
        return spec;
    }

   private:
    void addBody(const std::string& name, uint32_t ar, uint32_t kv_capacity) {
        std::vector<TensorDesc> inputs{
            {"input_embeds", QNN_DATATYPE_FLOAT_32, {ar, kHidden}},
            {"attention_mask", QNN_DATATYPE_FLOAT_32, {ar, kCL1}},  // sized to max CL
        };
        std::vector<TensorDesc> outputs{
            {"last_hidden_states", QNN_DATATYPE_FLOAT_32, {ar, kHidden}},
        };
        for (uint32_t l = 0; l < kKVLayers; ++l) {
            const std::string s = std::to_string(l);
            inputs.push_back({"past_key_" + s + "_in", QNN_DATATYPE_FLOAT_32, {kKVHeads, 1, kHeadDim, kv_capacity}});
            inputs.push_back({"past_value_" + s + "_in", QNN_DATATYPE_FLOAT_32, {kKVHeads, 1, kv_capacity, kHeadDim}});
            outputs.push_back({"past_key_" + s + "_out", QNN_DATATYPE_FLOAT_32, {kKVHeads, 1, kHeadDim, ar}});
            outputs.push_back({"past_value_" + s + "_out", QNN_DATATYPE_FLOAT_32, {kKVHeads, 1, ar, kHeadDim}});
        }
        emplace(name, inputs, outputs);
    }

    void addLMHead(const std::string& name, uint32_t ar) {
        std::vector<TensorDesc> inputs{
            {"last_hidden_states", QNN_DATATYPE_FLOAT_32, {ar, kHidden}},
        };
        std::vector<TensorDesc> outputs{
            {"logits", QNN_DATATYPE_FLOAT_32, {ar, kVocab}},
        };
        emplace(name, inputs, outputs);
    }

    void emplace(
        const std::string& name, const std::vector<TensorDesc>& inputs, const std::vector<TensorDesc>& outputs) {
        builders.emplace_back(name, inputs, outputs);
        Graph g(&builders.back().graphInfo(), &api, &io);
        g.setup(/*context=*/nullptr);
        graphs.push_back(std::move(g));
    }
};

// Two-CL ([cl8, cl16]) EAGLE draft: the multi-CL analog of EagleDraftFixture.
// Single-token decode (ar=1) whose body takes the extra `hidden_states` feature
// input, so the draft's per-round advance can also cross the CL boundary and
// exercise promoteDecodeCL on the draft engine.
struct MultiCLEagleDraftFixture {
    static constexpr uint32_t kVocab     = 8;
    static constexpr uint32_t kHidden    = 4;
    static constexpr uint32_t kKVHeads   = 1;
    static constexpr uint32_t kHeadDim   = 2;
    static constexpr uint32_t kCL0       = 8;
    static constexpr uint32_t kCL1       = 16;  // max CL
    static constexpr uint32_t kArPrefill = 4;
    static constexpr uint32_t kArDecode  = 1;
    static constexpr uint32_t kKVLayers  = 1;

    QnnApi   api;
    IOTensor io{BufferAlloc::DEFAULT};

    std::deque<GraphInfoBuilder> builders;
    std::vector<Graph>           graphs;

    MultiCLEagleDraftFixture() {
        const uint32_t kv_capacity = kCL1 - kArDecode;
        addBody("prefill_ar4_cl8_1_of_2", kArPrefill, kv_capacity);
        addBody("prefill_ar4_cl16_1_of_2", kArPrefill, kv_capacity);
        addLMHead("prefill_ar4_cl8_2_of_2", kArPrefill);
        addLMHead("prefill_ar4_cl16_2_of_2", kArPrefill);
        addBody("token_ar1_cl8_1_of_2", kArDecode, kv_capacity);
        addBody("token_ar1_cl16_1_of_2", kArDecode, kv_capacity);
        addLMHead("token_ar1_cl8_2_of_2", kArDecode);
        addLMHead("token_ar1_cl16_2_of_2", kArDecode);
    }

    MultiCLEagleDraftFixture(const MultiCLEagleDraftFixture&)            = delete;
    MultiCLEagleDraftFixture& operator=(const MultiCLEagleDraftFixture&) = delete;

    static LLMSpec makeSpec() {
        LLMSpec spec;
        spec.state_blocks.push_back(makeKVStateBlock());
        return spec;
    }

   private:
    void addBody(const std::string& name, uint32_t ar, uint32_t kv_capacity) {
        std::vector<TensorDesc> inputs{
            {"input_embeds", QNN_DATATYPE_FLOAT_32, {ar, kHidden}},
            {"attention_mask", QNN_DATATYPE_FLOAT_32, {ar, kCL1}},  // sized to max CL
            {"hidden_states", QNN_DATATYPE_FLOAT_32, {ar, kHidden}},
        };
        std::vector<TensorDesc> outputs{
            {"last_hidden_states", QNN_DATATYPE_FLOAT_32, {ar, kHidden}},
        };
        for (uint32_t l = 0; l < kKVLayers; ++l) {
            const std::string s = std::to_string(l);
            inputs.push_back({"past_key_" + s + "_in", QNN_DATATYPE_FLOAT_32, {kKVHeads, 1, kHeadDim, kv_capacity}});
            inputs.push_back({"past_value_" + s + "_in", QNN_DATATYPE_FLOAT_32, {kKVHeads, 1, kv_capacity, kHeadDim}});
            outputs.push_back({"past_key_" + s + "_out", QNN_DATATYPE_FLOAT_32, {kKVHeads, 1, kHeadDim, ar}});
            outputs.push_back({"past_value_" + s + "_out", QNN_DATATYPE_FLOAT_32, {kKVHeads, 1, ar, kHeadDim}});
        }
        emplace(name, inputs, outputs);
    }

    void addLMHead(const std::string& name, uint32_t ar) {
        std::vector<TensorDesc> inputs{
            {"last_hidden_states", QNN_DATATYPE_FLOAT_32, {ar, kHidden}},
        };
        std::vector<TensorDesc> outputs{
            {"logits", QNN_DATATYPE_FLOAT_32, {ar, kVocab}},
        };
        emplace(name, inputs, outputs);
    }

    void emplace(
        const std::string& name, const std::vector<TensorDesc>& inputs, const std::vector<TensorDesc>& outputs) {
        builders.emplace_back(name, inputs, outputs);
        Graph g(&builders.back().graphInfo(), &api, &io);
        g.setup(/*context=*/nullptr);
        graphs.push_back(std::move(g));
    }
};

}  // namespace geniex::testing
