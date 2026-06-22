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

    // One KV-only state block; one shard owning layer 0.
    static LLMSpec makeSpec() {
        LLMSpec spec;
        spec.shards.resize(1);  // in/out_state_name discovered from graphs
        spec.state_blocks.push_back(
            makeKVOnlyStateBlock(std::vector<std::vector<LayerRange>>{{LayerRange{0, kKVLayers - 1}}}));
        spec.hidden_size  = kHidden;
        spec.num_kv_heads = kKVHeads;
        spec.head_dim     = kHeadDim;
        spec.vocab_size   = kVocab;
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
        spec.shards.resize(1);
        spec.state_blocks.push_back(
            makeKVOnlyStateBlock(std::vector<std::vector<LayerRange>>{{LayerRange{0, kKVLayers - 1}}}));
        spec.hidden_size  = kHidden;
        spec.num_kv_heads = kKVHeads;
        spec.head_dim     = kHeadDim;
        spec.vocab_size   = kVocab;
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

}  // namespace geniex::testing
