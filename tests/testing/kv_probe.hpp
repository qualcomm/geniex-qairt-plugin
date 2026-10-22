// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause
//
// Helpers for tests that check which tokens a model's KV cache holds. They rely
// on the QnnApi stub's KV-from-embeddings mode (stubSetKVFromEmbeds) and an
// embedding table whose row t is all-t, so the key/value row written for a
// token is that token's id.

#pragma once

#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

#include "graph.h"
#include "llm/input_provider.h"
#include "llm/llm_model.h"
#include "testing/stub_qnnapi.hpp"

namespace geniex::testing {

// Enables the stub's KV-from-embeddings mode for the scope of a test.
struct KVFromEmbedsScope {
    KVFromEmbedsScope() { stubSetKVFromEmbeds(true); }
    ~KVFromEmbedsScope() { stubSetKVFromEmbeds(false); }
    KVFromEmbedsScope(const KVFromEmbedsScope&)            = delete;
    KVFromEmbedsScope& operator=(const KVFromEmbedsScope&) = delete;
};

// Makes row t of the model's `input_embeds` table all-t.
inline void loadIdentityEmbeddingTable(geniex::LLMModel& model, uint32_t vocab, uint32_t hidden) {
    static int counter = 0;

    std::vector<float> table(static_cast<size_t>(vocab) * hidden);
    for (uint32_t r = 0; r < vocab; ++r)
        for (uint32_t c = 0; c < hidden; ++c) table[static_cast<size_t>(r) * hidden + c] = static_cast<float>(r);

    const auto path =
        (std::filesystem::temp_directory_path() / ("geniex_identity_table_" + std::to_string(counter++) + ".bin"))
            .string();
    {
        std::ofstream f(path, std::ios::binary);
        f.write(reinterpret_cast<const char*>(table.data()), static_cast<std::streamsize>(table.size() * 4));
    }
    model.findEmbeddingProvider("input_embeds")->loadTable(path, vocab, hidden);
    std::remove(path.c_str());
}

// Marks a value row whose dimensions disagree about the token it holds.
constexpr int32_t kInconsistentKVRow = std::numeric_limits<int32_t>::min();

struct KVRowTokens {
    std::vector<int32_t> keys;
    std::vector<int32_t> values;
};

// Token ids held by KV rows [0, n_rows) of the first layer of shard 0, for
// models with a single KV head. Token id 0 cannot be told from an empty row.
//
// Keys are read from the first head dimension, whose position does not depend on
// the cache stride; values are token-major, so every dimension of a row must
// agree. The test fixtures give each graph its own KV buffers rather than one
// shared one: prompt tokens land in the prefill graphs and tokens fed back by the
// decode loop in the decode graphs. A row is therefore taken from whichever graph
// holds it. `Model` is a test double that exposes graph(), graphIndex(),
// requireKVStateBlock() and spec_.
template <typename Model>
KVRowTokens kvRowTokens(Model& model, size_t n_rows) {
    const auto& pair = model.requireKVStateBlock().shard_pairs.at(0).at(0);

    std::vector<geniex::Graph*> graphs;
    for (size_t phase = 0; phase < 2; ++phase)
        for (size_t cl = 0; cl < model.spec_.context_lengths.size(); ++cl)
            graphs.push_back(&model.graph(model.graphIndex(phase, /*shard=*/0, cl)));

    const size_t head_dim = graphs.front()->inputSpec(pair.value_in).shape[3];

    KVRowTokens out;
    for (size_t t = 0; t < n_rows; ++t) {
        float              key = 0.0f;
        std::vector<float> value(head_dim, 0.0f);
        for (geniex::Graph* g : graphs) {
            const auto* keys  = static_cast<const float*>(g->inputPtr(pair.key_in));
            const auto* vals  = static_cast<const float*>(g->inputPtr(pair.value_in)) + t * head_dim;
            bool        empty = keys[t] == 0.0f;
            for (size_t d = 0; d < head_dim; ++d) empty = empty && vals[d] == 0.0f;
            if (empty) continue;
            key = keys[t];
            value.assign(vals, vals + head_dim);
            break;
        }

        bool consistent = true;
        for (float v : value) consistent = consistent && v == value.front();
        out.keys.push_back(static_cast<int32_t>(key));
        out.values.push_back(consistent ? static_cast<int32_t>(value.front()) : kInconsistentKVRow);
    }
    return out;
}

}  // namespace geniex::testing
