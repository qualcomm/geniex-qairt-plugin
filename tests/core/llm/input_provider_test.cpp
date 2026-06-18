// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause
//
// Unit tests for core/src/llm/input_provider.cpp — CPU-side tensor writers
// (token ids, embedding lookup, RoPE cos/sin). Each writes into a real Graph
// backed by a ClientBuffer IOTensor; no QNN runtime.

#include "llm/input_provider.h"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "IOTensor.hpp"
#include "graph.h"
#include "llm/llm_types.h"
#include "testing/graph_info_builder.hpp"

namespace {

using geniex::testing::GraphInfoBuilder;

geniex::Graph makeGraph(GraphInfoBuilder& b, IOTensor& io) {
    geniex::Graph g(&b.graphInfo(), /*api=*/nullptr, &io);
    EXPECT_TRUE(g.setup(nullptr));
    return g;
}

// Writes a flat float32 table to a temp file so the real loadTable path is used.
std::string writeRawTable(const std::vector<float>& table) {
    auto path = (std::filesystem::temp_directory_path() /
                 ("geniex_embed_" + std::to_string(::testing::UnitTest::GetInstance()->random_seed()) + ".bin"))
                    .string();
    std::ofstream f(path, std::ios::binary);
    f.write(reinterpret_cast<const char*>(table.data()), static_cast<std::streamsize>(table.size() * sizeof(float)));
    return path;
}

}  // namespace

// TokenIdInputProvider pads to tensor capacity with the pad token.
TEST(TokenIdInputProvider, WritesTokensAndPads) {
    GraphInfoBuilder b("g", {{"input_ids", QNN_DATATYPE_INT_32, {4}}}, {{"out", QNN_DATATYPE_INT_32, {4}}});
    IOTensor         io(BufferAlloc::DEFAULT);
    geniex::Graph    g = makeGraph(b, io);

    geniex::TokenIdInputProvider provider("input_ids", /*pad_token_id=*/-1);
    const std::vector<int32_t>   tokens = {7, 8};
    const geniex::LLMRunContext  ctx{tokens, /*n_past=*/0, /*curr_len=*/2, /*phase=*/0};
    provider.write(g, ctx);

    const auto* got = static_cast<const int32_t*>(g.inputPtr("input_ids"));
    EXPECT_EQ(std::vector<int32_t>(got, got + 4), (std::vector<int32_t>{7, 8, -1, -1}));
}

// Absent target tensor → silent no-op (no throw).
TEST(TokenIdInputProvider, MissingTensorIsNoOp) {
    GraphInfoBuilder b("g", {{"other", QNN_DATATYPE_INT_32, {2}}}, {{"out", QNN_DATATYPE_INT_32, {2}}});
    IOTensor         io(BufferAlloc::DEFAULT);
    geniex::Graph    g = makeGraph(b, io);

    geniex::TokenIdInputProvider provider("input_ids");
    const std::vector<int32_t>   tokens = {1};
    const geniex::LLMRunContext  ctx{tokens, 0, 1, 0};
    EXPECT_NO_THROW(provider.write(g, ctx));
}

// EmbeddingInputProvider looks up rows from its table into the embeds tensor.
TEST(EmbeddingInputProvider, WritesLookedUpRows) {
    const size_t     hidden = 3;
    const size_t     vocab  = 4;
    GraphInfoBuilder b(
        "g", {{"input_embeds", QNN_DATATYPE_FLOAT_32, {2, hidden}}}, {{"out", QNN_DATATYPE_FLOAT_32, {2, hidden}}});
    IOTensor      io(BufferAlloc::DEFAULT);
    geniex::Graph g = makeGraph(b, io);

    // Row r = {r, r+0.5, r+1.0}.
    std::vector<float> table(vocab * hidden);
    for (size_t r = 0; r < vocab; ++r)
        for (size_t c = 0; c < hidden; ++c) table[r * hidden + c] = static_cast<float>(r) + 0.5f * c;

    geniex::EmbeddingInputProvider provider("input_embeds");
    const std::string              table_path = writeRawTable(table);
    provider.loadTable(table_path, vocab, hidden);

    const std::vector<int32_t>  tokens = {2, 0};
    const geniex::LLMRunContext ctx{tokens, 0, 2, 0};
    provider.write(g, ctx);

    const auto* got = static_cast<const float*>(g.inputPtr("input_embeds"));
    EXPECT_EQ(std::vector<float>(got, got + 2 * hidden), (std::vector<float>{2.0f, 2.5f, 3.0f, 0.0f, 0.5f, 1.0f}));

    std::remove(table_path.c_str());
}

// RoPEInputProvider writes cos/sin tables sized to the graph tensor.
TEST(RoPEInputProvider, WritesCosSinTables) {
    const size_t     head_dim = 4;  // half_dim = 2
    const size_t     rows     = 3;
    const size_t     half     = head_dim / 2;
    GraphInfoBuilder b("g",
        {{"position_ids_cos", QNN_DATATYPE_FLOAT_32, {rows, half}},
            {"position_ids_sin", QNN_DATATYPE_FLOAT_32, {rows, half}}},
        {{"out", QNN_DATATYPE_FLOAT_32, {1}}});
    IOTensor         io(BufferAlloc::DEFAULT);
    geniex::Graph    g = makeGraph(b, io);

    geniex::RoPEInputProvider   provider(head_dim, /*theta=*/10000.0f, "position_ids_cos", "position_ids_sin");
    const std::vector<int32_t>  tokens = {0, 0, 0};
    const geniex::LLMRunContext ctx{tokens, /*n_past=*/0, /*curr_len=*/rows, /*phase=*/0};
    provider.write(g, ctx);

    const auto* cos = static_cast<const float*>(g.inputPtr("position_ids_cos"));
    const auto* sin = static_cast<const float*>(g.inputPtr("position_ids_sin"));

    // Position 0: cos = 1, sin = 0 for every frequency.
    for (size_t c = 0; c < half; ++c) {
        EXPECT_NEAR(cos[c], 1.0f, 1e-5f);
        EXPECT_NEAR(sin[c], 0.0f, 1e-5f);
    }
}
