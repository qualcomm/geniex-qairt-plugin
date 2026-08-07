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
#include "xtensor/containers/xarray.hpp"
#include "xtensor/io/xnpy.hpp"

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

// Dumps a [vocab, hidden] table as a .npy so the xtensor load path is exercised.
std::string writeNpyTable(size_t vocab, size_t hidden) {
    xt::xarray<float> arr = xt::zeros<float>({vocab, hidden});
    for (size_t r = 0; r < vocab; ++r)
        for (size_t c = 0; c < hidden; ++c) arr(r, c) = static_cast<float>(r) + 0.5f * static_cast<float>(c);
    auto path = (std::filesystem::temp_directory_path() / "geniex_embed_table.npy").string();
    xt::dump_npy(path, arr);
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

// The explicit-config ctor (table_path + row width + pad override) is Gemma's
// per-layer embedding stream: onInitialized loads the dedicated table by its own
// row width, independent of spec.hidden_size, and picks the pad row by override.
TEST(EmbeddingInputProvider, ExplicitTablePathLoadsByRowWidth) {
    const size_t       row_hidden = 3, vocab = 4, rows = 2;
    std::vector<float> table(vocab * row_hidden);
    for (size_t r = 0; r < vocab; ++r)
        for (size_t c = 0; c < row_hidden; ++c) table[r * row_hidden + c] = static_cast<float>(r) + 0.5f * c;
    const std::string path = writeRawTable(table);

    // Explicit ctor: dedicated table, its own row width, pad override = token 0.
    geniex::EmbeddingInputProvider provider(
        "per_layer_inputs", path, /*row_hidden_size=*/row_hidden, /*pad_token_override=*/0);

    // spec.hidden_size deliberately differs from row_hidden to prove the
    // explicit row width wins; vocab_size comes from the spec (as in the real
    // Gemma flow, where it is inferred from the graphs before onInitialized).
    geniex::ModelConfig cfg;
    geniex::LLMSpec     spec;
    spec.vocab_size    = vocab;
    spec.hidden_size   = 999;  // must be ignored for this provider
    spec.eos_token_ids = {1};
    provider.onInitialized(cfg, spec);

    GraphInfoBuilder b(
        "g", {{"per_layer_inputs", QNN_DATATYPE_FLOAT_32, {rows, row_hidden}}}, {{"out", QNN_DATATYPE_FLOAT_32, {1}}});
    IOTensor      io(BufferAlloc::DEFAULT);
    geniex::Graph g = makeGraph(b, io);
    provider.write(g, geniex::LLMRunContext{{2, 0}, 0, 2, 1});

    const auto* got = static_cast<const float*>(g.inputPtr("per_layer_inputs"));
    EXPECT_EQ(
        std::vector<float>(got, got + rows * row_hidden), (std::vector<float>{2.0f, 2.5f, 3.0f, 0.0f, 0.5f, 1.0f}));
    std::remove(path.c_str());
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

// When the cos/sin tensors are quantized (UFIXED16, as in the on-device w4a16
// bundles), the provider must quantize with truncation so the written codes
// bit-match the encoding the graph was calibrated against. Round-to-nearest
// would shift codes by up to 1 LSB and perturb the RoPE rotation applied to keys.
// Regression guard for the on-device generation collapse this caused.
TEST(RoPEInputProvider, QuantizedCosSinUseTruncation) {
    const size_t  head_dim = 4;  // half_dim = 2
    const size_t  rows     = 4;
    const size_t  half     = head_dim / 2;
    const float   scale    = 3.0517578125e-05f;  // 1/32768
    const int32_t offset   = -32768;

    GraphInfoBuilder b("g",
        {{"position_ids_cos", QNN_DATATYPE_UFIXED_POINT_16, {rows, half}, scale, offset},
            {"position_ids_sin", QNN_DATATYPE_UFIXED_POINT_16, {rows, half}, scale, offset}},
        {{"out", QNN_DATATYPE_FLOAT_32, {1}}});
    IOTensor         io(BufferAlloc::DEFAULT);
    geniex::Graph    g = makeGraph(b, io);

    geniex::RoPEInputProvider   provider(head_dim, /*theta=*/10000.0f, "position_ids_cos", "position_ids_sin");
    const std::vector<int32_t>  tokens(rows, 0);
    const geniex::LLMRunContext ctx{tokens, /*n_past=*/0, /*curr_len=*/rows, /*phase=*/0};
    provider.write(g, ctx);

    // Recompute the reference cos/sin the provider produced, then quantize both
    // ways; the provider's bytes must match TowardZero and (on at least one code)
    // differ from Nearest.
    geniex::RotaryEmbedding rope(head_dim, /*theta=*/10000.0f);
    std::vector<int32_t>    pos(rows);
    for (size_t i = 0; i < rows; ++i) pos[i] = static_cast<int32_t>(i);
    auto [cos_ref, sin_ref] = rope.forward(pos);

    std::vector<uint16_t> cos_trunc(cos_ref.size()), cos_near(cos_ref.size());
    geniex::floatToTfN(
        cos_trunc.data(), cos_ref.data(), offset, scale, cos_ref.size(), geniex::RoundingMode::TowardZero);
    geniex::floatToTfN(cos_near.data(), cos_ref.data(), offset, scale, cos_ref.size(), geniex::RoundingMode::Nearest);

    const auto* cos = static_cast<const uint16_t*>(g.inputPtr("position_ids_cos"));
    EXPECT_EQ(std::vector<uint16_t>(cos, cos + cos_ref.size()), cos_trunc);
    EXPECT_NE(cos_trunc, cos_near) << "test scale/positions no longer expose a rounding difference";
}

// setRoundingMode(Nearest) overrides the truncating default. Models whose graphs were
// calibrated against round-to-nearest cos/sin tables (Gemma4) need this; leaving
// them truncated costs a systematic -0.5 LSB and degenerates long generations.
TEST(RoPEInputProvider, SetRoundingNearestOverridesDefault) {
    const size_t  head_dim = 4;  // half_dim = 2
    const size_t  rows     = 4;
    const size_t  half     = head_dim / 2;
    const float   scale    = 3.0517578125e-05f;  // 1/32768
    const int32_t offset   = -32768;

    GraphInfoBuilder b("g",
        {{"position_ids_cos", QNN_DATATYPE_UFIXED_POINT_16, {rows, half}, scale, offset},
            {"position_ids_sin", QNN_DATATYPE_UFIXED_POINT_16, {rows, half}, scale, offset}},
        {{"out", QNN_DATATYPE_FLOAT_32, {1}}});
    IOTensor         io(BufferAlloc::DEFAULT);
    geniex::Graph    g = makeGraph(b, io);

    geniex::RoPEInputProvider provider(head_dim, /*theta=*/10000.0f, "position_ids_cos", "position_ids_sin");
    EXPECT_EQ(provider.roundingMode(), geniex::RoundingMode::TowardZero) << "default must stay byte-preserving";
    provider.setRoundingMode(geniex::RoundingMode::Nearest);
    EXPECT_EQ(provider.roundingMode(), geniex::RoundingMode::Nearest);

    const std::vector<int32_t>  tokens(rows, 0);
    const geniex::LLMRunContext ctx{tokens, /*n_past=*/0, /*curr_len=*/rows, /*phase=*/0};
    provider.write(g, ctx);

    geniex::RotaryEmbedding rope(head_dim, /*theta=*/10000.0f);
    std::vector<int32_t>    pos(rows);
    for (size_t i = 0; i < rows; ++i) pos[i] = static_cast<int32_t>(i);
    auto [cos_ref, sin_ref] = rope.forward(pos);

    std::vector<uint16_t> cos_trunc(cos_ref.size()), cos_near(cos_ref.size());
    geniex::floatToTfN(
        cos_trunc.data(), cos_ref.data(), offset, scale, cos_ref.size(), geniex::RoundingMode::TowardZero);
    geniex::floatToTfN(cos_near.data(), cos_ref.data(), offset, scale, cos_ref.size(), geniex::RoundingMode::Nearest);
    ASSERT_NE(cos_trunc, cos_near) << "test scale/positions no longer expose a rounding difference";

    const auto* cos = static_cast<const uint16_t*>(g.inputPtr("position_ids_cos"));
    EXPECT_EQ(std::vector<uint16_t>(cos, cos + cos_ref.size()), cos_near);
}

namespace {
// Shared cos/sin graph builder for the RoPE-variant providers.
GraphInfoBuilder makeRopeGraph(size_t rows, size_t half) {
    const uint32_t r = static_cast<uint32_t>(rows), h = static_cast<uint32_t>(half);
    return GraphInfoBuilder("g",
        {{"position_ids_cos", QNN_DATATYPE_FLOAT_32, {r, h}}, {"position_ids_sin", QNN_DATATYPE_FLOAT_32, {r, h}}},
        {{"out", QNN_DATATYPE_FLOAT_32, {1}}});
}
}  // namespace

// LongRoPEInputProvider writes scaled cos/sin; at position 0 sin == 0.
TEST(LongRoPEInputProvider, WritesAtPositionZero) {
    const size_t     head_dim = 4, rows = 2, half = head_dim / 2;
    GraphInfoBuilder b = makeRopeGraph(rows, half);
    IOTensor         io(BufferAlloc::DEFAULT);
    geniex::Graph    g = makeGraph(b, io);

    geniex::LongRoPEInputProvider provider(head_dim,
        /*theta=*/10000.0f,
        /*ext_factors=*/{1.0f, 1.0f},
        /*max=*/2048,
        /*original=*/2048,
        "position_ids_cos",
        "position_ids_sin");
    const geniex::LLMRunContext   ctx{{0, 0}, /*n_past=*/0, /*curr_len=*/rows, /*phase=*/0};
    provider.write(g, ctx);

    const auto* sin = static_cast<const float*>(g.inputPtr("position_ids_sin"));
    for (size_t c = 0; c < half; ++c) EXPECT_NEAR(sin[c], 0.0f, 1e-5f);
}

// Llama3RoPEInputProvider writes cos/sin; position 0 -> cos 1, sin 0.
TEST(Llama3RoPEInputProvider, WritesAtPositionZero) {
    const size_t     head_dim = 8, rows = 2, half = head_dim / 2;
    GraphInfoBuilder b = makeRopeGraph(rows, half);
    IOTensor         io(BufferAlloc::DEFAULT);
    geniex::Graph    g = makeGraph(b, io);

    geniex::Llama3RoPEInputProvider provider(head_dim,
        /*theta=*/500000.0f,
        /*factor=*/8.0f,
        /*low_freq_factor=*/1.0f,
        /*high_freq_factor=*/4.0f,
        /*original=*/8192,
        "position_ids_cos",
        "position_ids_sin");
    const geniex::LLMRunContext     ctx{{0, 0}, 0, rows, 0};
    provider.write(g, ctx);

    const auto* cos = static_cast<const float*>(g.inputPtr("position_ids_cos"));
    const auto* sin = static_cast<const float*>(g.inputPtr("position_ids_sin"));
    for (size_t c = 0; c < half; ++c) {
        EXPECT_NEAR(cos[c], 1.0f, 1e-5f);
        EXPECT_NEAR(sin[c], 0.0f, 1e-5f);
    }
}

// PartialRoPEInputProvider applies the scale; position 0 -> cos == scale.
TEST(PartialRoPEInputProvider, AppliesScaleAtPositionZero) {
    const size_t     head_dim = 8, rows = 1, rope_half = 2;  // fraction 0.5 -> rope_dim 4 -> half 2
    GraphInfoBuilder b = makeRopeGraph(rows, rope_half);
    IOTensor         io(BufferAlloc::DEFAULT);
    geniex::Graph    g = makeGraph(b, io);

    geniex::PartialRoPEInputProvider provider(
        head_dim, /*theta=*/10000.0f, /*rope_fraction=*/0.5f, /*scale=*/2.0f, "position_ids_cos", "position_ids_sin");
    const geniex::LLMRunContext ctx{{0}, 0, rows, 0};
    provider.write(g, ctx);

    const auto* cos = static_cast<const float*>(g.inputPtr("position_ids_cos"));
    for (size_t c = 0; c < rope_half; ++c) EXPECT_NEAR(cos[c], 2.0f, 1e-5f);
}

// RoPE providers no-op when neither cos nor sin tensor is present.
TEST(RoPEInputProvider, MissingTensorsIsNoOp) {
    GraphInfoBuilder b("g", {{"other", QNN_DATATYPE_FLOAT_32, {2}}}, {{"out", QNN_DATATYPE_FLOAT_32, {1}}});
    IOTensor         io(BufferAlloc::DEFAULT);
    geniex::Graph    g = makeGraph(b, io);

    geniex::RoPEInputProvider provider(4, 10000.0f, "position_ids_cos", "position_ids_sin");
    EXPECT_NO_THROW(provider.write(g, geniex::LLMRunContext{{1}, 0, 1, 1}));
}

// ─── EmbeddingInputProvider: loadTable / onInitialized / padding ─────────────

// .npy load path: shape read from the header, rows written correctly.
TEST(EmbeddingInputProvider, LoadsNpyTable) {
    const size_t      vocab = 4, hidden = 3, rows = 2;
    const std::string path = writeNpyTable(vocab, hidden);

    geniex::EmbeddingInputProvider provider("input_embeds");
    provider.loadTable(path, /*vocab=*/0, /*hidden=*/0);  // dims read from header

    GraphInfoBuilder b(
        "g", {{"input_embeds", QNN_DATATYPE_FLOAT_32, {rows, hidden}}}, {{"out", QNN_DATATYPE_FLOAT_32, {1}}});
    IOTensor      io(BufferAlloc::DEFAULT);
    geniex::Graph g = makeGraph(b, io);
    provider.write(g, geniex::LLMRunContext{{2, 0}, 0, 2, 1});  // decode lookup

    const auto* got = static_cast<const float*>(g.inputPtr("input_embeds"));
    EXPECT_EQ(std::vector<float>(got, got + rows * hidden), (std::vector<float>{2.0f, 2.5f, 3.0f, 0.0f, 0.5f, 1.0f}));
    std::remove(path.c_str());
}

// .npy with hint dims that disagree with the header throws.
TEST(EmbeddingInputProvider, NpyShapeHintMismatchThrows) {
    const std::string              path = writeNpyTable(4, 3);
    geniex::EmbeddingInputProvider provider("input_embeds");
    EXPECT_THROW(provider.loadTable(path, /*vocab=*/4, /*hidden=*/8), std::runtime_error);
    std::remove(path.c_str());
}

// Raw binary with zero dims has no shape info -> throws.
TEST(EmbeddingInputProvider, RawRequiresNonZeroDims) {
    const std::string              path = writeRawTable({1.0f, 2.0f});
    geniex::EmbeddingInputProvider provider("input_embeds");
    EXPECT_THROW(provider.loadTable(path, /*vocab=*/0, /*hidden=*/0), std::runtime_error);
    std::remove(path.c_str());
}

// Missing raw file -> throws.
TEST(EmbeddingInputProvider, RawMissingFileThrows) {
    geniex::EmbeddingInputProvider provider("input_embeds");
    EXPECT_THROW(provider.loadTable("no_such_embedding_file.bin", 2, 2), std::runtime_error);
}

// Raw file whose byte size disagrees with vocab*hidden -> throws.
TEST(EmbeddingInputProvider, RawSizeMismatchThrows) {
    const std::string              path = writeRawTable({1.0f, 2.0f, 3.0f});  // 3 floats
    geniex::EmbeddingInputProvider provider("input_embeds");
    EXPECT_THROW(provider.loadTable(path, /*vocab=*/2, /*hidden=*/2), std::runtime_error);  // expects 4
    std::remove(path.c_str());
}

// onInitialized auto-loads from model_cfg.embedding_path and caches the EOS
// embedding as the prefill pad row.
TEST(EmbeddingInputProvider, OnInitializedLoadsAndCachesPad) {
    const size_t vocab = 4, hidden = 2;
    // Row r = {r, r}; eos=1 -> pad embedding {1, 1}.
    std::vector<float> table(vocab * hidden);
    for (size_t r = 0; r < vocab; ++r)
        for (size_t c = 0; c < hidden; ++c) table[r * hidden + c] = static_cast<float>(r);
    const std::string path = writeRawTable(table);

    geniex::ModelConfig cfg;
    cfg.embedding_path = path;
    geniex::LLMSpec spec;
    spec.vocab_size    = vocab;
    spec.hidden_size   = hidden;
    spec.eos_token_ids = {1};

    geniex::EmbeddingInputProvider provider("input_embeds");
    provider.onInitialized(cfg, spec);

    // Verify the table loaded by writing a decode lookup into a graph.
    GraphInfoBuilder b(
        "g", {{"input_embeds", QNN_DATATYPE_FLOAT_32, {1, hidden}}}, {{"out", QNN_DATATYPE_FLOAT_32, {1}}});
    IOTensor      io(BufferAlloc::DEFAULT);
    geniex::Graph g = makeGraph(b, io);
    provider.write(g, geniex::LLMRunContext{{3}, 0, 1, 1});
    const auto* got = static_cast<const float*>(g.inputPtr("input_embeds"));
    EXPECT_EQ(std::vector<float>(got, got + hidden), (std::vector<float>{3.0f, 3.0f}));  // table loaded
    std::remove(path.c_str());
}

// A prefill chunk shorter than the tensor pads trailing rows with the EOS
// embedding cached by onInitialized.
TEST(EmbeddingInputProvider, ShortChunkPadsWithEos) {
    const size_t       vocab = 4, hidden = 2, rows = 3;
    std::vector<float> table(vocab * hidden);
    for (size_t r = 0; r < vocab; ++r)
        for (size_t c = 0; c < hidden; ++c) table[r * hidden + c] = static_cast<float>(r);
    const std::string path = writeRawTable(table);

    geniex::ModelConfig cfg;
    cfg.embedding_path = path;
    geniex::LLMSpec spec;
    spec.vocab_size    = vocab;
    spec.hidden_size   = hidden;
    spec.eos_token_ids = {1};  // pad row = {1, 1}

    geniex::EmbeddingInputProvider provider("input_embeds");
    provider.onInitialized(cfg, spec);

    GraphInfoBuilder b(
        "g", {{"input_embeds", QNN_DATATYPE_FLOAT_32, {rows, hidden}}}, {{"out", QNN_DATATYPE_FLOAT_32, {1}}});
    IOTensor      io(BufferAlloc::DEFAULT);
    geniex::Graph g = makeGraph(b, io);

    // Only 2 tokens for a 3-row tensor -> row 2 is EOS-padded.
    const geniex::LLMRunContext ctx{{2, 0}, /*n_past=*/0, /*curr_len=*/2, /*phase=*/0};
    provider.write(g, ctx);

    const auto* got = static_cast<const float*>(g.inputPtr("input_embeds"));
    EXPECT_EQ(std::vector<float>(got, got + rows * hidden), (std::vector<float>{2, 2, 0, 0, 1, 1}));
    std::remove(path.c_str());
}

// setRoundingMode(Nearest) causes write() to round-to-nearest when quantizing
// into a UFIXED tensor — the behavior required by Gemma4's embedding LUT, which
// was calibrated against round-to-nearest. The default (TowardZero) must differ
// on a mid-code value so the test actually guards the distinction.
TEST(EmbeddingInputProvider, SetRoundingModeNearestRoundsUp) {
    // scale=1, offset=0 → q = trunc/round(src). 2.5 is the canonical mid-code.
    const float   scale  = 1.0f;
    const int32_t offset = 0;

    GraphInfoBuilder b("g",
        {{"input_embeds", QNN_DATATYPE_UFIXED_POINT_8, {1, 1}, scale, offset}},
        {{"out", QNN_DATATYPE_FLOAT_32, {1}}});
    IOTensor         io(BufferAlloc::DEFAULT);
    geniex::Graph    g = makeGraph(b, io);

    const std::vector<float> table = {2.5f};
    const std::string        path  = writeRawTable(table);

    geniex::EmbeddingInputProvider provider("input_embeds");
    provider.loadTable(path, /*vocab_size=*/1, /*hidden_size=*/1);
    provider.setRoundingMode(geniex::RoundingMode::Nearest);

    const geniex::LLMRunContext ctx{{0}, /*n_past=*/0, /*curr_len=*/1, /*phase=*/1};
    provider.write(g, ctx);

    const auto written = *static_cast<const uint8_t*>(g.inputPtr("input_embeds"));

    uint8_t nearest_code = 0, trunc_code = 0;
    geniex::floatToTfN(&nearest_code, table.data(), offset, scale, 1, geniex::RoundingMode::Nearest);
    geniex::floatToTfN(&trunc_code, table.data(), offset, scale, 1, geniex::RoundingMode::TowardZero);
    EXPECT_EQ(written, nearest_code) << "expected nearest (" << (int)nearest_code << "); got " << (int)written
                                     << " (toward_zero would be " << (int)trunc_code << ")";
    EXPECT_NE(nearest_code, trunc_code) << "test value no longer exposes a rounding difference";
    std::remove(path.c_str());
}
