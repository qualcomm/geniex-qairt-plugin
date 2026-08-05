// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause
//
// Unit tests for core/src/vlm/vlm_model.cpp - the VLM generate() orchestration
// (embedding lookup, vision encode, masked scatter, delegation to
// LLMModel::generate). A test-only subclass overrides the abstract
// encodeVision() with a deterministic fake and injects the LLM graph fixture,
// so no QNN device or real vision encoder is needed.

#include "vlm/vlm_model.h"

#include <gtest/gtest.h>

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "QnnApi.hpp"
#include "testing/llm_fixture.hpp"
#include "testing/stub_qnnapi.hpp"

namespace {

using geniex::testing::LLMFixture;

struct NoDecodePoolEnv {
    NoDecodePoolEnv() {
        _putenv_s("GENIEX_DECODE_WORKERS", "0");
        _putenv_s("GENIEX_CLOCK_KEEPER_THREADS", "0");
    }
    ~NoDecodePoolEnv() {
        _putenv_s("GENIEX_DECODE_WORKERS", "");
        _putenv_s("GENIEX_CLOCK_KEEPER_THREADS", "");
    }
};

// Test-only VLMModel: overrides encodeVision() with a fake, injects fixture
// graphs, and exposes maskedScatter + image_token_id_ for direct assertions.
class TestableVLMModel : public geniex::VLMModel {
   public:
    explicit TestableVLMModel(geniex::LLMSpec spec) : geniex::VLMModel(std::move(spec)) {}

    bool initFromFixture(LLMFixture& fx, std::unique_ptr<geniex::PrecomputedEmbeddingProvider> emb) {
        api_       = std::make_unique<QnnApi>();
        io_tensor_ = std::shared_ptr<IOTensor>(std::shared_ptr<void>{}, &fx.io);  // non-owning alias
        for (auto& g : fx.graphs) graphs_.push_back(std::move(g));
        setEmbeddingProvider(std::move(emb));
        const bool ok = onInitialized();
        initialized_  = ok;
        return ok;
    }

    void setImageTokenId(int32_t id) { image_token_id_ = id; }
    int  visionCalls() const { return vision_calls_; }

    // Expose the protected static for direct testing.
    using geniex::VLMModel::maskedScatter;

   protected:
    // Returns hidden_size floats per image token, each row filled with its index
    // so maskedScatter placement is observable.
    std::vector<float> encodeVision(const geniex::PixelData&) override {
        ++vision_calls_;
        std::vector<float> out(num_image_tokens_ * LLMFixture::kHidden);
        for (size_t r = 0; r < num_image_tokens_; ++r)
            for (size_t c = 0; c < LLMFixture::kHidden; ++c) out[r * LLMFixture::kHidden + c] = 100.0f + r;
        return out;
    }

   public:
    size_t num_image_tokens_ = 0;

   private:
    int vision_calls_ = 0;
};

// Test-only VLMModel built through the genie_config.json-aware constructor, so
// the config the subclass hands down can be read back off the base class.
class GenieConfigVLMModel : public geniex::VLMModel {
   public:
    GenieConfigVLMModel(geniex::LLMSpec spec, geniex::ParsedGenieConfig gc)
        : geniex::VLMModel(std::move(spec), std::move(gc)) {}

    const geniex::ParsedGenieConfig& genieConfig() const { return gc_; }
    const geniex::LLMSpec&           spec() const { return spec_; }

   protected:
    std::vector<float> encodeVision(const geniex::PixelData&) override { return {}; }
};

// Writes a flat float32 embedding table to a temp file (row r = {r, r, r, r}).
std::string writeTable(size_t vocab, size_t hidden) {
    std::vector<float> t(vocab * hidden);
    for (size_t r = 0; r < vocab; ++r)
        for (size_t c = 0; c < hidden; ++c) t[r * hidden + c] = static_cast<float>(r);
    auto          path = (std::filesystem::temp_directory_path() / "geniex_vlm_model_embed.bin").string();
    std::ofstream f(path, std::ios::binary);
    f.write(reinterpret_cast<const char*>(t.data()), static_cast<std::streamsize>(t.size() * sizeof(float)));
    return path;
}

std::unique_ptr<geniex::PrecomputedEmbeddingProvider> makeProvider() {
    auto emb  = std::make_unique<geniex::PrecomputedEmbeddingProvider>("input_embeds");
    auto path = writeTable(LLMFixture::kVocab, LLMFixture::kHidden);
    emb->loadTable(path, LLMFixture::kVocab, LLMFixture::kHidden);
    std::remove(path.c_str());
    return emb;
}

geniex::GenerationConfig greedyConfig(int32_t max_tokens) {
    geniex::GenerationConfig cfg;
    cfg.enable_sampling = false;
    cfg.max_tokens      = max_tokens;
    return cfg;
}

}  // namespace

// maskedScatter overwrites only the rows whose input id matches the target,
// pulling consecutive source rows in order.
TEST(VLMMaskedScatter, ReplacesMatchingRows) {
    const size_t               hidden = 2;
    std::vector<float>         embeds = {0, 0, 0, 0, 0, 0};  // 3 rows
    const std::vector<float>   vision = {1, 1, 2, 2};        // 2 source rows
    const std::vector<int32_t> ids    = {7, 9, 9};           // target = 9 -> rows 1,2

    TestableVLMModel::maskedScatter(embeds, vision, ids, /*target=*/9, hidden);
    EXPECT_EQ(embeds, (std::vector<float>{0, 0, 1, 1, 2, 2}));
}

// No matching token id -> embeds unchanged.
TEST(VLMMaskedScatter, NoMatchLeavesInputUntouched) {
    const size_t               hidden = 2;
    std::vector<float>         embeds = {5, 5, 6, 6};
    const std::vector<float>   vision = {9, 9};
    const std::vector<int32_t> ids    = {1, 2};
    TestableVLMModel::maskedScatter(embeds, vision, ids, /*target=*/99, hidden);
    EXPECT_EQ(embeds, (std::vector<float>{5, 5, 6, 6}));
}

// generate() without pixel data takes the text-only path: encodeVision is never
// called, and the stub-selected token is produced.
TEST(VLMModel, TextOnlySkipsVisionEncode) {
    NoDecodePoolEnv  no_pool;
    LLMFixture       fx;
    TestableVLMModel model{LLMFixture::makeSpec()};
    ASSERT_TRUE(model.initFromFixture(fx, makeProvider()));

    geniex::testing::stubSetVocabSize(LLMFixture::kVocab);
    geniex::testing::stubSetNextToken(5);

    geniex::VLMInput vlm_input;  // empty pixel_data
    const auto       out = model.generate({1, 2, 3}, vlm_input, greedyConfig(1));

    EXPECT_EQ(model.visionCalls(), 0);
    ASSERT_FALSE(out.empty());
    EXPECT_EQ(out.front(), 5);
}

// generate() with pixel data calls encodeVision once and scatters the vision
// embeddings into the image-token rows before delegating to the LLM loop.
TEST(VLMModel, WithPixelDataEncodesVisionOnce) {
    NoDecodePoolEnv  no_pool;
    LLMFixture       fx;
    TestableVLMModel model{LLMFixture::makeSpec()};
    model.num_image_tokens_ = 2;
    model.setImageTokenId(2);
    ASSERT_TRUE(model.initFromFixture(fx, makeProvider()));

    geniex::testing::stubSetVocabSize(LLMFixture::kVocab);
    geniex::testing::stubSetNextToken(6);

    geniex::VLMInput vlm_input;
    vlm_input.pixel_data.pixel_values = {0.1f, 0.2f};  // non-empty -> vision path
    const std::vector<int32_t> prompt = {1, 2, 2, 3};  // two image tokens (id 2)
    const auto                 out    = model.generate(prompt, vlm_input, greedyConfig(1));

    EXPECT_EQ(model.visionCalls(), 1);
    ASSERT_FALSE(out.empty());
    EXPECT_EQ(out.front(), 6);
}

// The genie_config.json-aware constructor forwards both arguments to LLMModel:
// families like Gemma4 read their per-layer embedding stream and local RoPE off
// gc_, so a dropped config would silently degrade to the text-only defaults.
TEST(VLMModel, GenieConfigCtorForwardsSpecAndConfig) {
    geniex::ParsedGenieConfig gc;
    gc.bos_token_id                      = 2;
    gc.eos_token_ids                     = {1, 106};
    gc.local_positional_encoding_present = true;
    gc.local_rope_theta                  = 10000.0f;
    gc.perlayer_embedding_size           = 8960;

    auto spec          = LLMFixture::makeSpec();
    spec.eos_token_ids = gc.eos_token_ids;

    const GenieConfigVLMModel model{spec, gc};

    EXPECT_EQ(model.genieConfig().bos_token_id, 2);
    EXPECT_EQ(model.genieConfig().eos_token_ids, (std::vector<int32_t>{1, 106}));
    EXPECT_TRUE(model.genieConfig().local_positional_encoding_present);
    EXPECT_EQ(model.genieConfig().perlayer_embedding_size, 8960u);

    // The spec argument is forwarded too, not dropped for the config.
    EXPECT_EQ(model.spec().eos_token_ids, (std::vector<int32_t>{1, 106}));
    ASSERT_EQ(model.spec().state_blocks.size(), 1u);
    EXPECT_EQ(model.spec().state_blocks[0].kind, geniex::StateBlockKind::KV);
}
