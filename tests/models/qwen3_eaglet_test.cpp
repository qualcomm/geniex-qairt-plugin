// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause
//
// Unit tests for models/qwen3_eaglet/qwen3_eaglet.h::parseEagletConfig - the
// eaglet genie_config.json reader that resolves the draft engine, the shared
// RoPE base, and the trimmed draft-vocab->full-vocab token map. Every branch is
// exercised against temp bundles; no QNN device or graph bring-up.

#include "qwen3_eaglet.h"

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <string>

#include "llm/llm_spec_loader.h"

namespace {

namespace fs = std::filesystem;

// A temp bundle dir seeded with an eaglet genie_config.json plus optional
// side files (the draft-token-map). Cleaned up on destruction.
struct EagletBundle {
    fs::path dir;

    EagletBundle() {
        dir = fs::temp_directory_path() / ("geniex_eaglet_test_" + std::to_string(counter_++));
        fs::create_directories(dir);
    }
    ~EagletBundle() {
        std::error_code ec;
        fs::remove_all(dir, ec);
    }
    EagletBundle(const EagletBundle&)            = delete;
    EagletBundle& operator=(const EagletBundle&) = delete;

    void write(const std::string& name, const std::string& body) const { std::ofstream(dir / name) << body; }

   private:
    static inline int counter_ = 0;
};

// A well-formed eaglet genie_config with target+draft engines sharing rope-theta
// and a draft ctx-bin. `token_map_file` names the draft-token-map (omitted if
// empty); `draft_theta` lets a test force a target/draft mismatch.
std::string eagletConfig(const std::string& token_map_file = "", float draft_theta = 1000000.0f) {
    std::string tm_field = token_map_file.empty() ? "" : R"(, "draft-token-map": ")" + token_map_file + R"(")";
    return R"({
        "dialog": {
            "type": "eaglet",
            "context": { "bos-token": 1, "eos-token": [2, 3], "pad-token": 0 },
            "eaglet": { "draft-len": 4, "n-branches": 3, "max-tokens-target-can-evaluate": 8 },
            "embedding": { "lut-path": "quantized_embedding_table.bin", "size": 4 },
            "engine": [
                {
                    "role": "target",
                    "model": {
                        "positional-encoding": { "rope-theta": 1000000.0 }
                    }
                },
                {
                    "role": "draft",
                    "model": {
                        "positional-encoding": { "rope-theta": )" +
           std::to_string(draft_theta) + R"( },
                        "binary": { "ctx-bins": ["draft_ar1_cl512.bin"] })" +
           tm_field + R"(
                    }
                }
            ]
        }
    })";
}

geniex::ParsedGenieConfig eagletGenieConfig() {
    geniex::ParsedGenieConfig gc;
    gc.dialog_type = "eaglet";
    return gc;
}

}  // namespace

// The draft-len / n-branches / max-verify knobs and the shared RoPE base are
// read straight off the eaglet + engine blocks.
TEST(ParseEagletConfig, ReadsCoreKnobsAndRopeTheta) {
    EagletBundle b;
    b.write("genie_config.json", eagletConfig());

    const auto cfg = geniex::qwen3_eaglet::parseEagletConfig(b.dir, eagletGenieConfig());

    EXPECT_EQ(cfg.draft_len, 4u);
    EXPECT_EQ(cfg.n_branches, 3u);
    EXPECT_EQ(cfg.max_verify_tokens, 8u);
    EXPECT_FLOAT_EQ(cfg.rope_theta, 1000000.0f);
    ASSERT_EQ(cfg.draft_model_paths.size(), 1u);
    EXPECT_NE(cfg.draft_model_paths[0].find("draft_ar1_cl512.bin"), std::string::npos);
}

// An array draft-token-map is a direct draft-index -> full-vocab-id list, so it
// is preserved in array order.
TEST(ParseEagletConfig, ArrayTokenMapPreservesOrder) {
    EagletBundle b;
    b.write("genie_config.json", eagletConfig("token_map.json"));
    b.write("token_map.json", "[10, 20, 30, 40]");

    const auto cfg = geniex::qwen3_eaglet::parseEagletConfig(b.dir, eagletGenieConfig());

    EXPECT_EQ(cfg.draft_token_map, (std::vector<int32_t>{10, 20, 30, 40}));
}

// An object-keyed draft-token-map must be placed by the PARSED numeric key, not
// nlohmann's lexicographic key order ("0","1","10","2",...). This is the map-
// scrambling guard: entry "10" must land at index 10, and gaps become 0.
TEST(ParseEagletConfig, ObjectTokenMapPlacedByNumericKey) {
    EagletBundle b;
    b.write("genie_config.json", eagletConfig("token_map.json"));
    // Deliberately out-of-order, sparse keys with a two-digit key that sorts
    // before "2" lexicographically.
    b.write("token_map.json", R"({"2": 300, "10": 999, "0": 100})");

    const auto cfg = geniex::qwen3_eaglet::parseEagletConfig(b.dir, eagletGenieConfig());

    ASSERT_EQ(cfg.draft_token_map.size(), 11u);  // sized to max_key + 1
    EXPECT_EQ(cfg.draft_token_map[0], 100);
    EXPECT_EQ(cfg.draft_token_map[1], 0);  // gap
    EXPECT_EQ(cfg.draft_token_map[2], 300);
    EXPECT_EQ(cfg.draft_token_map[10], 999);
}

// EAGLE drives both engines on one RoPE base; a target/draft rope-theta mismatch
// is a hard config error.
TEST(ParseEagletConfig, RejectsRopeThetaMismatch) {
    EagletBundle b;
    b.write("genie_config.json", eagletConfig(/*token_map_file=*/"", /*draft_theta=*/500000.0f));

    EXPECT_THROW(geniex::qwen3_eaglet::parseEagletConfig(b.dir, eagletGenieConfig()), std::runtime_error);
}

// A draft engine with no ctx-bins leaves draft_model_paths empty, which is fatal.
TEST(ParseEagletConfig, RejectsMissingDraftEngine) {
    EagletBundle b;
    b.write("genie_config.json", R"({
        "dialog": {
            "type": "eaglet",
            "eaglet": { "draft-len": 4 },
            "embedding": { "lut-path": "quantized_embedding_table.bin" },
            "engine": [
                { "role": "target", "model": { "positional-encoding": { "rope-theta": 1000000.0 } } }
            ]
        }
    })");

    EXPECT_THROW(geniex::qwen3_eaglet::parseEagletConfig(b.dir, eagletGenieConfig()), std::runtime_error);
}

// Both engines must declare a rope-theta; a draft engine missing it is fatal.
TEST(ParseEagletConfig, RejectsMissingRopeTheta) {
    EagletBundle b;
    b.write("genie_config.json", R"({
        "dialog": {
            "type": "eaglet",
            "eaglet": { "draft-len": 4 },
            "embedding": { "lut-path": "quantized_embedding_table.bin" },
            "engine": [
                { "role": "target", "model": { "positional-encoding": { "rope-theta": 1000000.0 } } },
                { "role": "draft",  "model": { "binary": { "ctx-bins": ["draft_ar1_cl512.bin"] } } }
            ]
        }
    })");

    EXPECT_THROW(geniex::qwen3_eaglet::parseEagletConfig(b.dir, eagletGenieConfig()), std::runtime_error);
}

// A bundle with no eaglet-typed genie_config.json at all fails the file scan.
TEST(ParseEagletConfig, RejectsBundleWithoutEagletConfig) {
    EagletBundle b;
    b.write("genie_config.json", R"({ "dialog": { "type": "basic" } })");

    EXPECT_THROW(geniex::qwen3_eaglet::parseEagletConfig(b.dir, eagletGenieConfig()), std::runtime_error);
}

// The draft embedding table is resolved by the "draft_" + target-lut convention
// when the file exists next to the config.
TEST(ParseEagletConfig, ResolvesDraftEmbeddingByConvention) {
    EagletBundle b;
    b.write("genie_config.json", eagletConfig());
    b.write("draft_quantized_embedding_table.bin", "stub");

    const auto cfg = geniex::qwen3_eaglet::parseEagletConfig(b.dir, eagletGenieConfig());

    ASSERT_FALSE(cfg.draft_embedding_path.empty());
    EXPECT_NE(cfg.draft_embedding_path.find("draft_quantized_embedding_table.bin"), std::string::npos);
}
