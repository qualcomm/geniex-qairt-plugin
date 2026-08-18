// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause
//
// Bundle-parsing coverage for llm_spec_loader: parseHtpCoreCount and the
// htp_backend_ext_config.json → ModelConfig.num_cores wiring in
// modelConfigFromDirectory. Pure filesystem + JSON logic; no QNN runtime.

#include "llm/llm_spec_loader.h"

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <string>

namespace geniex {
namespace {

namespace fs = std::filesystem;

class SpecLoaderBundleTest : public ::testing::Test {
   protected:
    void SetUp() override {
        dir_ = fs::temp_directory_path() /
               ("geniex_spec_loader_test_" + std::to_string(::testing::UnitTest::GetInstance()->random_seed()) + "_" +
                   ::testing::UnitTest::GetInstance()->current_test_info()->name());
        fs::create_directories(dir_);
    }

    void TearDown() override {
        std::error_code ec;
        fs::remove_all(dir_, ec);
    }

    fs::path write(const std::string& name, const std::string& content) {
        fs::path p = dir_ / name;
        std::ofstream(p) << content;
        return p;
    }

    fs::path dir_;
};

constexpr const char* kSingleCoreConfig = R"({
  "devices": [
    {
      "soc_model": 77,
      "dsp_arch": "v73",
      "cores": [ { "core_id": 0, "perf_profile": "burst", "rpc_control_latency": 100 } ]
    }
  ],
  "memory": { "mem_type": "shared_buffer" },
  "context": { "weight_sharing_enabled": true }
})";

constexpr const char* kDualCoreConfig = R"({
  "devices": [
    {
      "soc_model": 77,
      "dsp_arch": "v73",
      "cores": [
        { "core_id": 0, "perf_profile": "burst", "rpc_control_latency": 100 },
        { "core_id": 1, "perf_profile": "burst", "rpc_control_latency": 100 }
      ]
    }
  ]
})";

TEST_F(SpecLoaderBundleTest, CoreCountMissingFileIsZero) {
    EXPECT_EQ(parseHtpCoreCount(dir_ / "nonexistent.json"), 0u);
}

TEST_F(SpecLoaderBundleTest, CoreCountMalformedJsonIsZero) {
    auto p = write("htp_backend_ext_config.json", "{ not json");
    EXPECT_EQ(parseHtpCoreCount(p), 0u);
}

TEST_F(SpecLoaderBundleTest, CoreCountNoDevicesKeyIsZero) {
    auto p = write("htp_backend_ext_config.json", R"({"memory": {"mem_type": "shared_buffer"}})");
    EXPECT_EQ(parseHtpCoreCount(p), 0u);
}

TEST_F(SpecLoaderBundleTest, CoreCountDevicesNotArrayIsZero) {
    auto p = write("htp_backend_ext_config.json", R"({"devices": {"cores": []}})");
    EXPECT_EQ(parseHtpCoreCount(p), 0u);
}

TEST_F(SpecLoaderBundleTest, CoreCountDeviceWithoutCoresIsZero) {
    auto p = write("htp_backend_ext_config.json", R"({"devices": [{"soc_model": 77}]})");
    EXPECT_EQ(parseHtpCoreCount(p), 0u);
}

TEST_F(SpecLoaderBundleTest, CoreCountSingleCore) {
    auto p = write("htp_backend_ext_config.json", kSingleCoreConfig);
    EXPECT_EQ(parseHtpCoreCount(p), 1u);
}

TEST_F(SpecLoaderBundleTest, CoreCountDualCore) {
    auto p = write("htp_backend_ext_config.json", kDualCoreConfig);
    EXPECT_EQ(parseHtpCoreCount(p), 2u);
}

TEST_F(SpecLoaderBundleTest, CoreCountTakesMaxAcrossDevices) {
    auto p = write("htp_backend_ext_config.json", R"({
      "devices": [
        { "cores": [ { "core_id": 0 } ] },
        { "cores": [ { "core_id": 0 }, { "core_id": 1 }, { "core_id": 2 } ] }
      ]
    })");
    EXPECT_EQ(parseHtpCoreCount(p), 3u);
}

TEST_F(SpecLoaderBundleTest, ModelConfigPicksUpCoreCountFromBundle) {
    write("tokenizer.json", "{}");
    write("model.bin", "stub");
    write("htp_backend_ext_config.json", kDualCoreConfig);

    ModelConfig cfg = modelConfigFromDirectory(dir_);
    EXPECT_EQ(cfg.num_cores, 2u);
    EXPECT_FALSE(cfg.htp_config_path.empty());
}

TEST_F(SpecLoaderBundleTest, ModelConfigDefaultsToZeroCoresWithoutHtpConfig) {
    write("tokenizer.json", "{}");
    write("model.bin", "stub");

    ModelConfig cfg = modelConfigFromDirectory(dir_);
    EXPECT_EQ(cfg.num_cores, 0u);
    EXPECT_TRUE(cfg.htp_config_path.empty());
}

// ─────────────────────────────────────────────────────────────────────────────
// parseQAIRTMetadata: per-head-group KV tensor names
// ─────────────────────────────────────────────────────────────────────────────

// One shard, 2 KV layers, GQA groups split across tensors: "past_key_<N>_h<G>_in".
// Gemma4 E4B W4A16 (QAIRT 2.45) exports look like this.
constexpr const char* kSplitHeadMetadata = R"({
  "model_id": "gemma_4_e4b_it",
  "model_files": {
    "part1_of_1.bin": {
      "inputs": {
        "inputs_embeds":      { "shape": [1, 128, 2560], "dtype": "float32" },
        "past_key_0_h0_in":   { "shape": [1, 1, 512, 4095], "dtype": "uint16" },
        "past_key_0_h1_in":   { "shape": [1, 1, 512, 4095], "dtype": "uint16" },
        "past_value_0_h0_in": { "shape": [1, 1, 4095, 512], "dtype": "uint16" },
        "past_value_0_h1_in": { "shape": [1, 1, 4095, 512], "dtype": "uint16" },
        "past_key_1_h0_in":   { "shape": [1, 1, 512, 4095], "dtype": "uint16" },
        "past_key_1_h1_in":   { "shape": [1, 1, 512, 4095], "dtype": "uint16" },
        "past_value_1_h0_in": { "shape": [1, 1, 4095, 512], "dtype": "uint16" },
        "past_value_1_h1_in": { "shape": [1, 1, 4095, 512], "dtype": "uint16" }
      },
      "outputs": {
        "logits": { "shape": [1, 128, 262144], "dtype": "float32" }
      }
    }
  }
})";

// The classic unsplit layout: one tensor per layer, groups in shape[0].
constexpr const char* kUnsplitHeadMetadata = R"({
  "model_id": "llama_stub",
  "model_files": {
    "part1_of_1.bin": {
      "inputs": {
        "inputs_embeds":   { "shape": [1, 128, 2048], "dtype": "float32" },
        "past_key_0_in":   { "shape": [4, 1, 128, 4095], "dtype": "uint16" },
        "past_value_0_in": { "shape": [4, 1, 4095, 128], "dtype": "uint16" },
        "past_key_1_in":   { "shape": [4, 1, 128, 4095], "dtype": "uint16" },
        "past_value_1_in": { "shape": [4, 1, 4095, 128], "dtype": "uint16" }
      },
      "outputs": {
        "logits": { "shape": [1, 128, 32000], "dtype": "float32" }
      }
    }
  }
})";

// Regression: a digit-only "past_key_(\d+)_(in|out)" match rejects every
// "_h<G>"-infixed name, so no KV layer is discovered, num_hidden_layers stays 1
// and num_kv_heads is read as shape[0] (== 1 per split tensor). On Gemma4 E4B
// that under-allocates the KV cache and faults on the first graph execution.
TEST_F(SpecLoaderBundleTest, MetadataCountsSplitPerHeadKVGroups) {
    write("tokenizer.json", "{}");
    write("metadata.json", kSplitHeadMetadata);

    ParsedQAIRTMetadata md = parseQAIRTMetadata(dir_);
    EXPECT_EQ(md.num_hidden_layers, 2u);  // past_key_0 + past_key_1
    EXPECT_EQ(md.num_kv_heads, 2u);       // _h0 + _h1, NOT shape[0] == 1
    EXPECT_EQ(md.head_dim, 512u);
}

TEST_F(SpecLoaderBundleTest, MetadataKeepsUnsplitKVHeadCount) {
    write("tokenizer.json", "{}");
    write("metadata.json", kUnsplitHeadMetadata);

    ParsedQAIRTMetadata md = parseQAIRTMetadata(dir_);
    EXPECT_EQ(md.num_hidden_layers, 2u);
    EXPECT_EQ(md.num_kv_heads, 4u);  // from shape[0]; no "_h<G>" infix present
    EXPECT_EQ(md.head_dim, 128u);
}

}  // namespace
}  // namespace geniex
