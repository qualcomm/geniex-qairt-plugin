// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause
//
// Bundle-parsing coverage for llm_spec_loader: parseHtpCoreCount, the
// htp_backend_ext_config.json → ModelConfig.num_cores wiring in
// modelConfigFromDirectory, and parseModelArchitecture. Pure filesystem + JSON
// logic; no QNN runtime.

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

// ── parseModelArchitecture ───────────────────────────────────────────────────
// Some real exports (Qwen3-VL, Gemma4) ship no config.json at all, so every
// early-return here is a live case, not just defensive code.

TEST_F(SpecLoaderBundleTest, ArchitectureMissingConfigJsonIsEmpty) {
    // No config.json written.
    EXPECT_EQ(parseModelArchitecture(dir_), "");
}

TEST_F(SpecLoaderBundleTest, ArchitectureMalformedJsonIsEmpty) {
    write("config.json", "{not valid json");
    EXPECT_EQ(parseModelArchitecture(dir_), "");
}

TEST_F(SpecLoaderBundleTest, ArchitectureMissingKeyIsEmpty) {
    write("config.json", R"({"model_type": "phi3"})");
    EXPECT_EQ(parseModelArchitecture(dir_), "");
}

TEST_F(SpecLoaderBundleTest, ArchitectureNotArrayIsEmpty) {
    write("config.json", R"({"architectures": "Phi3ForCausalLM"})");
    EXPECT_EQ(parseModelArchitecture(dir_), "");
}

TEST_F(SpecLoaderBundleTest, ArchitectureEmptyArrayIsEmpty) {
    write("config.json", R"({"architectures": []})");
    EXPECT_EQ(parseModelArchitecture(dir_), "");
}

TEST_F(SpecLoaderBundleTest, ArchitectureFirstElementNotStringIsEmpty) {
    write("config.json", R"({"architectures": [123]})");
    EXPECT_EQ(parseModelArchitecture(dir_), "");
}

TEST_F(SpecLoaderBundleTest, ArchitectureReadsFirstElement) {
    write("config.json", R"({"architectures": ["Qwen3ForCausalLM", "SomethingElse"]})");
    EXPECT_EQ(parseModelArchitecture(dir_), "Qwen3ForCausalLM");
}

}  // namespace
}  // namespace geniex
