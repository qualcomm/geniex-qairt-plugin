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

// ── parseHtpConfig ───────────────────────────────────────────────────────────
// These knobs used to be applied by QnnHtpNetRunExtensions; we now read them
// ourselves and hand them to the QNN C API, so the parsing is worth pinning down.

// Convenience: parse into locals seeded with the production defaults.
struct HtpKnobs {
    PerfProfile profile        = PerfProfile::BURST;
    uint32_t    rpc_latency_us = 0;
    bool        weight_sharing = false;

    void parse(const fs::path& p) { parseHtpConfig(p, profile, rpc_latency_us, weight_sharing); }
};

TEST_F(SpecLoaderBundleTest, ParseHtpConfigReadsAllKnobs) {
    auto     p = write("htp.json", R"({
  "devices": [{"soc_model": 60, "dsp_arch": "v73",
               "cores": [{"core_id": 0, "perf_profile": "burst", "rpc_control_latency": 100}]}],
  "memory": {"mem_type": "shared_buffer"},
  "context": {"weight_sharing_enabled": true}
})");
    HtpKnobs k;
    k.parse(p);
    EXPECT_EQ(k.profile, PerfProfile::BURST);
    EXPECT_EQ(k.rpc_latency_us, 100u);
    EXPECT_TRUE(k.weight_sharing);
}

TEST_F(SpecLoaderBundleTest, ParseHtpConfigMapsEveryProfileName) {
    const std::pair<const char*, PerfProfile> cases[] = {
        {"low_balanced", PerfProfile::LOW_BALANCED},
        {"balanced", PerfProfile::BALANCED},
        {"default", PerfProfile::DEFAULT},
        {"high_performance", PerfProfile::HIGH_PERFORMANCE},
        {"sustained_high_performance", PerfProfile::SUSTAINED_HIGH_PERFORMANCE},
        {"burst", PerfProfile::BURST},
        {"extreme_power_saver", PerfProfile::EXTREME_POWER_SAVER},
        {"low_power_saver", PerfProfile::LOW_POWER_SAVER},
        {"power_saver", PerfProfile::POWER_SAVER},
        {"high_power_saver", PerfProfile::HIGH_POWER_SAVER},
        {"system_settings", PerfProfile::SYSTEM_SETTINGS},
    };
    for (const auto& [name, expected] : cases) {
        auto     p = write(std::string("htp_") + name + ".json",
            std::string(R"({"devices":[{"cores":[{"perf_profile":")") + name + R"("}]}]})");
        HtpKnobs k;
        k.profile = PerfProfile::INVALID;
        k.parse(p);
        EXPECT_EQ(k.profile, expected) << "perf_profile: " << name;
    }
}

TEST_F(SpecLoaderBundleTest, ParseHtpConfigKeepsDefaultOnUnknownProfile) {
    auto     p = write("htp.json", R"({"devices":[{"cores":[{"perf_profile":"ludicrous_speed"}]}]})");
    HtpKnobs k;
    k.parse(p);
    EXPECT_EQ(k.profile, PerfProfile::BURST);  // untouched
}

TEST_F(SpecLoaderBundleTest, ParseHtpConfigOnlyFirstCoreVotes) {
    auto     p = write("htp.json", R"({"devices":[{"cores":[
        {"core_id":0,"perf_profile":"power_saver","rpc_control_latency":42},
        {"core_id":1,"perf_profile":"burst","rpc_control_latency":999}]}]})");
    HtpKnobs k;
    k.parse(p);
    EXPECT_EQ(k.profile, PerfProfile::POWER_SAVER);
    EXPECT_EQ(k.rpc_latency_us, 42u);
}

TEST_F(SpecLoaderBundleTest, ParseHtpConfigHandlesWeightSharingFalse) {
    auto     p = write("htp.json", R"({"context": {"weight_sharing_enabled": false}})");
    HtpKnobs k;
    k.weight_sharing = true;
    k.parse(p);
    EXPECT_FALSE(k.weight_sharing);
}

TEST_F(SpecLoaderBundleTest, ParseHtpConfigIgnoresDevicesWithoutCores) {
    auto     p = write("htp.json", R"({"devices":[{"soc_model":60,"dsp_arch":"v73"}]})");
    HtpKnobs k;
    k.parse(p);
    EXPECT_EQ(k.profile, PerfProfile::BURST);
    EXPECT_EQ(k.rpc_latency_us, 0u);
}

TEST_F(SpecLoaderBundleTest, ParseHtpConfigMissingFileLeavesKnobsAlone) {
    HtpKnobs k;
    k.parse(dir_ / "does_not_exist.json");
    EXPECT_EQ(k.profile, PerfProfile::BURST);
    EXPECT_EQ(k.rpc_latency_us, 0u);
    EXPECT_FALSE(k.weight_sharing);
}

TEST_F(SpecLoaderBundleTest, ParseHtpConfigMalformedJsonIsNotFatal) {
    auto     p = write("htp.json", "{ this is not json ");
    HtpKnobs k;
    EXPECT_NO_THROW(k.parse(p));
    EXPECT_EQ(k.profile, PerfProfile::BURST);
}

}  // namespace
}  // namespace geniex
