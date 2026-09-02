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

// -- parseHtpConfig -----------------------------------------------------------
// These knobs used to be applied by QnnHtpNetRunExtensions; we now read them
// ourselves and hand them to the QNN C API, so the parsing is worth pinning down.
// Only keys the QAIRT schema marks "Used by qnn-net-run" are applied -- see
// <qairt-sdk>/docs/QAIRT-Docs/QNN/general/htp/htp_backend.html

TEST_F(SpecLoaderBundleTest, ParseHtpConfigReadsEveryLoadTimeKnob) {
    auto          p = write("htp.json", R"({
  "devices": [{"soc_model": 60, "dsp_arch": "v73",
               "cores": [{"core_id": 0, "perf_profile": "burst", "rpc_control_latency": 100,
                          "rpc_polling_time": 9999, "hmx_timeout_us": 300000,
                          "adaptive_polling_time": 42}]}],
  "memory": {"mem_type": "shared_buffer"}
})");
    HtpPerfConfig cfg;
    parseHtpConfig(p, cfg);
    EXPECT_EQ(cfg.profile, PerfProfile::BURST);
    EXPECT_EQ(cfg.rpc_control_latency_us, 100u);
    EXPECT_EQ(cfg.rpc_polling_time_us, 9999u);
    EXPECT_EQ(cfg.hmx_timeout_us, 300000u);
    EXPECT_EQ(cfg.adaptive_polling_time_us, 42u);
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
        auto          p = write(std::string("htp_") + name + ".json",
            std::string(R"({"devices":[{"cores":[{"perf_profile":")") + name + R"("}]}]})");
        HtpPerfConfig cfg;
        cfg.profile = PerfProfile::INVALID;
        parseHtpConfig(p, cfg);
        EXPECT_EQ(cfg.profile, expected) << "perf_profile: " << name;
    }
}

TEST_F(SpecLoaderBundleTest, ParseHtpConfigKeepsDefaultOnUnknownProfile) {
    auto          p = write("htp.json", R"({"devices":[{"cores":[{"perf_profile":"ludicrous_speed"}]}]})");
    HtpPerfConfig cfg;
    parseHtpConfig(p, cfg);
    EXPECT_EQ(cfg.profile, PerfProfile::BURST);  // untouched
}

TEST_F(SpecLoaderBundleTest, ParseHtpConfigOnlyFirstCoreVotes) {
    auto          p = write("htp.json", R"({"devices":[{"cores":[
        {"core_id":0,"perf_profile":"power_saver","rpc_control_latency":42},
        {"core_id":1,"perf_profile":"burst","rpc_control_latency":999}]}]})");
    HtpPerfConfig cfg;
    parseHtpConfig(p, cfg);
    EXPECT_EQ(cfg.profile, PerfProfile::POWER_SAVER);
    EXPECT_EQ(cfg.rpc_control_latency_us, 42u);
}

// weight_sharing_enabled is annotated "Used by qnn-context-binary-generator during
// offline preparation": it is decided when the .bin is generated, so a runtime that
// loads a prebuilt binary must NOT try to apply it. It must also not warn -- ignoring
// it is correct behaviour, and a warning here would train people to ignore warnings.
TEST_F(SpecLoaderBundleTest, ParseHtpConfigIgnoresOfflinePreparationKeys) {
    auto          p = write("htp.json", R"({
  "graphs": [{"vtcm_mb": 8, "weights_packing": true, "num_cores": 4}],
  "context": {"weight_sharing_enabled": true},
  "devices": [{"soc_model": 60, "dsp_arch": "v73",
               "cores": [{"core_id": 0, "perf_profile": "burst"}]}]
})");
    HtpPerfConfig cfg;
    EXPECT_NO_THROW(parseHtpConfig(p, cfg));
    // The load-time key still lands...
    EXPECT_EQ(cfg.profile, PerfProfile::BURST);
    // ...and nothing offline leaked into the load-time knobs.
    EXPECT_EQ(cfg.rpc_control_latency_us, 0u);
    EXPECT_EQ(cfg.hmx_timeout_us, 0u);
}

TEST_F(SpecLoaderBundleTest, ParseHtpConfigMissingFileLeavesKnobsAlone) {
    HtpPerfConfig cfg;
    parseHtpConfig(dir_ / "does_not_exist.json", cfg);
    EXPECT_EQ(cfg.profile, PerfProfile::BURST);
    EXPECT_EQ(cfg.rpc_control_latency_us, 0u);
    EXPECT_EQ(cfg.rpc_polling_time_us, 0u);
}

TEST_F(SpecLoaderBundleTest, ParseHtpConfigMalformedJsonIsNotFatal) {
    auto          p = write("htp.json", "{ this is not json ");
    HtpPerfConfig cfg;
    EXPECT_NO_THROW(parseHtpConfig(p, cfg));
    EXPECT_EQ(cfg.profile, PerfProfile::BURST);
}

TEST_F(SpecLoaderBundleTest, ParseHtpConfigWarnsOnKeysItDoesNotApply) {
    // Keys that are neither applied nor documented offline-only must warn, so a bundle
    // relying on one is never silently downgraded.
    auto          p = write("htp.json", R"({
  "devices": [{"pd_session": "unsigned", "cores": [{"core_id": 0, "profiling_level": "basic"}]}],
  "context": {"share_resources": true, "reused_io_limit_mb": 32},
  "unexpected_section": {"a": 1}
})");
    HtpPerfConfig cfg;
    EXPECT_NO_THROW(parseHtpConfig(p, cfg));
    EXPECT_EQ(cfg.profile, PerfProfile::BURST);
}

// ── parseModelArchitecture ───────────────────────────────────────────────────

TEST_F(SpecLoaderBundleTest, ArchitectureMissingConfigJsonIsEmpty) { EXPECT_EQ(parseModelArchitecture(dir_), ""); }

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
