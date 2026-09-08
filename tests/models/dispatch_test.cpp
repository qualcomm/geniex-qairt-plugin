// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause
//
// Routing coverage for models/dispatch.h -- nothing else in the repo includes
// it, so this also type-checks every inline function in the header. The
// accepting path needs real binaries + an NPU and isn't asserted here; it was
// verified manually (see the PR).

#include "dispatch.h"

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <string>

namespace geniex {
namespace {

namespace fs = std::filesystem;

class DispatchBundleTest : public ::testing::Test {
   protected:
    void SetUp() override {
        dir_ = fs::temp_directory_path() /
               ("geniex_dispatch_test_" + std::to_string(::testing::UnitTest::GetInstance()->random_seed()) + "_" +
                   ::testing::UnitTest::GetInstance()->current_test_info()->name());
        fs::create_directories(dir_);
    }

    void TearDown() override {
        std::error_code ec;
        fs::remove_all(dir_, ec);
    }

    void write(const std::string& name, const std::string& content) const { std::ofstream(dir_ / name) << content; }

    // parseQAIRTMetadata requires at least one parsable shard entry.
    void writeMetadata(const std::string& model_id, bool vision) const {
        std::string j = R"({"model_id": ")" + model_id + R"(", "model_files": {"part1_of_1.bin": {)" +
                        R"("inputs": {"input_ids": {"shape": [1, 1], "dtype": "int32"},)" +
                        R"("past_key_0_in": {"shape": [8, 1, 128, 4095], "dtype": "uint8"},)" +
                        R"("past_value_0_in": {"shape": [8, 1, 4095, 128], "dtype": "uint8"}},)" +
                        R"("outputs": {"logits": {"shape": [1, 1, 32000], "dtype": "uint16"},)" +
                        R"("past_key_0_out": {"shape": [8, 1, 128, 1], "dtype": "uint8"},)" +
                        R"("past_value_0_out": {"shape": [8, 1, 1, 128], "dtype": "uint8"}}}})";
        if (vision) {
            j += R"(, "genie": {"vision_preprocessing": {"image_width": 448, "image_height": 448)"
                 R"(, "patch_size": 14, "temporal_patch_size": 1, "spatial_merge_size": 2}})";
        }
        j += "}";
        write("metadata.json", j);
    }

    void writeGenieConfig(const std::string& dialog_type) const {
        write("genie_config.json",
            R"({"dialog": {"type": ")" + dialog_type + R"(", "context": {"bos-token": 1, "eos-token": 2}}})");
    }

    void writeConfigJson(const std::string& architecture) const {
        write("config.json", R"({"architectures": [")" + architecture + R"("]})");
    }

    // The .bin need not exist: every case here returns before the backend is touched.
    ModelConfig cfg() const {
        ModelConfig c;
        c.model_paths = {(dir_ / "part1_of_1.bin").string()};
        return c;
    }

    fs::path dir_;
};

TEST_F(DispatchBundleTest, FactsReadModelId) {
    writeMetadata("phi_4_mini_instruct", /*vision=*/false);
    const auto f = dispatch_detail::bundleFactsOf(cfg());
    ASSERT_TRUE(f.has_value());
    EXPECT_EQ(f->model_id, "phi_4_mini_instruct");
    EXPECT_FALSE(f->multimodal);
}

TEST_F(DispatchBundleTest, FactsDetectVisionUnderGenieKey) {
    writeMetadata("intern3_5_vl_2b", /*vision=*/true);
    const auto f = dispatch_detail::bundleFactsOf(cfg());
    ASSERT_TRUE(f.has_value());
    EXPECT_TRUE(f->multimodal);
}

TEST_F(DispatchBundleTest, FactsDialogTypeDefaultsToBasic) {
    writeMetadata("llama_v3_2_3b_instruct", /*vision=*/false);
    const auto f = dispatch_detail::bundleFactsOf(cfg());
    ASSERT_TRUE(f.has_value());
    EXPECT_EQ(f->dialog_type, "basic");
}

TEST_F(DispatchBundleTest, FactsReadDialogType) {
    writeMetadata("qwen3_4b", /*vision=*/false);
    writeGenieConfig("eaglet");
    const auto f = dispatch_detail::bundleFactsOf(cfg());
    ASSERT_TRUE(f.has_value());
    EXPECT_EQ(f->dialog_type, "eaglet");
}

TEST_F(DispatchBundleTest, FactsReadArchitecture) {
    writeMetadata("phi_4_mini_instruct", /*vision=*/false);
    writeConfigJson("Phi3ForCausalLM");
    const auto f = dispatch_detail::bundleFactsOf(cfg());
    ASSERT_TRUE(f.has_value());
    EXPECT_EQ(f->architecture, "Phi3ForCausalLM");
}

TEST_F(DispatchBundleTest, FactsArchitectureEmptyWithoutConfigJson) {
    writeMetadata("gemma_4_e4b_it", /*vision=*/false);
    const auto f = dispatch_detail::bundleFactsOf(cfg());
    ASSERT_TRUE(f.has_value());
    EXPECT_EQ(f->architecture, "");
}

TEST_F(DispatchBundleTest, FactsFailOnMissingMetadata) {
    EXPECT_FALSE(dispatch_detail::bundleFactsOf(cfg()).has_value());
}

TEST_F(DispatchBundleTest, LLMPipelineRefusesVisionBundle) {
    writeMetadata("qwen3_vl_4b", /*vision=*/true);
    writeGenieConfig("basic");
    QnnRuntimeConfig runtime_cfg;
    EXPECT_FALSE(makeLLMPipeline(runtime_cfg, cfg()).has_value());
}

TEST_F(DispatchBundleTest, LLMPipelineRefusesNonBasicDialogType) {
    writeMetadata("qwen3_4b_eaglet", /*vision=*/false);
    writeGenieConfig("eaglet");
    QnnRuntimeConfig runtime_cfg;
    EXPECT_FALSE(makeLLMPipeline(runtime_cfg, cfg()).has_value());
}

TEST_F(DispatchBundleTest, LLMPipelineFailsOnUnreadableBundle) {
    QnnRuntimeConfig runtime_cfg;
    EXPECT_FALSE(makeLLMPipeline(runtime_cfg, cfg()).has_value());
}

TEST_F(DispatchBundleTest, VLMPipelineRefusesUnknownModelId) {
    writeMetadata("not_a_known_vlm_family", /*vision=*/true);
    writeGenieConfig("basic");
    QnnRuntimeConfig runtime_cfg;
    VLMConfig        config;
    config.llm_config = cfg();
    EXPECT_FALSE(makeVLMPipeline(runtime_cfg, config).has_value());
}

}  // namespace
}  // namespace geniex
