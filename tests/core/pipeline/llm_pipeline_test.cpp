// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause
//
// LLMPipeline behaviour over a scripted fake generator: no QNN backend or real
// model is involved.

#include "pipeline/llm_pipeline.h"

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "llm/llm_model.h"
#include "llm/token_generator.h"
#include "model.h"
#include "testing/llm_fixture.hpp"
#include "testing/stub_qnnapi.hpp"
#include "testing/testable_llm_model.hpp"

namespace geniex {
namespace {

namespace fs = std::filesystem;

// Minimal word-level tokenizer plus a trivial chat template, enough for the
// pipeline's create() to accept the bundle and for encode()/decode_token().
constexpr const char* kTokenizerJson = R"({
  "version": "1.0",
  "truncation": null,
  "padding": null,
  "added_tokens": [],
  "normalizer": null,
  "pre_tokenizer": {"type": "Whitespace"},
  "post_processor": null,
  "decoder": null,
  "model": {
    "type": "WordLevel",
    "vocab": {"hello": 0, "world": 1, "foo": 2, "bar": 3, "[UNK]": 4},
    "unk_token": "[UNK]"
  }
})";

constexpr const char* kTokenizerConfigJson =
    R"({"chat_template": "{% for m in messages %}{{ m['content'] }} {% endfor %}"})";

// Records every call and replays a scripted token stream.
class FakeGenerator : public Model, public TokenGenerator {
   public:
    struct Observed {
        int                  generate_calls = 0;
        int                  reset_calls    = 0;
        std::vector<int32_t> last_prompt;
        bool                 saw_tokenizer = false;
        std::string          saved_path;
        std::string          loaded_path;
        std::vector<int32_t> logits_tokens;
        bool                 logits_all_positions = false;
    };

    explicit FakeGenerator(Observed* obs) : obs_(obs) {}

    bool initialize(const QnnRuntimeConfig&, const ModelConfig&) override {
        if (throw_on_init) throw std::runtime_error("init boom");
        initialized_ = init_result;
        return init_result;
    }

    std::vector<int32_t> generate(const std::vector<int32_t>& prompt_tokens, const GenerationConfig& gen_cfg,
        std::function<bool(int32_t)> token_callback) override {
        ++obs_->generate_calls;
        obs_->last_prompt   = prompt_tokens;
        obs_->saw_tokenizer = gen_cfg.tokenizer != nullptr;
        if (throw_prompt_too_long) throw PromptTooLongError("too long");

        std::vector<int32_t> out;
        for (int32_t tok : script) {
            if (static_cast<int>(out.size()) >= gen_cfg.max_tokens) break;
            out.push_back(tok);
            n_past_ += 1;
            if (token_callback && !token_callback(tok)) break;
        }
        if (throw_context_after_script) throw ContextLengthExceededError("full");
        return out;
    }

    void               resetKVCache() override { ++obs_->reset_calls; }
    void               saveKVCacheToFile(const std::string& path) const override { obs_->saved_path = path; }
    void               loadKVCacheFromFile(const std::string& path) override { obs_->loaded_path = path; }
    size_t             nPast() const override { return n_past_; }
    size_t             vocabSize() const override { return 5; }
    std::vector<float> forwardLogits(const std::vector<int32_t>& tokens, bool all_positions) override {
        obs_->logits_tokens        = tokens;
        obs_->logits_all_positions = all_positions;
        return std::vector<float>(5, 0.5f);
    }

    std::vector<int32_t> script                     = {0, 1};
    bool                 init_result                = true;
    bool                 throw_on_init              = false;
    bool                 throw_prompt_too_long      = false;
    bool                 throw_context_after_script = false;

   private:
    Observed* obs_;
    size_t    n_past_ = 0;
};

class LLMPipelineTest : public ::testing::Test {
   protected:
    void SetUp() override {
        dir_ = fs::temp_directory_path() /
               ("geniex_llm_pipeline_test_" + std::to_string(::testing::UnitTest::GetInstance()->random_seed()) + "_" +
                   ::testing::UnitTest::GetInstance()->current_test_info()->name());
        fs::create_directories(dir_);
        std::ofstream(dir_ / "tokenizer.json") << kTokenizerJson;
        std::ofstream(dir_ / "tokenizer_config.json") << kTokenizerConfigJson;
        model_cfg_.tokenizer_path        = (dir_ / "tokenizer.json").string();
        model_cfg_.tokenizer_config_path = (dir_ / "tokenizer_config.json").string();
    }

    void TearDown() override {
        std::error_code ec;
        fs::remove_all(dir_, ec);
    }

    // Builds a pipeline over a fake generator; `fake` stays valid for tweaking
    // before create() and is owned by the pipeline afterwards.
    bool createWith(std::unique_ptr<FakeGenerator> fake) {
        return pipe_.create(std::move(fake), runtime_cfg_, model_cfg_);
    }

    std::unique_ptr<FakeGenerator> makeFake() { return std::make_unique<FakeGenerator>(&obs_); }

    static GenerationConfig cfg(int max_tokens = 16) {
        GenerationConfig c;
        c.max_tokens = max_tokens;
        return c;
    }

    fs::path                dir_;
    QnnRuntimeConfig        runtime_cfg_;
    ModelConfig             model_cfg_;
    LLMPipeline             pipe_;
    FakeGenerator::Observed obs_;
};

TEST_F(LLMPipelineTest, CreateSucceedsAndPipelineIsReady) {
    EXPECT_FALSE(pipe_.isReady());
    ASSERT_TRUE(createWith(makeFake()));
    EXPECT_TRUE(pipe_.isReady());
}

TEST_F(LLMPipelineTest, CreateFailsWhenInitializeReturnsFalse) {
    auto fake         = makeFake();
    fake->init_result = false;
    EXPECT_FALSE(createWith(std::move(fake)));
    EXPECT_FALSE(pipe_.isReady());
}

TEST_F(LLMPipelineTest, CreateFailsWhenInitializeThrows) {
    auto fake           = makeFake();
    fake->throw_on_init = true;
    EXPECT_FALSE(createWith(std::move(fake)));
    EXPECT_FALSE(pipe_.isReady());
}

TEST_F(LLMPipelineTest, CreateFailsWhenChatTemplateMissing) {
    std::ofstream(dir_ / "tokenizer_config.json", std::ios::trunc) << "{}";
    EXPECT_FALSE(createWith(makeFake()));
    EXPECT_FALSE(pipe_.isReady());
}

TEST_F(LLMPipelineTest, GenerateTokenIdsStreamsEveryEmittedToken) {
    auto fake    = makeFake();
    fake->script = {0, 1, 2};
    ASSERT_TRUE(createWith(std::move(fake)));

    int  pieces = 0;
    auto result = pipe_.generate(std::vector<int32_t>{3, 3}, cfg(), [&](const char*) {
        ++pieces;
        return true;
    });

    EXPECT_EQ(obs_.generate_calls, 1);
    EXPECT_EQ(obs_.last_prompt, (std::vector<int32_t>{3, 3}));
    // Ending on EOS counts the terminating sample as generated (Genie's convention).
    EXPECT_EQ(result.generated_tokens, 3 + 1);
    EXPECT_EQ(result.prompt_tokens, 2);
    EXPECT_EQ(pieces, 3);
    EXPECT_FALSE(result.full_text.empty());
    EXPECT_EQ(result.stop_reason, "eos");
}

TEST_F(LLMPipelineTest, GenerateTextEncodesPromptAndInjectsTokenizer) {
    ASSERT_TRUE(createWith(makeFake()));

    auto result = pipe_.generate(std::string("hello world"), cfg());

    EXPECT_EQ(obs_.last_prompt, (std::vector<int32_t>{0, 1}));
    EXPECT_TRUE(obs_.saw_tokenizer);
    EXPECT_EQ(result.stop_reason, "eos");
}

TEST_F(LLMPipelineTest, StopReasonIsLengthWhenMaxTokensReached) {
    auto fake    = makeFake();
    fake->script = {0, 1, 2, 3};
    ASSERT_TRUE(createWith(std::move(fake)));

    auto result = pipe_.generate(std::vector<int32_t>{3}, cfg(/*max_tokens=*/2));

    EXPECT_EQ(result.generated_tokens, 2);
    EXPECT_EQ(result.stop_reason, "length");
}

TEST_F(LLMPipelineTest, CallbackReturningFalseStopsWithUserReason) {
    auto fake    = makeFake();
    fake->script = {0, 1, 2, 3};
    ASSERT_TRUE(createWith(std::move(fake)));

    int  seen   = 0;
    auto result = pipe_.generate(std::vector<int32_t>{3}, cfg(), [&](const char*) { return ++seen < 2; });

    EXPECT_EQ(result.stop_reason, "user");
    EXPECT_EQ(seen, 2);
}

TEST_F(LLMPipelineTest, PromptTooLongMapsToStopReason) {
    auto fake                   = makeFake();
    fake->throw_prompt_too_long = true;
    ASSERT_TRUE(createWith(std::move(fake)));

    auto result = pipe_.generate(std::vector<int32_t>{3}, cfg());

    EXPECT_EQ(result.stop_reason, "prompt_too_long");
    EXPECT_EQ(result.generated_tokens, 0);
}

TEST_F(LLMPipelineTest, ContextOverflowKeepsPartialOutput) {
    auto fake                        = makeFake();
    fake->script                     = {0, 1};
    fake->throw_context_after_script = true;
    ASSERT_TRUE(createWith(std::move(fake)));

    auto result = pipe_.generate(std::vector<int32_t>{3}, cfg());

    EXPECT_EQ(result.stop_reason, "context_length");
    EXPECT_EQ(result.generated_tokens, 2);
    EXPECT_FALSE(result.full_text.empty());
}

TEST_F(LLMPipelineTest, GenerateBeforeCreateReportsError) {
    auto result = pipe_.generate(std::vector<int32_t>{1}, cfg());
    EXPECT_EQ(result.stop_reason, "error");
}

TEST_F(LLMPipelineTest, ResetAndAccessorsForwardToGenerator) {
    ASSERT_TRUE(createWith(makeFake()));
    EXPECT_EQ(pipe_.vocabSize(), 5u);
    EXPECT_EQ(pipe_.nPast(), 0u);

    pipe_.generate(std::vector<int32_t>{3}, cfg());
    EXPECT_EQ(pipe_.nPast(), 2u);

    pipe_.reset();
    EXPECT_EQ(obs_.reset_calls, 1);
}

TEST_F(LLMPipelineTest, FirstTurnBosIsPrependedOnlyWhenNothingIsCached) {
    ASSERT_TRUE(createWith(makeFake()));
    pipe_.setBosTokenId(4);
    EXPECT_EQ(pipe_.bosTokenId(), 4);

    pipe_.generate(std::string("hello"), cfg());
    EXPECT_EQ(obs_.last_prompt, (std::vector<int32_t>{4, 0}));

    // The fake now reports a non-zero nPast, so the next turn must not add BOS.
    pipe_.generate(std::string("world"), cfg());
    EXPECT_EQ(obs_.last_prompt, (std::vector<int32_t>{1}));
}

// ── embedding input is only provided by an LLMModel ───────────────────────────────────

TEST_F(LLMPipelineTest, NonLlmGeneratorHasNoEmbeddingInput) {
    ASSERT_TRUE(createWith(makeFake()));
    EXPECT_FALSE(pipe_.supportsInputEmbeddings());
    EXPECT_EQ(pipe_.hiddenSize(), 0u);
}

TEST_F(LLMPipelineTest, ForwardLogitsIsForwardedToTheGenerator) {
    ASSERT_TRUE(createWith(makeFake()));
    const auto logits = pipe_.forwardLogits({0, 1}, /*all_positions=*/true);
    EXPECT_EQ(logits.size(), 5u);
    EXPECT_EQ(obs_.logits_tokens, (std::vector<int32_t>{0, 1}));
    EXPECT_TRUE(obs_.logits_all_positions);
}

TEST_F(LLMPipelineTest, ForwardLogitsBeforeCreateThrows) {
    EXPECT_THROW(pipe_.forwardLogits({0, 1}), std::runtime_error);
}

TEST_F(LLMPipelineTest, KVPersistenceIsForwardedToTheGenerator) {
    ASSERT_TRUE(createWith(makeFake()));
    pipe_.saveKVCache("save.bin");
    pipe_.loadKVCache("load.bin");
    EXPECT_EQ(obs_.saved_path, "save.bin");
    EXPECT_EQ(obs_.loaded_path, "load.bin");
}

TEST_F(LLMPipelineTest, NonLlmGeneratorRejectsEmbeddingInput) {
    ASSERT_TRUE(createWith(makeFake()));
    auto result = pipe_.generateFromEmbeddings(std::vector<float>(8, 0.0f), cfg());
    EXPECT_EQ(result.stop_reason, "error");
    EXPECT_EQ(obs_.generate_calls, 0);
}

TEST_F(LLMPipelineTest, KVPersistenceBeforeCreateIsANoOp) {
    EXPECT_NO_THROW(pipe_.saveKVCache("x.bin"));
    EXPECT_NO_THROW(pipe_.loadKVCache("x.bin"));
}

// ── LLMModel-backed pipeline ─────────────────────────────────────────────────

using testing::LLMFixture;

// An LLMModel over CPU graph fixtures whose initialize() loads the fixture
// graphs and an embedding table instead of bringing up a QNN backend.
class FixtureLLM : public testing::TestableLLMModel {
   public:
    FixtureLLM(LLMFixture& fx, std::string table_path)
        : testing::TestableLLMModel(LLMFixture::makeSpec()), fx_(fx), table_path_(std::move(table_path)) {}

    bool initialize(const QnnRuntimeConfig&, const ModelConfig&) override {
        if (!initFromFixture(fx_)) return false;
        findEmbeddingProvider("input_embeds")->loadTable(table_path_, LLMFixture::kVocab, LLMFixture::kHidden);
        return true;
    }

   private:
    LLMFixture& fx_;
    std::string table_path_;
};

class LLMBackedPipelineTest : public LLMPipelineTest {
   protected:
    void SetUp() override {
        LLMPipelineTest::SetUp();
        std::vector<float> table(LLMFixture::kVocab * LLMFixture::kHidden);
        for (size_t r = 0; r < LLMFixture::kVocab; ++r)
            for (size_t c = 0; c < LLMFixture::kHidden; ++c) table[r * LLMFixture::kHidden + c] = static_cast<float>(r);
        table_path_ = (dir_ / "embedding_table.bin").string();
        std::ofstream(table_path_, std::ios::binary)
            .write(reinterpret_cast<const char*>(table.data()), static_cast<std::streamsize>(table.size() * 4));
        testing::stubSetVocabSize(LLMFixture::kVocab);
        testing::stubSetNextToken(2);
    }

    void TearDown() override {
        // The model points into fx_, so it must go first.
        pipe_ = LLMPipeline();
        testing::stubSetNextToken(-1);
        testing::stubSetVocabSize(0);
        LLMPipelineTest::TearDown();
    }

    bool createLLM() { return pipe_.create(std::make_unique<FixtureLLM>(fx_, table_path_), runtime_cfg_, model_cfg_); }

    static GenerationConfig greedy(int max_tokens) {
        GenerationConfig c;
        c.enable_sampling = false;
        c.max_tokens      = max_tokens;
        return c;
    }

    testing::NoDecodePoolEnv no_pool_;
    LLMFixture               fx_;
    std::string              table_path_;
};

TEST_F(LLMBackedPipelineTest, LLMModelEnablesEmbeddingInput) {
    ASSERT_TRUE(createLLM());
    EXPECT_TRUE(pipe_.supportsInputEmbeddings());
    EXPECT_EQ(pipe_.hiddenSize(), size_t{LLMFixture::kHidden});
    EXPECT_EQ(pipe_.vocabSize(), size_t{LLMFixture::kVocab});
}

TEST_F(LLMBackedPipelineTest, GenerateRetainsKVAcrossTurns) {
    ASSERT_TRUE(createLLM());
    auto first = pipe_.generate(std::vector<int32_t>{0, 1}, greedy(1));
    EXPECT_EQ(first.generated_tokens, 1);
    const size_t after_first = pipe_.nPast();
    EXPECT_GT(after_first, 0u);

    pipe_.generate(std::vector<int32_t>{3}, greedy(1));
    EXPECT_GT(pipe_.nPast(), after_first);

    pipe_.reset();
    EXPECT_EQ(pipe_.nPast(), 0u);
}

TEST_F(LLMBackedPipelineTest, ForwardLogitsReturnsOneVocabRow) {
    ASSERT_TRUE(createLLM());
    EXPECT_EQ(pipe_.forwardLogits({1, 2, 3}).size(), size_t{LLMFixture::kVocab});
    EXPECT_EQ(pipe_.forwardLogits({1, 2, 3}, /*all_positions=*/true).size(), 3 * size_t{LLMFixture::kVocab});
}

TEST_F(LLMBackedPipelineTest, GenerateFromEmbeddingsDecodesAfterSuppliedRows) {
    ASSERT_TRUE(createLLM());
    const std::vector<float> rows(2 * LLMFixture::kHidden, 1.0f);
    auto                     result = pipe_.generateFromEmbeddings(rows, greedy(2));
    EXPECT_EQ(result.stop_reason, "length");
    EXPECT_EQ(result.generated_tokens, 2);
    EXPECT_EQ(result.prompt_tokens, 2);
}

TEST_F(LLMBackedPipelineTest, GenerateFromEmbeddingsRejectsPartialRow) {
    ASSERT_TRUE(createLLM());
    EXPECT_THROW(pipe_.generateFromEmbeddings(std::vector<float>(LLMFixture::kHidden + 1, 1.0f), greedy(1)),
        std::invalid_argument);
    EXPECT_THROW(pipe_.generateFromEmbeddings({}, greedy(1)), std::invalid_argument);
}

TEST_F(LLMBackedPipelineTest, KVCacheRoundTripsThroughFile) {
    ASSERT_TRUE(createLLM());
    pipe_.generate(std::vector<int32_t>{0, 1}, greedy(1));
    const size_t held = pipe_.nPast();
    ASSERT_GT(held, 0u);

    const std::string path = (dir_ / "kv.bin").string();
    pipe_.saveKVCache(path);
    pipe_.reset();
    ASSERT_EQ(pipe_.nPast(), 0u);
    pipe_.loadKVCache(path);
    EXPECT_EQ(pipe_.nPast(), held);
}

}  // namespace
}  // namespace geniex
