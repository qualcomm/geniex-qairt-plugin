// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause
#include "pipeline/tts_pipeline.h"

#include <gtest/gtest.h>

namespace {
class FakeSpeech final : public geniex::SpeechGenerator {
   public:
    bool                 ready = true;
    bool                 isReady() const override { return ready; }
    geniex::SpeechResult synthesize(const std::string& text, const geniex::SpeechConfig& config,
        const geniex::AudioCallback& audio, const geniex::StopCallback& stop) override {
        if (text == "throw") throw std::runtime_error("generator error");
        geniex::SpeechResult result;
        result.sample_rate = 24000;
        const float pcm[2] = {0.2f, -0.3f};
        for (int i = 0; i < 3; ++i) {
            if (stop() || (audio && !audio(pcm, 2, result.sample_rate))) {
                result.cancelled = true;
                break;
            }
            result.samples += 2;
            if (config.collect_audio) result.audio.insert(result.audio.end(), pcm, pcm + 2);
        }
        return result;
    }
};
TEST(TTSPipeline, RejectsNullAndUninitializedGenerators) {
    EXPECT_THROW(geniex::TTSPipeline(nullptr), std::invalid_argument);
    auto fake   = std::make_unique<FakeSpeech>();
    fake->ready = false;
    EXPECT_THROW(geniex::TTSPipeline(std::move(fake)), std::invalid_argument);
}
TEST(TTSPipeline, StreamsCollectsAndCanDiscardSamples) {
    geniex::TTSPipeline pipeline(std::make_unique<FakeSpeech>());
    EXPECT_TRUE(pipeline.isReady());
    size_t n = 0;
    auto   r = pipeline.synthesize("hello", {}, [&](const float* pcm, size_t count, int rate) {
        EXPECT_EQ(rate, 24000);
        EXPECT_FLOAT_EQ(pcm[0], 0.2f);
        n += count;
        return true;
    });
    EXPECT_EQ(n, 6);
    EXPECT_EQ(r.audio.size(), 6);
    EXPECT_FALSE(r.cancelled);
    geniex::SpeechConfig config;
    config.collect_audio = false;
    EXPECT_TRUE(pipeline.synthesize("hello", config).audio.empty());
}
TEST(TTSPipeline, RejectsEmptyInputAndReentrantRequests) {
    geniex::TTSPipeline pipeline(std::make_unique<FakeSpeech>());
    EXPECT_THROW(pipeline.synthesize(""), std::invalid_argument);
    pipeline.synthesize("hello", {}, [&](const float*, size_t, int) {
        EXPECT_THROW(pipeline.synthesize("overlap"), std::runtime_error);
        return true;
    });
}
TEST(TTSPipeline, CancellationAndCallbackStopDoNotPoisonNextRequest) {
    geniex::TTSPipeline pipeline(std::make_unique<FakeSpeech>());
    auto                r = pipeline.synthesize("hello", {}, [&](const float*, size_t, int) {
        pipeline.cancel();
        return true;
    });
    EXPECT_TRUE(r.cancelled);
    EXPECT_TRUE(pipeline.synthesize("hello", {}, [](const float*, size_t, int) { return false; }).cancelled);
    EXPECT_FALSE(pipeline.synthesize("again").cancelled);
}
TEST(TTSPipeline, ExceptionsReleaseRequestLock) {
    geniex::TTSPipeline pipeline(std::make_unique<FakeSpeech>());
    EXPECT_THROW(pipeline.synthesize("throw"), std::runtime_error);
    EXPECT_THROW(pipeline.synthesize(
                     "hello", {}, [](const float*, size_t, int) -> bool { throw std::runtime_error("callback"); }),
        std::runtime_error);
    EXPECT_FALSE(pipeline.synthesize("again").cancelled);
}
}  // namespace
