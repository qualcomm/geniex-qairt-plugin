// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause
#include <gtest/gtest.h>

#include <cmath>
#include <cstdlib>

#include "phonon/phonon_model.h"

TEST(PhononDevice, StreamingCancellationAndSessionReuse) {
    const char* bundle   = std::getenv("PHONON_TEST_BUNDLE");
    const char* soc      = std::getenv("PHONON_TEST_SOC");
    const char* frontend = std::getenv("PHONON_TEST_TEXT_LIBRARY");
    if (!bundle || !soc || !frontend)
        GTEST_SKIP() << "Set PHONON_TEST_BUNDLE, PHONON_TEST_SOC and PHONON_TEST_TEXT_LIBRARY";
    auto model = std::make_unique<geniex::PhononModel>(geniex::PhononOptions{bundle, soc, "en", frontend, true});
    geniex::QnnRuntimeConfig runtime;
    if (const char* sdk = std::getenv("PHONON_TEST_QAIRT")) runtime.htp_dir = sdk;
    ASSERT_TRUE(model->load(runtime));
    geniex::TTSPipeline  pipeline(std::move(model));
    geniex::SpeechConfig config;
    config.collect_audio = false;
    config.voice         = "not-a-bundle-voice";
    EXPECT_THROW(pipeline.synthesize("Hello", config), std::invalid_argument);
    ASSERT_TRUE(pipeline.isReady());
    config.voice.clear();
    size_t count = 0;
    auto   r     = pipeline.synthesize("A cancellation must not poison the next speech request.",
        config,
        [&](const float*, size_t, int) { return ++count < 2; });
    EXPECT_TRUE(r.cancelled);
    EXPECT_EQ(count, 2);
    count = 0;
    r     = pipeline.synthesize("An explicit cancel also stops generation.", config, [&](const float*, size_t, int) {
        ++count;
        pipeline.cancel();
        return true;
    });
    EXPECT_TRUE(r.cancelled);
    EXPECT_EQ(count, 1);
    for (int run = 0; run < 3; ++run) {
        uint64_t delivered = 0;
        r                  = pipeline.synthesize(
            "Hello from the shared GenieX QAIRT speech pipeline.", config, [&](const float* pcm, size_t n, int rate) {
                EXPECT_GT(rate, 0);
                for (size_t i = 0; i < n; ++i) EXPECT_TRUE(std::isfinite(pcm[i]));
                delivered += n;
                return true;
            });
        EXPECT_FALSE(r.cancelled);
        EXPECT_TRUE(r.audio.empty());
        EXPECT_EQ(r.samples, delivered);
        EXPECT_GT(delivered, 0);
        for (const auto& name : {"prefill", "step"}) {
            EXPECT_GT(r.graphs.at(name).calls, 0);
            EXPECT_GT(r.graphs.at(name).accelerator_ms, 0);
        }
    }
    EXPECT_THROW(pipeline.synthesize("A callback failure must invalidate this session.",
                     config,
                     [](const float*, size_t, int) -> bool { throw std::runtime_error("consumer failed"); }),
        std::runtime_error);
    EXPECT_FALSE(pipeline.isReady());
    EXPECT_THROW(pipeline.synthesize("Cannot reuse a failed session."), std::runtime_error);
}
