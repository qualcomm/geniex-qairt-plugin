// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

namespace geniex {
using AudioCallback = std::function<bool(const float*, size_t, int)>;
using StopCallback  = std::function<bool()>;
struct SpeechConfig {
    std::string voice;
    uint32_t    seed          = 0;
    float       temperature   = -1.0f;  // Negative uses the bundle's default.
    bool        collect_audio = true;
};
struct SpeechGraphTiming {
    uint64_t calls          = 0;
    double   accelerator_ms = 0;
};
struct SpeechResult {
    std::vector<float>                       audio;
    int                                      sample_rate    = 0;
    uint64_t                                 samples        = 0;
    uint32_t                                 chunks         = 0;
    bool                                     cancelled      = false;
    double                                   first_audio_ms = -1;
    double                                   total_ms       = 0;
    std::map<std::string, SpeechGraphTiming> graphs;
};
// Model-specific acoustic generation is injected, like TokenGenerator for text.
// Implementations own their model initialization and return mono float PCM.
class SpeechGenerator {
   public:
    virtual ~SpeechGenerator()           = default;
    virtual bool         isReady() const = 0;
    virtual SpeechResult synthesize(const std::string& text, const SpeechConfig& config, const AudioCallback& on_audio,
        const StopCallback& should_stop) = 0;
};
// One request per pipeline; cancellation never waits for the request mutex.
class TTSPipeline {
   public:
    explicit TTSPipeline(std::unique_ptr<SpeechGenerator> generator) : generator_(std::move(generator)) {
        if (!generator_ || !generator_->isReady()) throw std::invalid_argument("TTS generator is not ready");
    }
    bool         isReady() const { return generator_->isReady(); }
    void         cancel() { cancelled_.store(true); }
    SpeechResult synthesize(const std::string& text, const SpeechConfig& config = {}, AudioCallback on_audio = {}) {
        std::unique_lock<std::mutex> lock(mutex_, std::try_to_lock);
        if (!lock.owns_lock()) throw std::runtime_error("TTS pipeline is busy");
        if (text.empty()) throw std::invalid_argument("TTS text must not be empty");
        cancelled_.store(false);
        return generator_->synthesize(text, config, on_audio, [this] { return cancelled_.load(); });
    }

   private:
    std::unique_ptr<SpeechGenerator> generator_;
    std::atomic<bool>                cancelled_{false};
    std::mutex                       mutex_;
};
}  // namespace geniex
