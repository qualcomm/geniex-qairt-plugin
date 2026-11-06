// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <memory>
#include <string>

#include "geniex_export.h"
#include "model.h"
#include "pipeline/tts_pipeline.h"

namespace geniex {
struct PhononOptions {
    std::string bundle_dir;
    std::string soc_model;
    std::string language;
    // Explicit trusted native library; never load executable code named by bundle metadata.
    std::string text_library;
    bool        verify_npu = true;
};

// Phonon's acoustic loop runs through Model/Graph and the shared QnnApi backend.
// No second QNN engine or backend loader is used.
class GENIEX_TTS_API PhononModel final : public Model, public SpeechGenerator {
   public:
    explicit PhononModel(const PhononOptions& options);
    ~PhononModel() override;
    bool                     load(const QnnRuntimeConfig& runtime = {});
    bool                     isReady() const override;
    std::vector<std::string> voices() const;
    SpeechResult synthesize(const std::string& text, const SpeechConfig& config, const AudioCallback& on_audio,
        const StopCallback& should_stop) override;

   protected:
    bool onInitialized() override;

   private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}  // namespace geniex
