// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "geniex_export.h"
#include "llm/llm_model.h"
#include "types.h"

namespace geniex {

// Thrown by an LLMBackend for an operation its decoding strategy cannot offer
// (e.g. KV save/load on a two-engine speculative decoder).
class GENIEX_API UnsupportedOperationError : public std::runtime_error {
   public:
    using std::runtime_error::runtime_error;
};

// The decoding engine behind an LLMPipeline: everything the pipeline needs from
// a model once the tokenizer, chat template and streaming are factored out.
// A single `LLMModel` is the common case (LLMModelBackend); a multi-engine
// decoder that is not an `LLMModel` (EAGLE: target + draft) implements this
// directly.
class GENIEX_API LLMBackend {
   public:
    virtual ~LLMBackend() = default;

    // Returns generated token IDs (excluding the prompt). token_callback is
    // called with each emitted token; return false to stop early.
    virtual std::vector<int32_t> generate(const std::vector<int32_t>& prompt_tokens, const GenerationConfig& gen_cfg,
        std::function<bool(int32_t)> token_callback) = 0;

    virtual void   reset()           = 0;
    virtual size_t nPast() const     = 0;
    virtual size_t vocabSize() const = 0;

    // Optional capabilities; the defaults describe a backend without them.
    virtual bool   supportsInputEmbeddings() { return false; }
    virtual size_t hiddenSize() const { return 0; }

    virtual std::vector<int32_t> generateFromEmbeddings(
        std::vector<float>, const GenerationConfig&, std::function<bool(int32_t)>) {
        throw UnsupportedOperationError("input embeddings are not supported by this backend");
    }
    virtual std::vector<float> forwardLogits(const std::vector<int32_t>&, bool) {
        throw UnsupportedOperationError("forwardLogits is not supported by this backend");
    }
    virtual void saveKVCache(const std::string&) const {
        throw UnsupportedOperationError("KV cache save is not supported by this backend");
    }
    virtual void loadKVCache(const std::string&) {
        throw UnsupportedOperationError("KV cache load is not supported by this backend");
    }
};

// Adapter for the single-engine case: forwards to an owned LLMModel (or subclass).
class GENIEX_API LLMModelBackend final : public LLMBackend {
   public:
    explicit LLMModelBackend(std::unique_ptr<LLMModel> model) : model_(std::move(model)) {}

    LLMModel& model() { return *model_; }

    std::vector<int32_t> generate(const std::vector<int32_t>& prompt_tokens, const GenerationConfig& gen_cfg,
        std::function<bool(int32_t)> token_callback) override {
        return model_->generate(prompt_tokens, gen_cfg, std::move(token_callback));
    }
    void   reset() override { model_->resetKVCache(); }
    size_t nPast() const override { return model_->nPast(); }
    size_t vocabSize() const override { return model_->vocabSize(); }

    bool   supportsInputEmbeddings() override { return model_->supportsInputEmbeddings(); }
    size_t hiddenSize() const override { return model_->hiddenSize(); }

    std::vector<int32_t> generateFromEmbeddings(std::vector<float> input_embd, const GenerationConfig& gen_cfg,
        std::function<bool(int32_t)> token_callback) override {
        return model_->generateFromEmbeddings(std::move(input_embd), gen_cfg, std::move(token_callback));
    }
    std::vector<float> forwardLogits(const std::vector<int32_t>& input_ids, bool all_positions) override {
        return model_->forwardLogits(input_ids, all_positions);
    }
    void saveKVCache(const std::string& path) const override { model_->saveKVCacheToFile(path); }
    void loadKVCache(const std::string& path) override { model_->loadKVCacheFromFile(path); }

   private:
    std::unique_ptr<LLMModel> model_;
};

}  // namespace geniex
