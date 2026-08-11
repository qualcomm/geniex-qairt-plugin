// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

// Device-only EAGLE bring-up: registers the quantized embedding providers and
// drives the target/draft engines' QNN initialize(). Separated from
// eagle_model.cpp so that file's CPU-reachable decode logic can be measured by
// the coverage surface without pulling in QNN backend load. Mirrors the
// model.cpp / model_init.cpp split.

#include <memory>
#include <stdexcept>
#include <string>

#include "llm/eagle_model.h"
#include "llm/input_provider.h"
#include "logging.h"

namespace geniex {

namespace {

// Binds a quantized embedding provider to `embed_name`, replacing the default
// provider (which assumes an unquantized table). Registered before initialize()
// so createInputProviders() leaves it in place. An explicit table path overrides
// ModelConfig::embedding_path -- the draft needs its own embedding weights.
void registerQuantizedEmbedding(
    LLMModel& m, const std::string& embed_name, const EagleConfig& cfg, const std::string& table_path) {
    auto provider = table_path.empty() ? std::make_unique<EmbeddingInputProvider>(embed_name)
                                       : std::make_unique<EmbeddingInputProvider>(embed_name,
                                             table_path,
                                             /*row_hidden_size=*/0,
                                             /*pad_token_override=*/-1);
    provider->setQuantization(cfg.embedding_quant);
    m.addInputProvider(std::move(provider));
}

}  // namespace

bool EagleModel::initialize(const QnnRuntimeConfig& runtime_cfg, const ModelConfig& target_cfg) {
    // The draft's shard "state" output is its hidden feature, not a vocab head,
    // so the logits tensor must be named explicitly or argmax would run over the
    // hidden state. This is a required value.
    if (cfg_.draft_logits_name.empty())
        throw std::runtime_error("EagleModel: EagleConfig::draft_logits_name is required");

    // A shared RoPE base is fed raw to both engines' rotary embedding; 0 (the
    // "unset" sentinel) yields inf/nan rotations, so the bundle parser must have
    // supplied it and proven target/draft agree.
    if (!(cfg_.rope_theta > 0.0f))
        throw std::runtime_error(
            "EagleModel: EagleConfig::rope_theta must be > 0 (got " + std::to_string(cfg_.rope_theta) + ")");

    registerQuantizedEmbedding(target(), cfg_.target_embed_name, cfg_, /*table_path=*/"");
    if (!target().initialize(runtime_cfg, target_cfg)) {
        GENIEX_LOG_ERROR("EagleModel: target engine initialize() failed");
        return false;
    }

    ModelConfig draft_cfg = target_cfg;
    draft_cfg.model_paths = cfg_.draft_model_paths;
    registerQuantizedEmbedding(draft(), cfg_.draft_embed_name, cfg_, cfg_.draft_embedding_path);
    if (!draft().initialize(runtime_cfg, draft_cfg)) {
        GENIEX_LOG_ERROR("EagleModel: draft engine initialize() failed");
        return false;
    }

    // Multi-CL bundles are safe but unsupported: the speculation loop never
    // promotes, so a long prompt throws ContextLengthExceededError mid-generation
    // rather than growing the context. Nothing rejects this at load, so surface
    // it here to set expectations at the right time instead of on a long prompt.
    if (target().spec().context_lengths.size() > 1 || draft().spec().context_lengths.size() > 1) {
        GENIEX_LOG_WARN(
            "EagleModel: multi-CL bundle (target {} CL, draft {} CL); EAGLE speculation does not promote, so "
            "generation throws once the initial context length is exhausted",
            target().spec().context_lengths.size(),
            draft().spec().context_lengths.size());
    }

    ready_       = true;
    initialized_ = true;
    return true;
}

}  // namespace geniex
