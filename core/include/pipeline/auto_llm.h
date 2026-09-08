// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <optional>
#include <string>

#include "llm/llm_model.h"
#include "llm/llm_spec_loader.h"
#include "logging.h"
#include "pipeline/llm_pipeline.h"
#include "types.h"

// Generic factory for plain decoder-only LLM families: resolves purely from
// config.json's architectures[0] + genie_config.json + the compiled graphs,
// no per-family class. Per-family behaviour lives in models/dispatch.h.
//
// Families that override runtime behaviour keep their own factory: Gemma3/4
// (per-layer embeddings), SSD/EAGLE, the VLM families.

namespace geniex {
namespace auto_llm {

struct Options {
    // Not derivable from the tokenizer config; the caller resolves this from
    // config.json's architectures[0] -- see models/dispatch.h.
    bool prepend_bos = false;
};

inline LLMModel makeModel(const ModelConfig& model_cfg) {
    auto gc   = parseGenieConfig(bundleDirOf(model_cfg));
    auto spec = buildSpecSkeleton(gc);  // read gc before it's moved-from below
    return LLMModel(std::move(spec), std::move(gc));
}

inline std::optional<LLMPipeline> makePipeline(
    const QnnRuntimeConfig& runtime_cfg, const ModelConfig& model_cfg, const Options& opts = {}) {
    std::string bundle_label;
    try {
        const auto bundle = bundleDirOf(model_cfg);
        bundle_label      = bundle.string();

        // Parsed here rather than via makeModel() so bos_token_id survives gc's move.
        auto          gc   = parseGenieConfig(bundle);
        const int32_t bos  = gc.bos_token_id;
        auto          spec = buildSpecSkeleton(gc);

        LLMPipeline pipe;
        if (!pipe.create(LLMModel(std::move(spec), std::move(gc)), runtime_cfg, model_cfg)) return std::nullopt;
        if (opts.prepend_bos) pipe.setBosTokenId(bos);
        return pipe;
    } catch (const std::exception& e) {
        GENIEX_LOG_ERROR("auto_llm::makePipeline failed for bundle '{}': {}", bundle_label, e.what());
        return std::nullopt;
    }
}

}  // namespace auto_llm
}  // namespace geniex
