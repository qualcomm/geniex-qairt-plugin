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

// Generic factory for plain decoder-only LLM families.
//
// Named after transformers.AutoModelForCausalLM: like it, this resolves purely
// from the bundle's own config -- architectures[0] in config.json, everything
// else from genie_config.json and the compiled graphs -- with no per-family
// class of its own. This replaces the per-family makeModel/makePipeline headers
// that Llama-3, Qwen2.5, Qwen3, Falcon3, Phi-3.5 and Phi-4 each carried as
// identical copies; per-family behaviour is now declared at the call site
// (see models/dispatch.h) rather than restated in a header per family.
//
// Not to be confused with examples/auto_llm, an unrelated example predating
// this file that happens to also be architecture-agnostic; the two do not
// share code or a namespace.
//
// Families that override runtime behaviour still need their own factory:
// Gemma3/4 (subclasses LLMModel for the per-layer embedding stream), the SSD
// and EAGLE speculative variants, and the VLM families.
//
// Lives in pipeline/ rather than llm/ because it returns an LLMPipeline, and
// core/include/llm/ sits below core/include/pipeline/. Kept header-only inline,
// exactly like the per-family factories it replaces, so geniex_core's exported
// ABI is unchanged.

namespace geniex {
namespace auto_model {

struct Options {
    // Prepend genie_config.json's bos-token on the first turn
    // (LLMPipeline::setBosTokenId).
    //
    // Off by default: most chat templates already emit their own opening
    // special token, and a second BOS shifts every position by one. Turn it on
    // only for families whose model needs a leading BOS that the template does
    // not supply -- Qwen3 and Gemma do, Llama-3 / Qwen2.5 / Falcon3 / Phi do
    // not. It cannot be derived from the bundle's tokenizer config (Tokenizer
    // exposes no add_bos_token accessor), so the caller resolves it from
    // config.json's architectures[0] -- see models/dispatch.h.
    bool prepend_bos = false;
};

// Builds an uninitialized LLMModel from the bundle's genie_config.json. The
// caller owns initialize(); use makePipeline() for the tokenizer + chat-template
// path.
inline LLMModel makeModel(const ModelConfig& model_cfg) {
    auto gc   = parseGenieConfig(bundleDirOf(model_cfg));
    auto spec = buildSpecSkeleton(gc);  // must read gc before it's moved-from below
    return LLMModel(std::move(spec), std::move(gc));
}

// Builds and initializes a full pipeline. Returns std::nullopt on failure,
// logging the bundle that failed.
inline std::optional<LLMPipeline> makePipeline(
    const QnnRuntimeConfig& runtime_cfg, const ModelConfig& model_cfg, const Options& opts = {}) {
    std::string bundle_label;
    try {
        const auto bundle = bundleDirOf(model_cfg);
        bundle_label      = bundle.string();

        // Parsed once here rather than via makeModel(), so bos-token is still
        // readable after gc is moved into the model.
        auto          gc   = parseGenieConfig(bundle);
        const int32_t bos  = gc.bos_token_id;
        auto          spec = buildSpecSkeleton(gc);

        LLMPipeline pipe;
        if (!pipe.create(LLMModel(std::move(spec), std::move(gc)), runtime_cfg, model_cfg)) return std::nullopt;
        if (opts.prepend_bos) pipe.setBosTokenId(bos);
        return pipe;
    } catch (const std::exception& e) {
        GENIEX_LOG_ERROR("auto_model::makePipeline failed for bundle '{}': {}", bundle_label, e.what());
        return std::nullopt;
    }
}

}  // namespace auto_model
}  // namespace geniex
