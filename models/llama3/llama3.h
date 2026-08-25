// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "llm/input_provider.h"
#include "llm/llm_model.h"
#include "llm/llm_spec_loader.h"
#include "llm/llm_types.h"
#include "logging.h"
#include "pipeline/llm_pipeline.h"

namespace geniex {
namespace llama3 {

inline LLMModel makeModel(const ModelConfig& model_cfg) {
    auto gc   = parseGenieConfig(bundleDirOf(model_cfg));
    auto spec = buildSpecSkeleton(gc);  // must read gc before it's moved-from below
    return LLMModel(std::move(spec), std::move(gc));
}

inline std::optional<LLMPipeline> makePipeline(const QnnRuntimeConfig& runtime_cfg, const ModelConfig& model_cfg) {
    try {
        LLMPipeline pipe;
        if (!pipe.create(makeModel(model_cfg), runtime_cfg, model_cfg)) return std::nullopt;
        return pipe;
    } catch (const std::exception& e) {
        GENIEX_LOG_ERROR("llama3::makePipeline failed: {}", e.what());
        return std::nullopt;
    }
}

}  // namespace llama3
}  // namespace geniex
