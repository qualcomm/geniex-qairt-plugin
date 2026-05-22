// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "llm/input_provider.h"
#include "llm/llm_model.h"
#include "llm/llm_spec_loader.h"
#include "llm/llm_types.h"
#include "logging.h"
#include "pipeline/chat_template.h"
#include "pipeline/llm_pipeline.h"

namespace geniex {
namespace qwen2_5 {

inline LLMModel makeModel(const ModelConfig& model_cfg) {
    const auto bundle = bundleDirOf(model_cfg);
    auto       meta   = parseQAIRTMetadata(bundle);
    auto       gc     = parseGenieConfig(bundle);

    LLMModel m(buildSpec(meta, gc));
    m.addInputProvider(makeEmbeddingProvider(meta, gc));
    m.addInputProvider(makeRoPEProvider(meta, gc));
    return m;
}

inline std::optional<LLMPipeline> makePipeline(const QnnRuntimeConfig& runtime_cfg, const ModelConfig& model_cfg) {
    try {
        LLMPipeline pipe;
        if (!pipe.create(chatMLTemplate, makeModel(model_cfg), runtime_cfg, model_cfg)) return std::nullopt;
        return pipe;
    } catch (const std::exception& e) {
        GENIEX_LOG_ERROR("qwen2_5::makePipeline failed: {}", e.what());
        return std::nullopt;
    }
}

}  // namespace qwen2_5
}  // namespace geniex
