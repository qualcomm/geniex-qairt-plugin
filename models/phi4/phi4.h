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
namespace phi4 {

// Family factory for Phi-4 models. A Phi-3 decoder (`Phi3ForCausalLM`), so the
// generic spec + provider path covers it unchanged.
//
// RoPE width comes from the cos tensor: partial_rotary_factor 0.75 over
// head_dim 128 exports position_ids_cos/sin as [1,1,1,48] → 96 rotated dims.
// genie_config.json ships no rope-scaling block (the HF config's LongRoPE
// short_factor is all-ones at CL 4096), so standard RoPE is correct.
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
        GENIEX_LOG_ERROR("phi4::makePipeline failed: {}", e.what());
        return std::nullopt;
    }
}

}  // namespace phi4
}  // namespace geniex
