// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "llama3/llama3.h"
#include "llm/input_provider.h"
#include "llm/llm_spec_loader.h"
#include "llm/llm_types.h"
#include "logging.h"
#include "pipeline/llm_pipeline.h"
#include "ssd_model.h"
#include "ssd_types.h"

namespace geniex {
namespace llama3_2_3b_ssd {

// SSD-specific runtime constants (not in any bundle file today).
inline constexpr int kForecastPrefix = 16;

inline SSDConfig makeSSDConfig(const std::string& forecast_prefix_path, float rope_theta) {
    return SSDConfig{
        .branches             = {3, 2},
        .forecast_prefix      = kForecastPrefix,
        .forecast_prefix_path = forecast_prefix_path,
        .rope_theta           = rope_theta,
    };
}

inline SSDModel makeModel(const ModelConfig& model_cfg) {
    auto gc   = parseGenieConfig(bundleDirOf(model_cfg));
    auto spec = buildSpecSkeleton(gc);  // must read gc before makeSSDConfig below

    const std::string forecast_prefix_path = model_cfg.forecast_prefix_path.value_or("");
    return SSDModel(std::move(spec), makeSSDConfig(forecast_prefix_path, gc.rope_theta));
}

inline std::optional<LLMPipeline> makePipeline(const QnnRuntimeConfig& runtime_cfg, const ModelConfig& model_cfg) {
    try {
        LLMPipeline pipe;
        if (!pipe.create(makeModel(model_cfg), runtime_cfg, model_cfg)) return std::nullopt;
        return pipe;
    } catch (const std::exception& e) {
        GENIEX_LOG_ERROR("llama3_2_3b_ssd::makePipeline failed: {}", e.what());
        return std::nullopt;
    }
}

}  // namespace llama3_2_3b_ssd
}  // namespace geniex
