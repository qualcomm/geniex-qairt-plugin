// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <cstdint>
#include <exception>
#include <memory>
#include <vector>

#include "geniex-proc/sampler.h"
#include "logging.h"
#include "types.h"

namespace geniex {

// True when two configs agree on every field that affects the sampler chain.
inline bool samplerConfigEqual(const GenerationConfig& a, const GenerationConfig& b) {
    return a.enable_sampling == b.enable_sampling && a.temperature == b.temperature && a.top_p == b.top_p &&
           a.min_p == b.min_p && a.top_k == b.top_k && a.repetition_penalty == b.repetition_penalty &&
           a.presence_penalty == b.presence_penalty && a.frequency_penalty == b.frequency_penalty &&
           a.penalty_last_n == b.penalty_last_n && a.seed == b.seed && a.grammar_str == b.grammar_str &&
           a.grammar_root == b.grammar_root;
}

// Builds the geniex-proc sampler chain for `gen_cfg`. `eog_tokens` are the model's
// end-of-generation ids so Sampler::is_eog() works.
inline std::unique_ptr<Sampler> makeSampler(const GenerationConfig& gen_cfg, const std::vector<int32_t>& eog_tokens) {
    geniex_sampler_params sp;
    sp.seed            = gen_cfg.seed;
    sp.temp            = gen_cfg.temperature;
    sp.top_k           = gen_cfg.top_k;
    sp.top_p           = gen_cfg.top_p;
    sp.min_p           = gen_cfg.min_p;
    sp.penalty_repeat  = gen_cfg.repetition_penalty;
    sp.penalty_freq    = gen_cfg.frequency_penalty;
    sp.penalty_present = gen_cfg.presence_penalty;
    sp.penalty_last_n  = gen_cfg.penalty_last_n;
    sp.no_perf         = true;
    sp.eog_tokens.assign(eog_tokens.begin(), eog_tokens.end());

    auto sampler = std::make_unique<Sampler>(sp);

    // Grammar needs a tokenizer (for the vocab interface). The pipeline injects
    // one via gen_cfg.tokenizer; if missing, warn and skip.
    if (!gen_cfg.grammar_str.empty()) {
        if (gen_cfg.tokenizer) {
            try {
                sampler->set_grammar(
                    std::make_unique<Grammar>(gen_cfg.grammar_str, *gen_cfg.tokenizer, gen_cfg.grammar_root));
            } catch (const std::exception& e) {
                GENIEX_LOG_WARN("grammar init failed, continuing without grammar: {}", e.what());
            }
        } else {
            GENIEX_LOG_WARN("grammar_str set but no tokenizer provided — grammar disabled");
        }
    }
    return sampler;
}

}  // namespace geniex
