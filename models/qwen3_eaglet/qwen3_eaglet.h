// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "llm/eagle_model.h"
#include "llm/eagle_types.h"
#include "llm/llm_spec_loader.h"
#include "llm/llm_types.h"
#include "logging.h"
#include "types.h"
#include "utils/detail/json.hpp"

namespace geniex {
namespace qwen3_eaglet {

using json = qualla::json;

// Reads genie_config.json for the eaglet-specific fields the two-engine driver
// needs. Complements parseGenieConfig() (which already yields the embedding
// quant spec and EOS/BOS tokens) with the draft engine's paths, the trimmed
// draft-token map, and the shared RoPE base. The graph tensor bindings
// (embedding entries, feature/logits names) are inferred from the loaded graphs
// in EagleModel::initialize(), so no export-specific names are set here.
inline EagleConfig parseEagletConfig(const std::filesystem::path& bundle_dir, const ParsedGenieConfig& gc) {
    const auto cfg_path = [&]() -> std::filesystem::path {
        for (auto& e : std::filesystem::directory_iterator(bundle_dir)) {
            if (e.path().extension() == ".json") {
                std::ifstream probe(e.path());
                try {
                    json j = json::parse(probe);
                    if (j.contains("dialog") && j["dialog"].value("type", "") == "eaglet") return e.path();
                } catch (...) {
                }
            }
        }
        throw std::runtime_error("qwen3_eaglet: no eaglet genie_config.json in " + bundle_dir.string());
    }();

    std::ifstream f(cfg_path);
    json          root   = json::parse(f);
    const json&   dialog = root.at("dialog");

    EagleConfig cfg;
    cfg.embedding_quant = gc.embedding_quant;

    if (dialog.contains("eaglet")) {
        cfg.draft_len         = dialog["eaglet"].value("draft-len", cfg.draft_len);
        cfg.n_branches        = dialog["eaglet"].value("n-branches", cfg.n_branches);
        cfg.max_verify_tokens = dialog["eaglet"].value("max-tokens-target-can-evaluate", cfg.max_verify_tokens);
    }

    // Resolve the two engines by role: collect the draft's ctx-bins + token map,
    // and read each engine's RoPE base so we can prove they agree (EAGLE drives
    // both on one shared value).
    std::string          draft_token_map;
    std::optional<float> target_theta;
    std::optional<float> draft_theta;
    for (const json& eng : dialog.at("engine")) {
        const std::string role = eng.value("role", "");
        const json*       pe   = (eng.contains("model") && eng["model"].contains("positional-encoding"))
                                     ? &eng["model"]["positional-encoding"]
                                     : nullptr;
        if (role == "target") {
            if (pe && pe->contains("rope-theta")) target_theta = pe->at("rope-theta").get<float>();
        } else if (role == "draft") {
            if (eng.contains("model") && eng["model"].contains("binary")) {
                for (const auto& b : eng["model"]["binary"].at("ctx-bins"))
                    cfg.draft_model_paths.push_back((bundle_dir / b.get<std::string>()).string());
            }
            if (eng.contains("model")) draft_token_map = eng["model"].value("draft-token-map", "");
            if (pe && pe->contains("rope-theta")) draft_theta = pe->at("rope-theta").get<float>();
        }
    }
    if (cfg.draft_model_paths.empty()) {
        throw std::runtime_error("qwen3_eaglet: no draft engine (role=draft) with ctx-bins in genie_config.json");
    }

    // The draft embeds proposed tokens with its own weights. The genie_config
    // only declares the shared (target) embedding table, so the draft table is
    // resolved by the export convention "draft_" + the declared lut filename.
    {
        const std::string target_lut = root["dialog"]["embedding"].value("lut-path", "quantized_embedding_table.bin");
        const auto        draft_lut  = bundle_dir / ("draft_" + target_lut);
        if (std::filesystem::exists(draft_lut)) cfg.draft_embedding_path = draft_lut.string();
    }
    if (!target_theta || !draft_theta) {
        throw std::runtime_error(
            "qwen3_eaglet: both target and draft engines must declare positional-encoding rope-theta");
    }
    if (*target_theta != *draft_theta) {
        throw std::runtime_error("qwen3_eaglet: target/draft rope-theta mismatch (" + std::to_string(*target_theta) +
                                 " vs " + std::to_string(*draft_theta) + "); EAGLE requires a shared value");
    }
    cfg.rope_theta = *draft_theta;

    // Load the trimmed draft-vocab → full-vocab id map (ordered JSON array/object of ints).
    if (!draft_token_map.empty()) {
        std::ifstream tf(bundle_dir / draft_token_map);
        if (!tf) throw std::runtime_error("qwen3_eaglet: cannot open draft-token-map " + draft_token_map);
        json tm = json::parse(tf);

        // Declared vocab sizes bound the map: a key indexes the draft vocab, a
        // value indexes the full target vocab. Validating here (rather than on
        // every proposal) turns a malformed bundle into one clear load-time error
        // and stops an out-of-range key from driving a multi-GB allocation or an
        // out-of-range value from reaching the embedding lookup. Absent counts
        // (0) disable the corresponding bound rather than reject the bundle.
        const json&  ctx          = dialog.contains("context") ? dialog.at("context") : dialog;
        const size_t draft_nvocab = ctx.value("draft-n-vocab", 0);
        const size_t full_nvocab  = ctx.value("n-vocab", 0);
        auto         check_value  = [&](int32_t value) {
            if (value < 0 || (full_nvocab && static_cast<size_t>(value) >= full_nvocab))
                throw std::runtime_error("qwen3_eaglet: draft-token-map value " + std::to_string(value) +
                                         " is out of range for the target vocab (" + std::to_string(full_nvocab) +
                                         ") in " + draft_token_map);
            return value;
        };

        if (tm.is_array()) {
            cfg.draft_token_map.reserve(tm.size());
            for (const auto& v : tm) cfg.draft_token_map.push_back(check_value(v.get<int32_t>()));
        } else {
            // Object keyed by decimal draft index. nlohmann stores object keys
            // sorted lexicographically ("0","1","10",...), so iteration order does
            // NOT match numeric index -- place each value at its parsed key or the
            // draft-token map ends up scrambled and every proposal is wrong.
            // Parse keys ourselves: a non-numeric key or a negative/oversized index
            // is a malformed bundle, reported as a named error rather than a bare
            // std::stoi throw or an absurd allocation.
            auto parse_key = [&](const std::string& key) -> int32_t {
                size_t consumed = 0;
                long   parsed   = 0;
                try {
                    parsed = std::stol(key, &consumed);
                } catch (const std::exception&) {
                    consumed = 0;  // fall through to the shared diagnostic
                }
                if (consumed != key.size() || parsed < 0 ||
                    (draft_nvocab && static_cast<size_t>(parsed) >= draft_nvocab))
                    throw std::runtime_error(
                        "qwen3_eaglet: draft-token-map has a non-numeric, negative, or "
                        "out-of-range key '" +
                        key + "' (draft vocab " + std::to_string(draft_nvocab) + ") in " + draft_token_map);
                return static_cast<int32_t>(parsed);
            };
            int32_t max_key = -1;
            for (auto it = tm.begin(); it != tm.end(); ++it) max_key = std::max(max_key, parse_key(it.key()));
            cfg.draft_token_map.assign(static_cast<size_t>(max_key + 1), 0);
            for (auto it = tm.begin(); it != tm.end(); ++it)
                cfg.draft_token_map[static_cast<size_t>(parse_key(it.key()))] = check_value(it.value().get<int32_t>());
        }
    }

    return cfg;
}

// Builds and fully initializes both engines. The returned model is ready for
// generate(): the target is initialized via ModelConfig::model_paths and the
// draft via the paths parsed from genie_config.json.
inline std::unique_ptr<EagleModel> makeModel(const QnnRuntimeConfig& runtime_cfg, const ModelConfig& model_cfg) {
    const auto bundle = bundleDirOf(model_cfg);
    auto       gc     = parseGenieConfig(bundle);
    if (gc.dialog_type != "eaglet") {
        throw std::runtime_error("qwen3_eaglet::makeModel requires dialog.type == \"eaglet\"");
    }

    EagleConfig ecfg        = parseEagletConfig(bundle, gc);
    LLMSpec     target_spec = buildSpecSkeleton(gc);
    LLMSpec     draft_spec  = buildSpecSkeleton(gc);

    GENIEX_LOG_INFO(
        "qwen3_eaglet: config parsed (draft_len={} n_branches={} max_verify={} rope_theta={} draft_paths={} "
        "token_map={})",
        ecfg.draft_len,
        ecfg.n_branches,
        ecfg.max_verify_tokens,
        ecfg.rope_theta,
        ecfg.draft_model_paths.size(),
        ecfg.draft_token_map.size());

    auto model = std::make_unique<EagleModel>(std::move(target_spec), std::move(draft_spec), ecfg);
    GENIEX_LOG_INFO("qwen3_eaglet: initializing target + draft engines...");
    if (!model->initialize(runtime_cfg, model_cfg)) {
        throw std::runtime_error("qwen3_eaglet: engine initialize() failed");
    }
    GENIEX_LOG_INFO("qwen3_eaglet: engines ready.");
    return model;
}

}  // namespace qwen3_eaglet
}  // namespace geniex
