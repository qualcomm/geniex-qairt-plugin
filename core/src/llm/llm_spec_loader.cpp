// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#include "llm/llm_spec_loader.h"

#include <algorithm>
#include <fstream>
#include <map>
#include <regex>
#include <set>
#include <stdexcept>
#include <string>

#include "llm/llm_utils.h"  // isKVTensor / isSpecialTensor
#include "logging.h"
#include "utils/detail/json.hpp"

namespace geniex {
namespace {

using json = qualla::json;

// Reads file into a json document; throws on missing file or parse error.
json loadJson(const std::filesystem::path& path) {
    std::ifstream f(path);
    if (!f) {
        throw std::runtime_error("llm_spec_loader: cannot open " + path.string());
    }
    try {
        return json::parse(f);
    } catch (const std::exception& e) {
        throw std::runtime_error("llm_spec_loader: failed to parse " + path.string() + ": " + e.what());
    }
}

// Captures (ar, cl, shard, total) and an optional phase prefix from an LLM-
// style graph name. Accepts forms with or without a phase prefix.
struct GraphNameParts {
    std::string phase_prefix;
    size_t      ar    = 0;
    size_t      cl    = 0;
    size_t      shard = 0;
    size_t      total = 0;
};

bool parseGraphName(const std::string& name, GraphNameParts& out) {
    static const std::regex re(R"((?:([a-zA-Z]+)_)?ar(\d+)_cl(\d+)_(\d+)_of_(\d+))");
    std::smatch             m;
    if (!std::regex_match(name, m, re)) return false;
    out.phase_prefix = m[1].matched ? m[1].str() : "";
    out.ar           = std::stoul(m[2].str());
    out.cl           = std::stoul(m[3].str());
    out.shard        = std::stoul(m[4].str());
    out.total        = std::stoul(m[5].str());
    return true;
}

// Captures (shard, total) from `partN_of_M.bin`-style VLM keys.
bool parsePartShardName(const std::string& name, size_t& shard, size_t& total) {
    static const std::regex re(R"((?:.*_)?part(\d+)_of_(\d+)\.bin)");
    std::smatch             m;
    if (!std::regex_match(name, m, re)) return false;
    shard = std::stoul(m[1].str());
    total = std::stoul(m[2].str());
    return true;
}

// Parses the integer N from "past_key_<N>_in" or "past_value_<N>_in".
std::optional<size_t> parsePastIndex(const std::string& name) {
    static const std::regex re(R"(past_(?:key|value)_(\d+)_(?:in|out))");
    std::smatch             m;
    if (!std::regex_match(name, m, re)) return std::nullopt;
    return std::stoul(m[1].str());
}

template <typename T>
std::optional<T> getOpt(const json& j, const std::string& key) {
    if (!j.contains(key) || j.at(key).is_null()) return std::nullopt;
    return j.at(key).get<T>();
}

// Multi-key lookup: tries each key in order, returning the first present. Lets
// one parser accept both genie_config.json's hyphenated schema and
// metadata.json's snake_case schema without duplicating logic.
template <typename T>
std::optional<T> getAny(const json& j, std::initializer_list<const char*> keys) {
    for (const char* key : keys) {
        if (auto v = getOpt<T>(j, key)) return v;
    }
    return std::nullopt;
}

// Reads the shape of one tensor entry inside metadata.json's inputs/outputs
// map. Returns an empty vector if the entry is missing or malformed.
std::vector<size_t> readShape(const json& tensor_entry) {
    std::vector<size_t> shape;
    if (!tensor_entry.is_object() || !tensor_entry.contains("shape") || !tensor_entry.at("shape").is_array()) {
        return shape;
    }
    for (const auto& d : tensor_entry.at("shape")) {
        if (d.is_number_integer() || d.is_number_unsigned()) shape.push_back(d.get<size_t>());
    }
    return shape;
}

// Per-shard hyperparameter signal from one metadata.json graph entry.
struct ShardWiring {
    std::set<size_t>    kv_layer_indices;
    std::vector<size_t> in_state_shape;    // first non-special input shape (for hidden_size)
    std::vector<size_t> out_state_shape;   // first non-special output shape (fallback)
    std::vector<size_t> past_key_shape;    // first past_key_* shape (for num_kv_heads / head_dim)
    std::vector<size_t> logits_shape;      // for vocab_size
    std::string         first_input_name;  // raw JSON key of the first non-special input
};

ShardWiring readShardWiring(const json& graph_entry, const std::string& /*diag_label*/) {
    ShardWiring w;
    bool        in_state_seen = false, out_state_seen = false;
    if (graph_entry.contains("inputs") && graph_entry.at("inputs").is_object()) {
        for (auto it = graph_entry.at("inputs").begin(); it != graph_entry.at("inputs").end(); ++it) {
            const std::string& key = it.key();
            if (isSpecialTensor(key)) {
                if (auto idx = parsePastIndex(key)) w.kv_layer_indices.insert(*idx);
                if (key.rfind("past_key_", 0) == 0 && w.past_key_shape.empty()) {
                    w.past_key_shape = readShape(it.value());
                }
                continue;
            }
            if (!in_state_seen) {
                w.in_state_shape   = readShape(it.value());
                w.first_input_name = key;
                in_state_seen      = true;
            }
        }
    }
    if (graph_entry.contains("outputs") && graph_entry.at("outputs").is_object()) {
        for (auto it = graph_entry.at("outputs").begin(); it != graph_entry.at("outputs").end(); ++it) {
            const std::string& key = it.key();
            if (key == "logits") w.logits_shape = readShape(it.value());
            if (isSpecialTensor(key)) {
                if (key.rfind("past_key_", 0) == 0 && w.past_key_shape.empty()) {
                    w.past_key_shape = readShape(it.value());
                }
                continue;
            }
            if (!out_state_seen) {
                w.out_state_shape = readShape(it.value());
                out_state_seen    = true;
            }
        }
    }
    return w;
}

void parseVisionPreprocessing(const json& j, ParsedVisionPreprocessing& out) {
    out.image_width         = j.value("image_width", 0);
    out.image_height        = j.value("image_height", 0);
    out.patch_size          = j.value("patch_size", 0);
    out.temporal_patch_size = j.value("temporal_patch_size", 0);
    out.spatial_merge_size  = j.value("spatial_merge_size", 0);
    if (j.contains("normalize_mean") && j.at("normalize_mean").is_array()) {
        for (const auto& v : j.at("normalize_mean")) out.normalize_mean.push_back(v.get<float>());
    }
    if (j.contains("normalize_std") && j.at("normalize_std").is_array()) {
        for (const auto& v : j.at("normalize_std")) out.normalize_std.push_back(v.get<float>());
    }
}

// Maps the rope-type strings to our RopeScaling variant. Accepts both
// genie_config.json's hyphenated keys and metadata.json's snake_case keys, so
// this one implementation serves both schemas.
RopeScaling parseRopeScaling(const json& rs) {
    if (!rs.is_object()) return StandardRope{};
    const std::string type = getAny<std::string>(rs, {"rope-type", "rope_type", "type"}).value_or("default");

    if (type == "llama3") {
        Llama3RopeScaling s;
        s.factor           = getAny<float>(rs, {"factor"}).value_or(1.0f);
        s.low_freq_factor  = getAny<float>(rs, {"low-freq-factor", "low_freq_factor"}).value_or(1.0f);
        s.high_freq_factor = getAny<float>(rs, {"high-freq-factor", "high_freq_factor"}).value_or(4.0f);
        s.original_max_position_embeddings =
            getAny<size_t>(rs, {"original-max-position-embeddings", "original_max_position_embeddings"})
                .value_or(size_t{8192});
        return s;
    }
    if (type == "longrope") {
        LongRopeScaling s;
        for (const char* key : {"long-factor", "long_factor"}) {
            if (rs.contains(key) && rs.at(key).is_array()) {
                for (const auto& v : rs.at(key)) s.long_factor.push_back(v.get<float>());
                break;
            }
        }
        for (const char* key : {"short-factor", "short_factor"}) {
            if (rs.contains(key) && rs.at(key).is_array()) {
                for (const auto& v : rs.at(key)) s.short_factor.push_back(v.get<float>());
                break;
            }
        }
        s.original_max_position_embeddings =
            getAny<size_t>(rs, {"original-max-position-embeddings", "original_max_position_embeddings"})
                .value_or(size_t{4096});
        return s;
    }
    if (type == "qwen2vl-mrope" || type == "qwen3vl-mrope") {
        MRopeScaling s;
        s.spatial_merge_size = getAny<int>(rs, {"spatial-merge-size", "spatial_merge_size"}).value_or(2);
        s.time_step          = getAny<int>(rs, {"time-step", "time_step"}).value_or(50);
        bool found_section   = false;
        for (const char* key : {"mrope-section", "mrope_section"}) {
            if (rs.contains(key) && rs.at(key).is_array()) {
                for (const auto& v : rs.at(key)) s.mrope_section.push_back(v.get<int>());
                found_section = true;
                break;
            }
        }
        if (!found_section) s.mrope_section = {16, 24, 24};  // qwen2-vl default
        return s;
    }
    if (type == "partial" || type == "proportional") {
        // "proportional" is Gemma3/4's name for partial-rotary RoPE: only the
        // front `partial-rotary-factor` of head dims are rotated. Its `factor`
        // is the post-scale (usually 1.0).
        PartialRopeScaling s;
        s.rope_fraction =
            getAny<float>(rs, {"rope-fraction", "rope_fraction", "partial-rotary-factor", "partial_rotary_factor"})
                .value_or(1.0f);
        s.scale = getAny<float>(rs, {"scale", "factor"}).value_or(1.0f);
        return s;
    }
    return StandardRope{};
}

// Reads an embedding block's `datatype` + `quant-param`/`quant_param` (a
// quantized LUT). Absent datatype, or "float32", leaves the spec unquantized.
QuantizedLutSpec parseLutQuant(const json& block) {
    QuantizedLutSpec q;
    if (auto v = getAny<std::string>(block, {"datatype"})) q.datatype = *v;
    for (const char* key : {"quant-param", "quant_param"}) {
        if (block.contains(key) && block.at(key).is_object()) {
            const auto& qp = block.at(key);
            q.scale        = qp.value("scale", 1.0f);
            q.offset       = qp.value("offset", 0);
            break;
        }
    }
    return q;
}

// Sampler defaults, read from metadata.json's `sampler` block.
ParsedSamplerConfig parseSamplerBlock(const json& s) {
    ParsedSamplerConfig out;
    if (!s.is_object()) return out;

    out.seed        = getAny<uint32_t>(s, {"seed"});
    out.temperature = getAny<float>(s, {"temp", "temperature"});
    out.top_k       = getAny<int32_t>(s, {"top-k", "top_k"});
    out.top_p       = getAny<float>(s, {"top-p", "top_p"});

    const json* tp = nullptr;
    for (const char* key : {"token-penalty", "token_penalty"}) {
        if (s.contains(key) && s.at(key).is_object()) {
            tp = &s.at(key);
            break;
        }
    }
    if (tp) {
        out.repetition_penalty = getAny<float>(*tp, {"repetition-penalty", "repetition_penalty"});
        out.presence_penalty   = getAny<float>(*tp, {"presence-penalty", "presence_penalty"});
        out.frequency_penalty  = getAny<float>(*tp, {"frequency-penalty", "frequency_penalty"});
        out.penalty_last_n     = getAny<int32_t>(*tp, {"penalize-last-n", "penalty_last_n"});
    }
    return out;
}

}  // namespace

// ─────────────────────────────────────────────────────────────────────────────
// parseQAIRTMetadata
// ─────────────────────────────────────────────────────────────────────────────
ParsedQAIRTMetadata parseQAIRTMetadata(const std::filesystem::path& bundle_dir) {
    auto path = bundle_dir / "metadata.json";
    auto j    = loadJson(path);

    if (!j.contains("model_files") || !j.at("model_files").is_object()) {
        throw std::runtime_error("llm_spec_loader: metadata.json missing 'model_files' object");
    }
    const auto& model_files = j.at("model_files");

    ParsedQAIRTMetadata out;
    out.model_id = j.value("model_id", std::string{});

    size_t      total_shards = 0;
    std::string vision_encoder_key;

    // Per-shard representative graph entry. Any AR/CL variant of a given
    // shard exposes the same num_kv_heads / head_dim / hidden_size /
    // kv_layer_indices, so the first match wins.
    std::map<size_t, const json*> per_shard_entry;

    for (auto it = model_files.begin(); it != model_files.end(); ++it) {
        const std::string& key = it.key();
        GraphNameParts     parts;
        if (parseGraphName(key, parts)) {
            total_shards = std::max(total_shards, parts.total);
            per_shard_entry.try_emplace(parts.shard, &it.value());
            continue;
        }
        size_t shard = 0, total = 0;
        if (parsePartShardName(key, shard, total)) {
            total_shards           = std::max(total_shards, total);
            per_shard_entry[shard] = &it.value();
            continue;
        }
        if (key == "vision_encoder.bin") {
            vision_encoder_key = key;
            continue;
        }
        GENIEX_LOG_WARN("llm_spec_loader: ignoring unrecognised graph entry '{}'", key);
    }

    if (total_shards == 0) {
        throw std::runtime_error("llm_spec_loader: metadata.json contains no recognisable shard entries");
    }
    out.vision_encoder_graph = vision_encoder_key;

    out.shards.resize(total_shards);

    size_t max_past_key_idx   = 0;
    auto   absorb_hyperparams = [&](const ShardWiring& w) {
        if (out.hidden_size == 0 && w.in_state_shape.size() >= 3) {
            out.hidden_size = w.in_state_shape[w.in_state_shape.size() - 1];
        }
        if (out.hidden_size == 0 && w.out_state_shape.size() >= 3) {
            out.hidden_size = w.out_state_shape[w.out_state_shape.size() - 1];
        }
        if (!w.past_key_shape.empty()) {
            if (out.num_kv_heads == 0) out.num_kv_heads = w.past_key_shape[0];
            if (out.head_dim == 0 && w.past_key_shape.size() >= 3) out.head_dim = w.past_key_shape[2];
        }
        if (out.vocab_size == 0 && !w.logits_shape.empty()) {
            out.vocab_size = w.logits_shape.back();
        }
    };

    for (size_t s = 1; s <= total_shards; ++s) {
        auto it = per_shard_entry.find(s);
        if (it == per_shard_entry.end() || it->second == nullptr) {
            throw std::runtime_error("llm_spec_loader: could not locate graph entry for shard " + std::to_string(s));
        }
        auto w            = readShardWiring(*it->second, "shard " + std::to_string(s));
        out.shards[s - 1] = ShardSpec{};

        // Record shard 0's first-input name (raw JSON key) so the embedding
        // provider factory can decide between input_ids and inputs_embeds.
        if (s == 1) out.first_shard_input_hint = w.first_input_name;

        for (size_t idx : w.kv_layer_indices) max_past_key_idx = std::max(max_past_key_idx, idx);

        absorb_hyperparams(w);
    }

    // num_hidden_layers = highest past_key_<N> index seen + 1.
    if (max_past_key_idx > 0 || out.num_kv_heads > 0) {
        out.num_hidden_layers = max_past_key_idx + 1;
    }

    // Top-level architectures[0]. Empty for bundles that predate this field --
    // caller falls back to parseModelArchitecture.
    out.architecture = j.value("architectures", std::string{});

    // Absent for bundles that predate this field -- dialog_type/ctx_bins stay
    // empty so callers treat that as "basic"/unset, not a real value. EAGLE's
    // `geniex` uses its own nested schema (see qwen3_eaglet.h), so ctx_bins
    // stays empty for it too.
    if (j.contains("geniex") && j.at("geniex").is_object()) {
        const auto& gx = j.at("geniex");

        out.dialog_type = gx.value("dialog_type", std::string{"basic"});

        if (gx.contains("supports_vision") && gx.at("supports_vision").is_boolean()) {
            out.supports_vision = gx.at("supports_vision").get<bool>();
        }

        if (gx.contains("vision_preprocessing") && gx.at("vision_preprocessing").is_object()) {
            ParsedVisionPreprocessing vp;
            parseVisionPreprocessing(gx.at("vision_preprocessing"), vp);
            out.vision_preprocessing = vp;
        }

        if (gx.contains("ctx_bins") && gx.at("ctx_bins").is_array()) {
            for (const auto& b : gx.at("ctx_bins")) {
                if (b.is_string()) out.ctx_bins.push_back(b.get<std::string>());
            }
        }

        if (gx.contains("context") && gx.at("context").is_object()) {
            const auto& ctx        = gx.at("context");
            out.max_context_length = ctx.value("max_context_length", size_t{0});
            if (auto v = getOpt<int32_t>(ctx, "bos_token")) out.bos_token_id = *v;
            if (auto v = getOpt<int32_t>(ctx, "pad_token")) out.pad_token_id = *v;
            if (ctx.contains("eos_token") && !ctx.at("eos_token").is_null()) {
                const auto& eos = ctx.at("eos_token");
                if (eos.is_number_integer()) {
                    out.eos_token_ids.push_back(eos.get<int32_t>());
                } else if (eos.is_array()) {
                    for (const auto& e : eos) out.eos_token_ids.push_back(e.get<int32_t>());
                }
            }
        }

        if (gx.contains("positional_encoding") && gx.at("positional_encoding").is_object()) {
            const auto& pe = gx.at("positional_encoding");
            if (auto v = getOpt<float>(pe, "rope_theta")) out.rope_theta = *v;
            if (pe.contains("rope_scaling") && pe.at("rope_scaling").is_object()) {
                out.rope_scaling = parseRopeScaling(pe.at("rope_scaling"));
            }
        }

        if (gx.contains("local_positional_encoding") && gx.at("local_positional_encoding").is_object()) {
            const auto& lpe                       = gx.at("local_positional_encoding");
            out.local_positional_encoding_present = true;
            if (auto v = getOpt<float>(lpe, "rope_theta")) out.local_rope_theta = *v;
            if (lpe.contains("rope_scaling") && lpe.at("rope_scaling").is_object()) {
                out.local_rope_scaling = parseRopeScaling(lpe.at("rope_scaling"));
            }
        }

        // lut_path is bundle-relative; resolve it against bundle_dir so
        // EmbeddingInputProvider's mmap open works regardless of cwd.
        if (gx.contains("embedding") && gx.at("embedding").is_object()) {
            const auto& emb = gx.at("embedding");
            if (auto v = getOpt<std::string>(emb, "lut_path")) out.embedding_lut_path = (bundle_dir / *v).string();
            out.embedding_quant = parseLutQuant(emb);
        }

        if (gx.contains("perlayer_embedding") && gx.at("perlayer_embedding").is_object()) {
            const auto& ple = gx.at("perlayer_embedding");
            if (auto v = getOpt<std::string>(ple, "lut_path")) {
                out.perlayer_embedding_lut_path = (bundle_dir / *v).string();
            }
            out.perlayer_embedding_size  = ple.value("size", size_t{0});
            out.perlayer_embedding_quant = parseLutQuant(ple);
        }

        if (gx.contains("sampler") && gx.at("sampler").is_object()) {
            out.sampler = parseSamplerBlock(gx.at("sampler"));
        }
    }

    return out;
}

// ─────────────────────────────────────────────────────────────────────────────
// runtimeConfigFromMetadata
// ─────────────────────────────────────────────────────────────────────────────
ParsedGenieConfig runtimeConfigFromMetadata(const ParsedQAIRTMetadata& meta) {
    ParsedGenieConfig gc;
    gc.bos_token_id                      = meta.bos_token_id;
    gc.eos_token_ids                     = meta.eos_token_ids;
    gc.pad_token_id                      = meta.pad_token_id;
    gc.rope_theta                        = meta.rope_theta;
    gc.rope_scaling                      = meta.rope_scaling;
    gc.embedding_lut_path                = meta.embedding_lut_path;
    gc.embedding_quant                   = meta.embedding_quant;
    gc.local_positional_encoding_present = meta.local_positional_encoding_present;
    gc.local_rope_theta                  = meta.local_rope_theta;
    gc.local_rope_scaling                = meta.local_rope_scaling;
    gc.perlayer_embedding_lut_path       = meta.perlayer_embedding_lut_path;
    gc.perlayer_embedding_size           = meta.perlayer_embedding_size;
    gc.perlayer_embedding_quant          = meta.perlayer_embedding_quant;
    return gc;
}

// ─────────────────────────────────────────────────────────────────────────────
// buildSpecSkeleton
// ─────────────────────────────────────────────────────────────────────────────
LLMSpec buildSpecSkeleton(const ParsedGenieConfig& gc) {
    LLMSpec spec;
    spec.state_blocks  = {makeKVStateBlock()};
    spec.eos_token_ids = gc.eos_token_ids;
    spec.bos_token_id  = gc.bos_token_id;
    return spec;
}

// ─────────────────────────────────────────────────────────────────────────────
// Provider factories
// ─────────────────────────────────────────────────────────────────────────────
std::unique_ptr<InputProvider> makeRoPEProvider(
    size_t head_dim, const ParsedGenieConfig& gc, std::string cos_name, std::string sin_name) {
    return std::visit(
        [&](const auto& s) -> std::unique_ptr<InputProvider> {
            using T = std::decay_t<decltype(s)>;
            if constexpr (std::is_same_v<T, Llama3RopeScaling>) {
                // llama3 frequency scaling materially changes the RoPE table
                // (low-freq dims divided by factor, mid band interpolated). The
                // compiled graph consumes position_ids_cos/sin as live inputs,
                // so we must reproduce Genie's scaled table to match its logits.
                GENIEX_LOG_INFO(
                    "llm_spec_loader: rope_scaling=llama3 (factor={}); using Llama3 RoPE provider", s.factor);
                return std::make_unique<Llama3RoPEInputProvider>(head_dim,
                    gc.rope_theta,
                    s.factor,
                    s.low_freq_factor,
                    s.high_freq_factor,
                    static_cast<int>(s.original_max_position_embeddings),
                    cos_name,
                    sin_name);
            } else if constexpr (std::is_same_v<T, LongRopeScaling>) {
                const size_t orig = s.original_max_position_embeddings ? s.original_max_position_embeddings : 4096;
                return std::make_unique<LongRoPEInputProvider>(head_dim,
                    gc.rope_theta,
                    s.long_factor,
                    /*max_position_embeddings=*/131072,
                    static_cast<int>(orig),
                    cos_name,
                    sin_name);
            } else if constexpr (std::is_same_v<T, PartialRopeScaling>) {
                // Gemma3/4's global RoPE ships two on-disk layouts. The classic
                // export names the pair position_ids_cos/sin and stores the
                // compact rope_dim/2-wide table; the newer export (QAIRT 2.45+)
                // names it position_ids_global_cos/sin and stores the full
                // head_dim/2-wide zero-padded rotate_half table. Detect by name.
                const bool full_width = cos_name.find("_global") != std::string::npos;
                return std::make_unique<PartialRoPEInputProvider>(
                    head_dim, gc.rope_theta, s.rope_fraction, s.scale, cos_name, sin_name, full_width);
            } else if constexpr (std::is_same_v<T, MRopeScaling>) {
                // Caller (VLM family) wires a dedicated MRoPEInputProvider with
                // the full mrope_section; for the LLM dispatch this branch is
                // unreachable. Falling back here keeps the function total.
                GENIEX_LOG_INFO("llm_spec_loader: rope_scaling=mrope (mrope_section={}); using standard RoPE provider",
                    s.mrope_section.size());
                return std::make_unique<RoPEInputProvider>(head_dim, gc.rope_theta, cos_name, sin_name);
            } else {
                return std::make_unique<RoPEInputProvider>(head_dim, gc.rope_theta, cos_name, sin_name);
            }
        },
        gc.rope_scaling);
}

std::unique_ptr<InputProvider> makeEmbeddingProvider(
    const std::string& first_shard_input, const ParsedGenieConfig& gc) {
    if (first_shard_input == "input_ids") {
        int32_t pad = gc.pad_token_id;
        if (pad < 0) pad = gc.eos_token_ids.empty() ? 0 : gc.eos_token_ids.front();
        return std::make_unique<TokenIdInputProvider>("input_ids", pad);
    }
    if (first_shard_input == "input_embeds" || first_shard_input == "inputs_embeds") {
        auto p = std::make_unique<EmbeddingInputProvider>(first_shard_input);
        // Quantized main LUT (Gemma4 and other large-vocab bundles): mmap +
        // per-row conversion instead of a dequantized in-RAM table.
        if (gc.embedding_quant.quantized()) p->setQuantization(gc.embedding_quant);
        return p;
    }
    throw std::runtime_error("llm_spec_loader: unrecognised first-shard input '" + first_shard_input +
                             "' — expected 'input_ids', 'input_embeds', or 'inputs_embeds'");
}

// ─────────────────────────────────────────────────────────────────────────────
// Bundle helpers
// ─────────────────────────────────────────────────────────────────────────────
std::filesystem::path bundleDirOf(const ModelConfig& model_cfg) {
    if (model_cfg.model_paths.empty()) {
        throw std::runtime_error("llm_spec_loader: model_cfg.model_paths is empty");
    }
    return std::filesystem::path(model_cfg.model_paths.front()).parent_path();
}

ModelConfig modelConfigFromDirectory(const std::filesystem::path& bundle_dir) {
    ModelConfig cfg;

    auto tok = bundle_dir / "tokenizer.json";
    if (!std::filesystem::exists(tok)) {
        throw std::runtime_error("llm_spec_loader: tokenizer.json not found in " + bundle_dir.string());
    }
    cfg.tokenizer_path = tok.string();

    auto htp = bundle_dir / "htp_backend_ext_config.json";
    if (std::filesystem::exists(htp)) {
        cfg.htp_config_path = htp.string();
        cfg.num_cores       = parseHtpCoreCount(htp);
    }

    // metadata.json's `geniex` block is the sole source for ctx-bins ordering
    // and the embedding LUT path. A bundle missing either is reported below
    // rather than silently falling through to the directory glob.
    try {
        auto meta = parseQAIRTMetadata(bundle_dir);
        for (const auto& b : meta.ctx_bins) cfg.model_paths.push_back((bundle_dir / b).string());
        if (!cfg.embedding_path && meta.embedding_lut_path) {
            cfg.embedding_path = *meta.embedding_lut_path;
        }
        if (meta.ctx_bins.empty()) {
            GENIEX_LOG_WARN("llm_spec_loader: metadata.json in {} has no geniex.ctx_bins", bundle_dir.string());
        }
    } catch (const std::exception& e) {
        GENIEX_LOG_WARN("llm_spec_loader: failed to read metadata.json in {}: {}", bundle_dir.string(), e.what());
    }

    if (cfg.model_paths.empty()) {
        // Fallback for a bundle with no geniex.ctx_bins. Skip a .bin the config
        // would have flagged as the embedding LUT -- it is not a context binary,
        // and handing it to contextCreateFromBinary fails with an opaque
        // "Failed to get context binary info" rather than naming the file.
        std::vector<std::string> bins;
        for (const auto& entry : std::filesystem::directory_iterator(bundle_dir)) {
            if (entry.path().extension() != ".bin") continue;
            if (cfg.embedding_path && entry.path().string() == *cfg.embedding_path) continue;
            bins.push_back(entry.path().string());
        }
        std::sort(bins.begin(), bins.end());
        cfg.model_paths = std::move(bins);
    }

    if (cfg.model_paths.empty()) {
        throw std::runtime_error("llm_spec_loader: no .bin shards found in " + bundle_dir.string());
    }
    return cfg;
}

uint32_t parseHtpCoreCount(const std::filesystem::path& htp_config_path) {
    if (!std::filesystem::exists(htp_config_path)) return 0;

    json j;
    try {
        j = loadJson(htp_config_path);
    } catch (const std::exception& e) {
        // Malformed JSON is not fatal here — QnnHtpNetRunExtensions will surface
        // its own error when it parses the same file at backend init.
        GENIEX_LOG_WARN("llm_spec_loader: could not parse {}: {}", htp_config_path.string(), e.what());
        return 0;
    }

    if (!j.contains("devices") || !j.at("devices").is_array()) return 0;

    uint32_t max_cores = 0;
    for (const auto& device : j.at("devices")) {
        if (!device.contains("cores") || !device.at("cores").is_array()) continue;
        max_cores = std::max<uint32_t>(max_cores, static_cast<uint32_t>(device.at("cores").size()));
    }
    return max_cores;
}

void parseHtpConfig(const std::filesystem::path& htp_config_path, HtpPerfConfig& cfg) {
    if (!std::filesystem::exists(htp_config_path)) return;

    json j;
    try {
        j = loadJson(htp_config_path);
    } catch (const std::exception& e) {
        GENIEX_LOG_WARN("htp config: could not parse {}: {}", htp_config_path.string(), e.what());
        return;
    }

    static const std::unordered_map<std::string, PerfProfile> kProfiles = {
        {"low_balanced", PerfProfile::LOW_BALANCED},
        {"balanced", PerfProfile::BALANCED},
        {"default", PerfProfile::DEFAULT},
        {"high_performance", PerfProfile::HIGH_PERFORMANCE},
        {"sustained_high_performance", PerfProfile::SUSTAINED_HIGH_PERFORMANCE},
        {"burst", PerfProfile::BURST},
        {"extreme_power_saver", PerfProfile::EXTREME_POWER_SAVER},
        {"low_power_saver", PerfProfile::LOW_POWER_SAVER},
        {"power_saver", PerfProfile::POWER_SAVER},
        {"high_power_saver", PerfProfile::HIGH_POWER_SAVER},
        {"system_settings", PerfProfile::SYSTEM_SETTINGS},
    };

    // Keys we apply. Everything else is classified below.
    static const std::set<std::string> kApplied = {"devices",
        "cores",
        "core_id",
        "perf_profile",
        "rpc_control_latency",
        "rpc_polling_time",
        "hmx_timeout_us",
        "adaptive_polling_time"};

    // Keys the QAIRT schema marks "Used by qnn-context-binary-generator during offline
    // preparation" -- already baked into the binary, so INFO rather than WARN.
    static const std::set<std::string> kPrepareOnly = {"graphs",
        "graph_names",
        "graph_name",
        "vtcm_mb",
        "hvx_threads",
        "dlbc",
        "dlbc_weights",
        "weights_packing",
        "num_cores",
        "short_depth_conv_on_hmx_off",
        "fold_relu_activation_into_conv_off",
        "advanced_activation_fusion",
        "use_high_precision_fp16_sigmoid",
        "monolithic_lstm",
        "weight_sharing_enabled",
        "lora_weight_sharing",
        "soc_id",
        "soc_model",
        "dsp_arch"};

    // Handled outside this parser (parseHtpCoreCount, the RpcMem zero-copy path),
    // so they must not be reported as unimplemented.
    static const std::set<std::string> kHandledElsewhere = {"memory", "mem_type", "context"};

    const auto audit = [](const json& obj, const char* where) {
        if (!obj.is_object()) return;
        for (const auto& [key, _] : obj.items()) {
            if (kApplied.count(key) || kHandledElsewhere.count(key)) continue;
            if (kPrepareOnly.count(key)) {
                GENIEX_LOG_INFO(
                    "htp config: '{}' under {} is an offline-preparation option; already baked "
                    "into the context binary, nothing to apply at load time",
                    key,
                    where);
            } else {
                GENIEX_LOG_WARN("htp config: '{}' under {} is not applied by this runtime", key, where);
            }
        }
    };

    audit(j, "the document root");
    if (j.contains("memory")) audit(j.at("memory"), "memory");
    if (j.contains("context")) audit(j.at("context"), "context");

    if (j.contains("devices") && j.at("devices").is_array()) {
        for (const auto& device : j.at("devices")) {
            audit(device, "devices[]");
            if (!device.contains("cores") || !device.at("cores").is_array()) continue;
            for (const auto& core : device.at("cores")) {
                audit(core, "devices[].cores[]");
                if (core.contains("perf_profile") && core.at("perf_profile").is_string()) {
                    const auto name = core.at("perf_profile").get<std::string>();
                    const auto it   = kProfiles.find(name);
                    if (it != kProfiles.end()) {
                        cfg.profile = it->second;
                    } else {
                        GENIEX_LOG_WARN("htp config: unknown perf_profile '{}'; keeping default", name);
                    }
                }
                const auto readUs = [&core](const char* key, uint32_t& out) {
                    if (core.contains(key) && core.at(key).is_number_unsigned()) out = core.at(key).get<uint32_t>();
                };
                readUs("rpc_control_latency", cfg.rpc_control_latency_us);
                readUs("rpc_polling_time", cfg.rpc_polling_time_us);
                readUs("hmx_timeout_us", cfg.hmx_timeout_us);
                readUs("adaptive_polling_time", cfg.adaptive_polling_time_us);
                break;  // one vote per device; core 0 wins
            }
        }
    }

    GENIEX_LOG_INFO(
        "htp config: perf_profile={} rpc_control_latency={}us rpc_polling={}us hmx_timeout={}us "
        "adaptive_polling={}us",
        static_cast<int>(cfg.profile),
        cfg.rpc_control_latency_us,
        cfg.rpc_polling_time_us,
        cfg.hmx_timeout_us,
        cfg.adaptive_polling_time_us);
}

HtpPerfConfig resolveHtpPerfConfig(const ModelConfig& model_cfg) {
    HtpPerfConfig cfg{};
    if (!model_cfg.htp_config_path.empty()) {
        parseHtpConfig(model_cfg.htp_config_path, cfg);
    }
    if (model_cfg.perf_profile) cfg.profile = *model_cfg.perf_profile;

    const auto prefer = [](uint32_t requested, uint32_t& out) {
        if (requested != 0) out = requested;
    };
    prefer(model_cfg.rpc_control_latency_us, cfg.rpc_control_latency_us);
    prefer(model_cfg.rpc_polling_time_us, cfg.rpc_polling_time_us);
    prefer(model_cfg.hmx_timeout_us, cfg.hmx_timeout_us);
    prefer(model_cfg.adaptive_polling_time_us, cfg.adaptive_polling_time_us);
    return cfg;
}

std::string parseModelArchitecture(const std::filesystem::path& bundle_dir) {
    const auto path = bundle_dir / "config.json";
    if (!std::filesystem::exists(path)) return {};

    json j;
    try {
        j = loadJson(path);
    } catch (const std::exception& e) {
        GENIEX_LOG_WARN("llm_spec_loader: could not parse {}: {}", path.string(), e.what());
        return {};
    }

    if (!j.contains("architectures") || !j.at("architectures").is_array() || j.at("architectures").empty()) {
        return {};
    }
    const auto& first = j.at("architectures").front();
    return first.is_string() ? first.get<std::string>() : std::string{};
}

}  // namespace geniex
