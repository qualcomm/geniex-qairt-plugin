// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#include <filesystem>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

#include "llm_model_registry.h"
#include "pipeline/llm_pipeline.h"
#include "types.h"

namespace fs = std::filesystem;

// ─── Per-model file layout ───────────────────────────────────────────────────
struct ModelFiles {
    std::string                modelfiles_subdir;
    std::vector<std::string>   bin_shards;        // in execution order
    std::string                tokenizer  = "tokenizer.json";
    std::string                htp_config = "htp_backend_ext_config.json";
    std::string                embedding;         // empty = on-device embedding
};

static const std::vector<std::pair<std::string, ModelFiles>>& modelFilesTable() {
    // clang-format off
    static const std::vector<std::pair<std::string, ModelFiles>> table = {
        {"qwen3_4b", {
            "qwen3_4b",
            {"qwen3_4b_part_1_of_4.bin", "qwen3_4b_part_2_of_4.bin",
             "qwen3_4b_part_3_of_4.bin", "qwen3_4b_part_4_of_4.bin"}}},

        {"qwen3_4b_instruct_2507", {
            "qwen3_4b_instruct_2507",
            {"qwen3_4b_instruct_2507_part_1_of_4.bin",
             "qwen3_4b_instruct_2507_part_2_of_4.bin",
             "qwen3_4b_instruct_2507_part_3_of_4.bin",
             "qwen3_4b_instruct_2507_part_4_of_4.bin"}}},
    };
    // clang-format on
    return table;
}

static const ModelFiles* findModelFiles(const std::string& name) {
    for (const auto& [key, files] : modelFilesTable())
        if (key == name) return &files;
    return nullptr;
}

// ─── CLI ─────────────────────────────────────────────────────────────────────
struct Args {
    std::string model;
    std::string prompt      = "Hello, briefly introduce yourself.";
    int32_t     max_tokens  = 32;
    int32_t     min_tokens  = 5;
    bool        list_models = false;
};

static void printUsage(const char* prog) {
    std::cout << "Usage: " << prog << " --model <name> [OPTIONS]\n"
              << "  --model <name>      Model id (see --list-models)\n"
              << "  --prompt <text>     Prompt (default: brief intro)\n"
              << "  --max-tokens <n>    Max tokens to generate (default 32)\n"
              << "  --min-tokens <n>    Fail with exit 2 if fewer tokens (default 5)\n"
              << "  --list-models       Print known model ids and exit\n"
              << "  --help\n";
}

static bool parseArgs(int argc, char** argv, Args& args) {
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> std::string {
            return (i + 1 < argc) ? argv[++i] : std::string{};
        };
        if      (a == "--model")       args.model       = next();
        else if (a == "--prompt")      args.prompt      = next();
        else if (a == "--max-tokens")  args.max_tokens  = std::stoi(next());
        else if (a == "--min-tokens")  args.min_tokens  = std::stoi(next());
        else if (a == "--list-models") args.list_models = true;
        else if (a == "--help" || a == "-h") { printUsage(argv[0]); return false; }
        else { std::cerr << "Unknown argument: " << a << "\n"; return false; }
    }
    return true;
}

// ─── main ────────────────────────────────────────────────────────────────────
int main(int argc, char** argv) {
    Args args;
    if (!parseArgs(argc, argv, args)) return 1;

    if (args.list_models) {
        for (const auto& [name, _] : geniex::llm_model_registry())
            std::cout << name << "\n";
        return 0;
    }

    if (args.model.empty()) {
        std::cerr << "--model is required. Use --list-models to see options.\n";
        return 1;
    }

    const auto& registry = geniex::llm_model_registry();
    auto reg_it = registry.find(args.model);
    if (reg_it == registry.end()) {
        std::cerr << "Unknown model '" << args.model << "'. Use --list-models.\n";
        return 1;
    }
    const ModelFiles* files = findModelFiles(args.model);
    if (!files) {
        std::cerr << "Model '" << args.model
                  << "' has no file-path entry in tests/llm.cpp modelFilesTable().\n";
        return 1;
    }

    const fs::path model_dir =
        fs::current_path() / "modelfiles" / files->modelfiles_subdir;

    geniex::QnnRuntimeConfig runtime_cfg;  // auto-detect HTP paths

    geniex::ModelConfig model_cfg;
    for (const auto& shard : files->bin_shards)
        model_cfg.model_paths.push_back((model_dir / shard).string());
    model_cfg.tokenizer_path  = (model_dir / files->tokenizer).string();
    model_cfg.htp_config_path = (model_dir / files->htp_config).string();
    if (!files->embedding.empty())
        model_cfg.embedding_path = (model_dir / files->embedding).string();

    // Fail fast if any expected file is missing, so failures look like
    // "files not staged" rather than deep QNN init errors.
    for (const auto& p : model_cfg.model_paths) {
        if (!fs::exists(p)) {
            std::cerr << "Missing shard: " << p << "\n";
            return 1;
        }
    }
    if (!fs::exists(model_cfg.tokenizer_path)) {
        std::cerr << "Missing tokenizer: " << model_cfg.tokenizer_path << "\n";
        return 1;
    }

    std::cout << "[llm_test] model=" << args.model
              << " prompt=\"" << args.prompt << "\""
              << " max_tokens=" << args.max_tokens
              << " min_tokens=" << args.min_tokens << "\n";

    std::optional<geniex::LLMPipeline> pipe;
    try {
        pipe = reg_it->second.make_pipeline(runtime_cfg, model_cfg);
    } catch (const std::exception& e) {
        std::cerr << "Pipeline construction threw: " << e.what() << "\n";
        return 1;
    }
    if (!pipe) {
        std::cerr << "Pipeline construction returned nullopt.\n";
        return 1;
    }

    geniex::GenerationConfig gen_cfg;
    gen_cfg.max_tokens = args.max_tokens;

    std::string formatted_prompt;
    try {
        formatted_prompt = pipe->applyChatTemplate(args.prompt);
    } catch (const std::exception& e) {
        std::cerr << "applyChatTemplate() threw: " << e.what() << "\n";
        return 1;
    }
    if (formatted_prompt.empty()) {
        std::cerr << "applyChatTemplate() returned empty string.\n";
        return 2;
    }
    std::cout << "[llm_test] formatted_prompt_len=" << formatted_prompt.size() << "\n";

    geniex::GenerateResult result;
    try {
        result = pipe->generate(formatted_prompt, gen_cfg, /*on_token=*/nullptr);
    } catch (const std::exception& e) {
        std::cerr << "generate() threw: " << e.what() << "\n";
        return 1;
    }

    std::cout << "[llm_test] generated_tokens=" << result.generated_tokens
              << " stop_reason=" << result.stop_reason
              << " tps=" << result.tokens_per_second << "\n";

    if (static_cast<int32_t>(result.generated_tokens) < args.min_tokens) {
        std::cerr << "Test FAILED: generated " << result.generated_tokens
                  << " tokens, expected at least " << args.min_tokens << "\n";
        return 2;
    }
    return 0;
}
