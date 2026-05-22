// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#include <filesystem>
#include <iostream>
#include <optional>
#include <string>

#include "dispatch.h"
#include "llm/llm_spec_loader.h"
#include "pipeline/llm_pipeline.h"
#include "types.h"

namespace fs = std::filesystem;

// ─── CLI ─────────────────────────────────────────────────────────────────────
//
// `--model <name>` names a bundle directory under `modelfiles/<name>/`.
// Everything else (shard ordering, tokenizer, htp config) is read from the
// bundle's genie_config.json by geniex::modelConfigFromDirectory().

struct Args {
    std::string model;
    std::string prompt          = "Hello, briefly introduce yourself.";
    int32_t     max_tokens      = 32;
    int32_t     min_tokens      = 5;
    bool        list_modelfiles = false;
    bool        verbose         = false;
};

static void printUsage(const char* prog) {
    std::cout << "Usage: " << prog << " --model <name> [OPTIONS]\n"
              << "  --model <name>      Bundle directory under modelfiles/\n"
              << "  --prompt <text>     Prompt (default: brief intro)\n"
              << "  --max-tokens <n>    Max tokens to generate (default 32)\n"
              << "  --min-tokens <n>    Fail with exit 2 if fewer tokens (default 5)\n"
              << "  --list-modelfiles   List bundle directories under modelfiles/ and exit\n"
              << "  --verbose           Print generated text and timing\n"
              << "  --help\n";
}

static bool parseArgs(int argc, char** argv, Args& args) {
    for (int i = 1; i < argc; ++i) {
        std::string a    = argv[i];
        auto        next = [&]() -> std::string { return (i + 1 < argc) ? argv[++i] : std::string{}; };
        if (a == "--model")
            args.model = next();
        else if (a == "--prompt")
            args.prompt = next();
        else if (a == "--max-tokens")
            args.max_tokens = std::stoi(next());
        else if (a == "--min-tokens")
            args.min_tokens = std::stoi(next());
        else if (a == "--list-modelfiles")
            args.list_modelfiles = true;
        else if (a == "--verbose")
            args.verbose = true;
        else if (a == "--help" || a == "-h") {
            printUsage(argv[0]);
            return false;
        } else {
            std::cerr << "Unknown argument: " << a << "\n";
            return false;
        }
    }
    return true;
}

int main(int argc, char** argv) {
    Args args;
    if (!parseArgs(argc, argv, args)) return 1;

    const fs::path modelfiles_root = fs::current_path() / "modelfiles";

    if (args.list_modelfiles) {
        for (const auto& entry : fs::directory_iterator(modelfiles_root)) {
            if (entry.is_directory() && fs::exists(entry.path() / "metadata.json"))
                std::cout << entry.path().filename().string() << "\n";
        }
        return 0;
    }

    if (args.model.empty()) {
        std::cerr << "--model is required. Use --list-modelfiles to see options.\n";
        return 1;
    }

    const fs::path bundle_dir = modelfiles_root / args.model;
    if (!fs::exists(bundle_dir / "metadata.json")) {
        std::cerr << "Bundle " << bundle_dir << " has no metadata.json\n";
        return 1;
    }

    geniex::QnnRuntimeConfig runtime_cfg;  // auto-detect HTP paths

    geniex::ModelConfig model_cfg;
    try {
        model_cfg = geniex::modelConfigFromDirectory(bundle_dir);
    } catch (const std::exception& e) {
        std::cerr << "modelConfigFromDirectory threw: " << e.what() << "\n";
        return 1;
    }

    std::cout << "[llm_test] model=" << args.model << " prompt=\"" << args.prompt << "\""
              << " max_tokens=" << args.max_tokens << " min_tokens=" << args.min_tokens << "\n"
              << "[llm_test] shards=" << model_cfg.model_paths.size() << "\n";

    std::optional<geniex::LLMPipeline> pipe;
    try {
        pipe = geniex::makeLLMPipeline(runtime_cfg, model_cfg);
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
        if (args.verbose) {
            std::cout << "[llm_test] generated text:\n  ";
            result = pipe->generate(formatted_prompt, gen_cfg, [](const char* piece) -> bool {
                std::cout << piece << std::flush;
                return true;
            });
            std::cout << "\n";
        } else {
            result = pipe->generate(formatted_prompt, gen_cfg, /*on_token=*/nullptr);
        }
    } catch (const std::exception& e) {
        std::cerr << "generate() threw: " << e.what() << "\n";
        return 1;
    }

    std::cout << "[llm_test] generated_tokens=" << result.generated_tokens << " stop_reason=" << result.stop_reason
              << " ttft_ms=" << result.ttft_ms << " tps=" << result.tokens_per_second << "\n";

    if (args.verbose) {
        std::cout << "[llm_test] full_text:\n" << result.full_text << "\n";
    }

    if (static_cast<int32_t>(result.generated_tokens) < args.min_tokens) {
        std::cerr << "Test FAILED: generated " << result.generated_tokens << " tokens, expected at least "
                  << args.min_tokens << "\n";
        return 2;
    }
    return 0;
}
