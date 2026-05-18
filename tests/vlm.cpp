// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#include <filesystem>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

#include "pipeline/vlm_pipeline.h"
#include "qwen2_5_vl/qwen2_5_vl.h"
#include "types.h"
#include "geniex-proc/types.h"

namespace fs = std::filesystem;

// ─── CLI ─────────────────────────────────────────────────────────────────────
struct Args {
    std::string model      = "qwen2_5_vl_7b";
    std::string prompt     = "What is in this image?";
    // Resolved relative to CWD (repo root when invoked via ctest).
    fs::path    image_path = "modelfiles/assets/test_image.png";
    int32_t     max_tokens = 32;
    int32_t     min_tokens = 5;
};

static void printUsage(const char* prog) {
    std::cout << "Usage: " << prog << " [OPTIONS]\n"
              << "  --model <name>      VLM model id (default qwen2_5_vl_7b)\n"
              << "  --image <path>      Image file (default: modelfiles/assets/test_image.png)\n"
              << "  --prompt <text>     Prompt to generate on\n"
              << "  --max-tokens <n>    Max tokens to generate (default 32)\n"
              << "  --min-tokens <n>    Fail with exit 2 if fewer tokens (default 5)\n"
              << "  --help\n";
}

static bool parseArgs(int argc, char** argv, Args& args) {
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> std::string {
            return (i + 1 < argc) ? argv[++i] : std::string{};
        };
        if      (a == "--model")      args.model      = next();
        else if (a == "--prompt")     args.prompt     = next();
        else if (a == "--image")      args.image_path = fs::path(next());
        else if (a == "--max-tokens") args.max_tokens = std::stoi(next());
        else if (a == "--min-tokens") args.min_tokens = std::stoi(next());
        else if (a == "--help" || a == "-h") { printUsage(argv[0]); return false; }
        else { std::cerr << "Unknown argument: " << a << "\n"; return false; }
    }
    return true;
}

// ─── main ────────────────────────────────────────────────────────────────────
int main(int argc, char** argv) {
    Args args;
    if (!parseArgs(argc, argv, args)) return 1;

    if (args.model != "qwen2_5_vl_7b") {
        std::cerr << "Unknown VLM model '" << args.model
                  << "' (only 'qwen2_5_vl_7b' is supported).\n";
        return 1;
    }

    if (!fs::exists(args.image_path)) {
        std::cerr << "Image not found: " << args.image_path << "\n";
        return 1;
    }

    const fs::path model_dir =
        fs::current_path() / "modelfiles" / "qwen2_5_vl_7b";

    geniex::QnnRuntimeConfig runtime_cfg;  // auto-detect HTP paths

    geniex::VLMConfig config;
    config.llm_config.model_paths = {
        (model_dir / "part1_of_5.bin").string(),
        (model_dir / "part2_of_5.bin").string(),
        (model_dir / "part3_of_5.bin").string(),
        (model_dir / "part4_of_5.bin").string(),
        (model_dir / "part5_of_5.bin").string(),
    };
    config.llm_config.tokenizer_path  = (model_dir / "tokenizer.json").string();
    config.llm_config.htp_config_path = (model_dir / "htp_backend_ext_config.json").string();
    config.llm_config.embedding_path  = (model_dir / "embedding_weights.raw").string();

    config.vision_config.model_paths     = {(model_dir / "vision_encoder.bin").string()};
    config.vision_config.htp_config_path = (model_dir / "htp_backend_ext_config.json").string();

    // Fail fast if any expected file is missing.
    for (const auto& p : config.llm_config.model_paths) {
        if (!fs::exists(p)) {
            std::cerr << "Missing shard: " << p << "\n";
            return 1;
        }
    }
    for (const auto& p : {config.llm_config.tokenizer_path,
                          config.llm_config.htp_config_path,
                          config.vision_config.model_paths.front()}) {
        if (!fs::exists(p)) {
            std::cerr << "Missing VLM asset: " << p << "\n";
            return 1;
        }
    }
    
    if (!config.llm_config.embedding_path ||
        !fs::exists(*config.llm_config.embedding_path)) {
        std::cerr << "Missing VLM asset (embedding): "
                  << config.llm_config.embedding_path.value_or("<unset>") << "\n";
        return 1;
    }

    std::cout << "[vlm_test] model=" << args.model
              << " prompt=\"" << args.prompt << "\""
              << " image=" << args.image_path
              << " max_tokens=" << args.max_tokens
              << " min_tokens=" << args.min_tokens << "\n";

    std::optional<geniex::VLMPipeline> pipe;
    try {
        pipe = geniex::qwen2_5_vl_7b::makePipeline(runtime_cfg, config);
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

    geniex::ChatMessage user_msg;
    user_msg.role    = geniex::Role::User;
    user_msg.content = args.prompt;
    user_msg.mm_content.push_back(
        geniex::MMContent{geniex::Modality::Image, args.image_path.string()});

    std::string formatted_prompt;
    try {
        formatted_prompt = pipe->applyChatTemplate({user_msg},
                                                   /*add_generation_prompt=*/true);
    } catch (const std::exception& e) {
        std::cerr << "applyChatTemplate() threw: " << e.what() << "\n";
        return 1;
    }
    if (formatted_prompt.empty()) {
        std::cerr << "applyChatTemplate() returned empty string.\n";
        return 2;
    }
    std::cout << "[vlm_test] formatted_prompt_len=" << formatted_prompt.size() << "\n";

    geniex::GenerateResult result;
    try {
        result = pipe->generate(formatted_prompt,
                                std::vector<std::string>{args.image_path.string()},
                                gen_cfg,
                                /*on_token=*/nullptr);
    } catch (const std::exception& e) {
        std::cerr << "generate() threw: " << e.what() << "\n";
        return 1;
    }

    std::cout << "[vlm_test] generated_tokens=" << result.generated_tokens
              << " stop_reason=" << result.stop_reason
              << " tps=" << result.tokens_per_second << "\n";

    if (static_cast<int32_t>(result.generated_tokens) < args.min_tokens) {
        std::cerr << "Test FAILED: generated " << result.generated_tokens
                  << " tokens, expected at least " << args.min_tokens << "\n";
        return 2;
    }
    return 0;
}
