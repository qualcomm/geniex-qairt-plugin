// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#include <filesystem>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

#include "dispatch.h"
#include "geniex-proc/types.h"
#include "llm/llm_spec_loader.h"
#include "pipeline/vlm_pipeline.h"
#include "types.h"

namespace fs = std::filesystem;

// ─── CLI ─────────────────────────────────────────────────────────────────────
struct Args {
    std::string model  = "qwen2_5_vl_7b_instruct";
    std::string prompt = "What is in this image?";
    // Resolved relative to CWD (repo root when invoked via ctest).
    fs::path image_path = "modelfiles/assets/test_image.png";
    int32_t  max_tokens = 32;
    int32_t  min_tokens = 5;
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
        std::string a    = argv[i];
        auto        next = [&]() -> std::string { return (i + 1 < argc) ? argv[++i] : std::string{}; };
        if (a == "--model")
            args.model = next();
        else if (a == "--prompt")
            args.prompt = next();
        else if (a == "--image")
            args.image_path = fs::path(next());
        else if (a == "--max-tokens")
            args.max_tokens = std::stoi(next());
        else if (a == "--min-tokens")
            args.min_tokens = std::stoi(next());
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

// ─── main ────────────────────────────────────────────────────────────────────
int main(int argc, char** argv) {
    Args args;
    if (!parseArgs(argc, argv, args)) return 1;

    if (!fs::exists(args.image_path)) {
        std::cerr << "Image not found: " << args.image_path << "\n";
        return 1;
    }

    const fs::path model_dir = fs::current_path() / "modelfiles" / args.model;
    if (!fs::exists(model_dir / "metadata.json")) {
        std::cerr << "Bundle " << model_dir << " has no metadata.json\n";
        return 1;
    }

    geniex::QnnRuntimeConfig runtime_cfg;  // auto-detect HTP paths

    // LLM config: derive shard order, tokenizer, htp from genie_config.json.
    geniex::VLMConfig config;
    try {
        config.llm_config = geniex::modelConfigFromDirectory(model_dir);
    } catch (const std::exception& e) {
        std::cerr << "modelConfigFromDirectory threw: " << e.what() << "\n";
        return 1;
    }

    // Vision config: standard QAIRT VLM bundle convention.
    config.vision_config.model_paths     = {(model_dir / "vision_encoder.bin").string()};
    config.vision_config.htp_config_path = config.llm_config.htp_config_path;

    // External embedding LUT for VLM bundles.
    const auto emb_path = model_dir / "embedding_weights.raw";
    if (fs::exists(emb_path)) config.llm_config.embedding_path = emb_path.string();

    if (!fs::exists(config.vision_config.model_paths.front())) {
        std::cerr << "Missing vision_encoder.bin in " << model_dir << "\n";
        return 1;
    }
    if (!config.llm_config.embedding_path || !fs::exists(*config.llm_config.embedding_path)) {
        std::cerr << "Missing VLM asset (embedding): " << config.llm_config.embedding_path.value_or("<unset>") << "\n";
        return 1;
    }

    std::cout << "[vlm_test] model=" << args.model << " prompt=\"" << args.prompt << "\""
              << " image=" << args.image_path << " max_tokens=" << args.max_tokens << " min_tokens=" << args.min_tokens
              << "\n";

    std::optional<geniex::VLMPipeline> pipe;
    try {
        pipe = geniex::makeVLMPipeline(runtime_cfg, config);
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
    user_msg.mm_content.push_back(geniex::MMContent{geniex::Modality::Image, args.image_path.string()});

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

    std::cout << "[vlm_test] generated_tokens=" << result.generated_tokens << " stop_reason=" << result.stop_reason
              << " tps=" << result.tokens_per_second << "\n";

    if (static_cast<int32_t>(result.generated_tokens) < args.min_tokens) {
        std::cerr << "Test FAILED: generated " << result.generated_tokens << " tokens, expected at least "
                  << args.min_tokens << "\n";
        return 2;
    }
    return 0;
}
