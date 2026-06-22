// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "geniex-proc/types.h"
#include "qwen3_vl.h"
#include "types.h"

#ifdef _WIN32
#include <windows.h>
static void enable_utf8_io() {
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);
    DWORD  mode = 0;
    HANDLE hOut = GetStdHandle(STD_OUTPUT_HANDLE);
    if (GetConsoleMode(hOut, &mode)) SetConsoleMode(hOut, mode | ENABLE_VIRTUAL_TERMINAL_PROCESSING);
}
#endif

struct Args {
    int32_t     max_tokens    = 512;
    bool        verbose       = false;
    std::string system_prompt = "You are a helpful AI assistant";
};

static void printUsage(const char* prog) {
    std::cout << "Usage: " << prog << " [OPTIONS]\n"
              << "  --max-tokens <n>      Max tokens to generate (default 512)\n"
              << "  --system-prompt <s>   System prompt\n"
              << "  --verbose             Print performance metrics\n"
              << "  --help\n"
              << "\n"
              << "At the prompt, include an image path anywhere in the line:\n"
              << "  > describe this picture /path/to/cat.jpg\n";
}

static bool parseArgs(int argc, char** argv, Args& args) {
    for (int i = 1; i < argc; ++i) {
        std::string a    = argv[i];
        auto        next = [&]() -> std::string { return (i + 1 < argc) ? argv[++i] : std::string{}; };
        if (a == "--max-tokens")
            args.max_tokens = std::stoi(next());
        else if (a == "--system-prompt")
            args.system_prompt = next();
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

static bool isImageFile(const std::string& path) {
    std::string p = path;
    std::transform(p.begin(), p.end(), p.begin(), ::tolower);
    return p.ends_with(".jpg") || p.ends_with(".jpeg") || p.ends_with(".png") || p.ends_with(".bmp") ||
           p.ends_with(".gif") || p.ends_with(".webp");
}

static void parseInput(const std::string& input, std::string& prompt_text, std::vector<std::string>& image_paths) {
    image_paths.clear();
    std::vector<std::string> text_tokens;
    std::istringstream       iss(input);
    std::string              token;
    while (iss >> token) {
        if (isImageFile(token))
            image_paths.push_back(token);
        else
            text_tokens.push_back(token);
    }
    prompt_text.clear();
    for (size_t i = 0; i < text_tokens.size(); ++i) {
        if (i > 0) prompt_text += ' ';
        prompt_text += text_tokens[i];
    }
}

int main(int argc, char** argv) {
#ifdef _WIN32
    enable_utf8_io();
#endif

    Args args;
    if (!parseArgs(argc, argv, args)) return 1;

    const auto model_dir = std::filesystem::current_path() / "modelfiles" / "qwen3_vl_4b_instruct";

    geniex::QnnRuntimeConfig runtime_cfg;
    geniex::VLMConfig        config;

    config.llm_config.model_paths = {
        (model_dir / "part1_of_4.bin").string(),
        (model_dir / "part2_of_4.bin").string(),
        (model_dir / "part3_of_4.bin").string(),
        (model_dir / "part4_of_4.bin").string(),
    };
    config.llm_config.tokenizer_path  = (model_dir / "tokenizer.json").string();
    config.llm_config.htp_config_path = (model_dir / "htp_backend_ext_config.json").string();
    config.llm_config.embedding_path  = (model_dir / "embedding_weights.raw").string();

    config.vision_config.model_paths     = {(model_dir / "vision_encoder.bin").string()};
    config.vision_config.htp_config_path = (model_dir / "htp_backend_ext_config.json").string();

    geniex::GenerationConfig gen_cfg;
    gen_cfg.max_tokens = args.max_tokens;

    std::cout << "\033[1;32m"
              << "   ______           _     _  __\n"
              << "  / ____/__  ____  (_)__ | |/ /\n"
              << " / / __/ _ \\/ __ \\/ / _ \\|   / \n"
              << "/ /_/ /  __/ / / / /  __/   |  \n"
              << "\\____/\\___/_/ /_/_/\\___/_/|_| \n"
              << "\033[0m\n";

    std::cout << "\033[1;36mLoading Qwen3-VL-4B-Instruct...\033[0m\n";
    auto maybe_pipe = geniex::qwen3_vl::makePipeline(runtime_cfg, config);
    if (!maybe_pipe) {
        std::cerr << "Failed to initialize pipeline.\n";
        return 1;
    }
    auto& pipe = *maybe_pipe;

    std::cout << "\033[1;32mModel loaded.\033[0m\n\n";

    bool first_turn = true;
    while (true) {
        std::cout << "Enter your prompt (type 'exit' to quit, 'reset' to clear): ";
        std::string input;
        if (!std::getline(std::cin, input)) break;
        if (input == "exit" || input == "quit") break;
        if (input == "reset") {
            pipe.reset();
            first_turn = true;
            std::cout << "\033[1;36m[conversation reset]\033[0m\n\n";
            continue;
        }

        std::string              prompt_text;
        std::vector<std::string> image_paths;
        parseInput(input, prompt_text, image_paths);

        // Build messages for this turn only.
        std::vector<geniex::ChatMessage> messages;
        if (first_turn && !args.system_prompt.empty()) messages.push_back({geniex::Role::System, args.system_prompt});
        geniex::ChatMessage user_msg{geniex::Role::User, prompt_text};
        for (const auto& img : image_paths) user_msg.mm_content.push_back({geniex::Modality::Image, img});
        messages.push_back(std::move(user_msg));

        std::string formatted = pipe.applyChatTemplate(messages);
        first_turn            = false;

        std::cout << "\033[33m";
        auto on_token = [](const char* piece) -> bool {
            std::cout << piece << std::flush;
            return true;
        };

        geniex::GenerateResult result = pipe.generate(formatted, image_paths, gen_cfg, on_token);
        std::cout << "\033[0m\n";

        if (result.stop_reason == "error") {
            std::cerr << "Error during generation — resetting conversation.\n";
            pipe.reset();
            first_turn = true;
            continue;
        }

        if (args.verbose) {
            std::cout << "\033[1;36m=== Performance ===\033[0m\n"
                      << "Prompt tokens    : " << result.prompt_tokens << "\n"
                      << "Generated tokens : " << result.generated_tokens << "\n"
                      << "TTFT             : " << std::fixed << std::setprecision(1) << result.ttft_ms << " ms\n"
                      << "Decode time      : " << std::fixed << std::setprecision(1) << result.decode_ms << " ms\n"
                      << "Decode speed     : " << std::fixed << std::setprecision(2) << result.tokens_per_second
                      << " tokens/s\n"
                      << "Stop reason      : " << result.stop_reason << "\n"
                      << "===================\n\n";
        } else {
            std::cout << "TTFT: " << std::fixed << std::setprecision(1) << result.ttft_ms << " ms"
                      << "  |  " << std::setprecision(2) << result.tokens_per_second << " tokens/s\n\n";
        }
    }

    return 0;
}
