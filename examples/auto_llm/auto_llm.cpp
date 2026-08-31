// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause
//
// Family-free LLM REPL. The model + chat template are loaded entirely from
// the bundle directory, via geniex::auto_llm::makePipeline
// (core/include/pipeline/auto_llm.h) -- no per-family header, and no
// example-local pipeline class either; this file is just CLI parsing + REPL
// plumbing around the shared factory every other model example also uses.

#include "pipeline/auto_llm.h"

#include <filesystem>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

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

namespace fs = std::filesystem;

namespace {

struct Args {
    std::string model_dir;
    std::string tokenizer_config_path;
    std::string system_prompt;
    int32_t     max_tokens      = 512;
    bool        enable_thinking = false;
    bool        verbose         = false;
};

void printUsage(const char* prog) {
    std::cout << "Usage: " << prog << " --model-dir <path> [OPTIONS]\n"
              << "  --model-dir <path>          Required. QAIRT bundle directory.\n"
              << "  --tokenizer-config <path>   tokenizer_config.json path\n"
              << "                              (default: <model-dir>/tokenizer_config.json)\n"
              << "  --system <text>             System prompt, applied once at startup\n"
              << "  --max-tokens <n>            Max tokens to generate (default 512)\n"
              << "  --enable-thinking           Plumb {\"enable_thinking\":true} to the\n"
              << "                              chat template (Qwen3 reasoning models)\n"
              << "  --verbose                   Print TTFT / TPS metrics each turn\n"
              << "  --help, -h\n";
}

bool parseArgs(int argc, char** argv, Args& args) {
    for (int i = 1; i < argc; ++i) {
        std::string a    = argv[i];
        auto        next = [&]() -> std::string { return (i + 1 < argc) ? argv[++i] : std::string{}; };
        if (a == "--model-dir")
            args.model_dir = next();
        else if (a == "--tokenizer-config")
            args.tokenizer_config_path = next();
        else if (a == "--system")
            args.system_prompt = next();
        else if (a == "--max-tokens")
            args.max_tokens = std::stoi(next());
        else if (a == "--enable-thinking")
            args.enable_thinking = true;
        else if (a == "--verbose")
            args.verbose = true;
        else if (a == "--help" || a == "-h") {
            printUsage(argv[0]);
            return false;
        } else {
            std::cerr << "Unknown argument: " << a << "\n";
            printUsage(argv[0]);
            return false;
        }
    }
    if (args.model_dir.empty()) {
        std::cerr << "--model-dir is required\n";
        printUsage(argv[0]);
        return false;
    }
    return true;
}

// Probe for the optional CPU-side embedding LUT. modelConfigFromDirectory
// doesn't populate it; matches what the SDK plugin layer probes for.
void populateEmbeddingPathIfPresent(geniex::ModelConfig& cfg, const fs::path& bundle_dir) {
    for (const char* name : {"embedding_weights.raw", "embed_tokens.npy"}) {
        const auto p = bundle_dir / name;
        if (fs::exists(p)) {
            cfg.embedding_path = p.string();
            return;
        }
    }
}

void printPerfLine(const geniex::GenerateResult& r, bool verbose) {
    if (r.ttft_ms <= 0.0 && r.decode_ms <= 0.0) return;
    if (verbose) {
        std::cout << "\033[1;36m=== Performance ===\033[0m\n"
                  << "Generated tokens : " << r.generated_tokens << "\n"
                  << "TTFT             : " << std::fixed << std::setprecision(1) << r.ttft_ms << " ms\n"
                  << "Decode time      : " << std::fixed << std::setprecision(1) << r.decode_ms << " ms\n"
                  << "Decode speed     : " << std::fixed << std::setprecision(2) << r.tokens_per_second << " tokens/s\n"
                  << "Stop reason      : " << r.stop_reason << "\n"
                  << "===================\n\n";
    } else {
        std::cout << "TTFT: " << std::fixed << std::setprecision(1) << r.ttft_ms << " ms"
                  << "  |  " << std::setprecision(2) << r.tokens_per_second << " tokens/s\n\n";
    }
}

// Runs one turn over the whole conversation so far, streaming the reply.
// Resets the KV cache first: `messages` is re-rendered with full history each
// call, and LLMPipeline::generate appends its prefill at the current n_past_
// with no prefix reuse, so generating without resetting would write a second
// copy of the history into the cache every turn.
geniex::GenerateResult runTurn(geniex::LLMPipeline& pipe, std::vector<geniex::ChatMessage>& messages,
    const geniex::GenerationConfig& gen_cfg, const geniex::ApplyChatTemplateOptions& opts) {
    pipe.reset();

    std::string prompt;
    try {
        prompt = pipe.applyChatTemplate(messages, opts);
    } catch (const std::exception& e) {
        std::cerr << "Chat-template error: " << e.what() << "\n";
        geniex::GenerateResult result;
        result.stop_reason = "error";
        return result;
    }

    std::cout << "\033[33m";
    const auto result = pipe.generate(prompt, gen_cfg, [](const char* piece) {
        std::cout << piece << std::flush;
        return true;
    });
    std::cout << "\033[0m\n";
    return result;
}

}  // namespace

int main(int argc, char** argv) {
#ifdef _WIN32
    enable_utf8_io();
#endif

    Args args;
    if (!parseArgs(argc, argv, args)) return 1;

    const fs::path bundle_dir(args.model_dir);
    if (!fs::is_directory(bundle_dir)) {
        std::cerr << "--model-dir does not exist or is not a directory: " << args.model_dir << "\n";
        return 1;
    }

    std::cout << "\033[1;36mLoading model from " << bundle_dir.string() << "\033[0m\n";

    geniex::ModelConfig model_cfg;
    try {
        model_cfg = geniex::modelConfigFromDirectory(bundle_dir);
        populateEmbeddingPathIfPresent(model_cfg, bundle_dir);
    } catch (const std::exception& e) {
        std::cerr << "Failed to read bundle: " << e.what() << "\n";
        return 1;
    }
    if (!args.tokenizer_config_path.empty()) {
        model_cfg.tokenizer_config_path = args.tokenizer_config_path;
    }

    auto pipe_opt = geniex::auto_llm::makePipeline(geniex::QnnRuntimeConfig{}, model_cfg);
    if (!pipe_opt) {
        std::cerr << "Failed to create pipeline. See logs for details.\n";
        return 1;
    }
    auto& pipe = *pipe_opt;

    std::cout << "\033[1;32mModel loaded.\033[0m\n\n";

    // The pipeline's KV cache holds the prefix matching `messages`; runTurn
    // resets and re-prefills the full history each call, never mid-history.
    std::vector<geniex::ChatMessage> messages;
    if (!args.system_prompt.empty()) {
        messages.push_back({geniex::Role::System, args.system_prompt});
    }

    geniex::GenerationConfig gen_cfg;
    gen_cfg.max_tokens = args.max_tokens;

    geniex::ApplyChatTemplateOptions opts;
    opts.enable_thinking = args.enable_thinking;

    while (true) {
        std::cout << "Enter your prompt (type 'exit' to quit): ";
        std::string input;
        if (!std::getline(std::cin, input) || input == "exit" || input == "quit") break;
        if (input.empty()) continue;

        messages.push_back({geniex::Role::User, input});

        const auto result = runTurn(pipe, messages, gen_cfg, opts);

        if (result.stop_reason == "error" || result.stop_reason == "prompt_too_long" ||
            result.stop_reason == "context_length") {
            // Drop the user turn whose generation failed. prompt_too_long /
            // context_length mean the conversation outgrew the window, so
            // drop the history too rather than let it grow again on retry.
            std::cerr << "Turn dropped (" << result.stop_reason << ").\n";
            messages.pop_back();
            continue;
        }

        messages.push_back({geniex::Role::Assistant, result.full_text});
        printPerfLine(result, args.verbose);
    }

    return 0;
}
