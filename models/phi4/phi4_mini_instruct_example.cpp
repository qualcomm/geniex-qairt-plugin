// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#include <filesystem>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "pipeline/auto_model.h"
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

namespace fs = std::filesystem;

struct Args {
    fs::path    model_dir;
    std::string prompt;  // non-empty → single-shot, non-interactive
    std::string system_prompt = "You are a helpful AI assistant.";
    int32_t     max_tokens    = 512;
    bool        verbose       = false;
    bool        sample        = false;
};

static void printUsage(const char* prog) {
    std::cout << "Usage: " << prog << " [OPTIONS]\n"
              << "  --model-dir <path> QAIRT bundle directory (default: ./modelfiles/phi4_mini_instruct)\n"
              << "  --prompt <text>    Run once on this prompt and exit (non-interactive)\n"
              << "  --system <text>    System prompt (default: \"You are a helpful AI assistant.\")\n"
              << "  --max-tokens <n>   Max tokens to generate (default 512)\n"
              << "  --sample           Sample using genie_config.json's dialog.sampler defaults.\n"
              << "                     Without this, decoding is greedy argmax, which can loop\n"
              << "                     on list-style answers.\n"
              << "  --verbose          Print performance metrics\n"
              << "  --help\n";
}

static bool parseArgs(int argc, char** argv, Args& args) {
    for (int i = 1; i < argc; ++i) {
        std::string a    = argv[i];
        auto        next = [&]() -> std::string { return (i + 1 < argc) ? argv[++i] : std::string{}; };
        if (a == "--model-dir")
            args.model_dir = next();
        else if (a == "--prompt")
            args.prompt = next();
        else if (a == "--system")
            args.system_prompt = next();
        else if (a == "--max-tokens")
            args.max_tokens = std::stoi(next());
        else if (a == "--sample")
            args.sample = true;
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
    if (args.model_dir.empty()) args.model_dir = fs::current_path() / "modelfiles" / "phi4_mini_instruct";
    return true;
}

// Sampling config. Greedy argmax by default (repo convention); --sample applies
// the bundle's dialog.sampler defaults, which avoids the repetition loops greedy
// decoding falls into on list-style answers.
static geniex::GenerationConfig makeGenCfg(const Args& args) {
    geniex::GenerationConfig gen_cfg;
    gen_cfg.max_tokens = args.max_tokens;
    if (!args.sample) return gen_cfg;

    const auto s            = geniex::parseGenieSamplerConfig(args.model_dir);
    gen_cfg.enable_sampling = true;
    if (s.seed) gen_cfg.seed = *s.seed;
    if (s.temperature) gen_cfg.temperature = *s.temperature;
    if (s.top_k) gen_cfg.top_k = *s.top_k;
    if (s.top_p) gen_cfg.top_p = *s.top_p;
    if (s.repetition_penalty) gen_cfg.repetition_penalty = *s.repetition_penalty;
    return gen_cfg;
}

// Runs one turn over the whole conversation so far, streaming the reply.
//
// `messages` accumulates across turns and the full history is re-rendered and
// re-prefilled each time, so the cache must start empty: LLMModel::generate
// appends its prefill at the current n_past_ and has no prefix reuse, so
// generating over the full history without resetting would write a second copy
// of it into the KV cache. Cost is a re-prefill per turn, which grows with the
// conversation; TTFT rises accordingly.
//
// Returns false if the turn failed and the caller should drop it: on error, or
// when the conversation outgrew the context window.
static bool runTurn(geniex::LLMPipeline& pipe, std::vector<geniex::ChatMessage>& messages, const Args& args) {
    pipe.reset();

    std::string prompt;
    try {
        prompt = pipe.applyChatTemplate(messages);
    } catch (const std::exception& e) {
        std::cerr << "Chat-template error: " << e.what() << "\n";
        return false;
    }

    const geniex::GenerationConfig gen_cfg = makeGenCfg(args);

    geniex::GenerateResult result;
    std::cout << "\033[33m";
    try {
        result = pipe.generate(prompt, gen_cfg, [](const char* piece) {
            std::cout << piece << std::flush;
            return true;
        });
    } catch (const std::exception& e) {
        // LLMPipeline::generate already folds prompt-too-long and
        // context-length-exceeded into stop_reason, so anything landing here is
        // unexpected. Report it instead of letting it unwind out of main, where
        // it would abort with no message.
        std::cout << "\033[0m\n";
        std::cerr << "Generation error: " << e.what() << "\n";
        return false;
    }
    std::cout << "\033[0m\n";

    if (result.stop_reason == "error" || result.stop_reason == "context_length" ||
        result.stop_reason == "prompt_too_long") {
        std::cerr << "Turn dropped (" << result.stop_reason << "); conversation reset.\n";
        return false;
    }

    messages.push_back({geniex::Role::Assistant, result.full_text});

    if (args.verbose) {
        std::cout << "\033[1;36m=== Performance ===\033[0m\n"
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
    return true;
}

int main(int argc, char** argv) {
#ifdef _WIN32
    enable_utf8_io();
#endif

    Args args;
    if (!parseArgs(argc, argv, args)) return 1;

    // All QNN runtime paths are left as std::nullopt → auto-detected from
    // htp-files/ installed alongside geniex_core.
    geniex::QnnRuntimeConfig runtime_cfg;

    // Derive model_paths (in ctx-bins order), tokenizer_path, and
    // htp_config_path from the bundle's genie_config.json rather than
    // hardcoding shard filenames.
    geniex::ModelConfig model_cfg;
    try {
        model_cfg = geniex::modelConfigFromDirectory(args.model_dir);
    } catch (const std::exception& e) {
        std::cerr << "Failed to read bundle: " << e.what() << "\n";
        return 1;
    }
    // tokenizer_config_path left unset → discovered next to the bundle.

    std::cout << "\033[1;32m"
              << "   ______           _     _  __\n"
              << "  / ____/__  ____  (_)__ | |/ /\n"
              << " / / __/ _ \\/ __ \\/ / _ \\|   / \n"
              << "/ /_/ /  __/ / / / /  __/   |  \n"
              << "\\____/\\___/_/ /_/_/\\___/_/|_| \n"
              << "\033[0m\n";

    std::cout << "\033[1;36mLoading model from " << args.model_dir.string() << "...\033[0m\n";
    auto pipe_opt = geniex::auto_model::makePipeline(runtime_cfg, model_cfg);
    if (!pipe_opt) {
        std::cerr << "Failed to create pipeline. See logs for details.\n";
        return 1;
    }
    auto& pipe = *pipe_opt;
    std::cout << "\033[1;32mModel loaded.\033[0m\n\n";

    // Conversation history. The system turn is sent once, at the front.
    std::vector<geniex::ChatMessage> messages;
    if (!args.system_prompt.empty()) messages.push_back({geniex::Role::System, args.system_prompt});
    const size_t base_messages = messages.size();

    if (!args.prompt.empty()) {
        messages.push_back({geniex::Role::User, args.prompt});
        return runTurn(pipe, messages, args) ? 0 : 1;
    }

    while (true) {
        std::cout << "Enter your prompt (type 'exit' to quit): ";
        std::string input;
        if (!std::getline(std::cin, input)) {
            // EOF or a console read error. Say so — exiting silently here is
            // indistinguishable from a crash.
            std::cout << "\n";
            if (std::cin.eof()) {
                std::cout << "stdin closed (EOF); exiting.\n";
            } else {
                std::cerr << "stdin read error; exiting.\n";
            }
            break;
        }
        if (input == "exit" || input == "quit") break;
        if (input.empty()) continue;

        messages.push_back({geniex::Role::User, input});
        if (!runTurn(pipe, messages, args)) {
            // Drop the failed turn and start a fresh conversation.
            messages.resize(base_messages);
        }
    }
    return 0;
}
