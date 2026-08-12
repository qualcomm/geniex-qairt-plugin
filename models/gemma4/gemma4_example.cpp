// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause
//
// Gemma4 (E2B / E4B) inference — text-only and image + text in one binary.
//
// Every `gemma_4_*` bundle ships a vision encoder context binary, so the family
// is always multimodal and this example always builds a VLMPipeline — the same
// path the dispatcher, the CLI, geniex-bench and the REST server take. A
// text-only prompt is just a turn with no attachment.
//
//   text only:     gemma4_e4b --model-dir <bundle> --prompt "hi" --chat
//   image + text:  gemma4_e4b --model-dir <bundle> --prompt "describe it" --image cat.jpg
//   interactive:   gemma4_e4b --model-dir <bundle>
//                  > describe this picture /path/to/cat.jpg
//
// The image span is produced by the processor from the chat template's
// mm_content, so no image marker is ever assembled by hand here — doing that in
// an earlier version of this example hid a real bug in
// Gemma4Processor::apply_chat_template().

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "gemma4/gemma4.h"
#include "geniex-proc/types.h"
#include "types.h"

// Which Gemma4 variant this executable defaults to. E2B and E4B differ only in
// dimensions (hidden 1536 vs 2560, 35 vs 42 layers, per-layer stream 8960 vs
// 10752, 1 vs 2 KV heads) — every one of those is read from the bundle's
// genie_config.json or inferred from the loaded graphs, so the two targets share
// this source verbatim. Set by CMake; see models/gemma4/CMakeLists.txt.
#ifndef GEMMA4_VARIANT
#define GEMMA4_VARIANT "gemma4_e2b"
#endif
#ifndef GEMMA4_VARIANT_LABEL
#define GEMMA4_VARIANT_LABEL "Gemma4-E2B"
#endif

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
    fs::path                 model_dir;
    fs::path                 mmproj;  // vision encoder override; default <bundle>/vision_encoder.bin
    std::string              prompt;  // non-empty → single-shot, non-interactive
    std::vector<std::string> images;  // --image (repeatable)
    std::vector<std::string> turns;   // --turn (repeatable) → scripted multi-round
    int32_t                  max_tokens = 256;
    bool                     verbose    = false;
    bool                     chat       = false;  // apply chat template (instruct models)
    // Sampling — greedy when temperature <= 0
    float    temperature = 0.0f;
    float    top_p       = 0.95f;
    int32_t  top_k       = 40;
    uint32_t seed        = 42;
};

void printUsage(const char* prog) {
    std::cout << "Usage: " << prog << " [OPTIONS]\n"
              << "  --model-dir <path> QAIRT bundle directory (default: ./modelfiles/" GEMMA4_VARIANT ")\n"
              << "  --prompt <text>    Run once on this prompt and exit (non-interactive)\n"
              << "  --image <path>     Attach an image; repeat for several. Implies --chat\n"
              << "  --mmproj <path>    Vision encoder context binary\n"
              << "                     (default: <model-dir>/vision_encoder.bin)\n"
              << "  --turn <text>      One user turn of a scripted conversation; repeat to\n"
              << "                     script multiple rounds (implies --chat). KV is kept\n"
              << "                     across turns, so each round only prefills its own text.\n"
              << "  --max-tokens <n>   Max tokens to generate (default 256)\n"
              << "  --chat             Apply the chat template (instruct-tuned checkpoints)\n"
              << "  --temperature <f>  Sampling temperature; >0 enables sampling (default 0 = greedy)\n"
              << "  --top-p <f>        Top-p nucleus sampling (default 0.95)\n"
              << "  --top-k <n>        Top-k sampling (default 40)\n"
              << "  --seed <n>         RNG seed for sampling (default 42)\n"
              << "  --verbose          Print performance metrics\n"
              << "  --help\n"
              << "\n"
              << "Interactively, include an image path anywhere in the line:\n"
              << "  > describe this picture /path/to/cat.jpg\n";
}

bool parseArgs(int argc, char** argv, Args& args) {
    for (int i = 1; i < argc; ++i) {
        std::string a    = argv[i];
        auto        next = [&]() -> std::string { return (i + 1 < argc) ? argv[++i] : std::string{}; };
        if (a == "--model-dir")
            args.model_dir = next();
        else if (a == "--mmproj" || a == "--mmproj-path")
            args.mmproj = next();
        else if (a == "--prompt")
            args.prompt = next();
        else if (a == "--image")
            args.images.push_back(next());
        else if (a == "--turn")
            args.turns.push_back(next());
        else if (a == "--max-tokens")
            args.max_tokens = std::stoi(next());
        else if (a == "--temperature")
            args.temperature = std::stof(next());
        else if (a == "--top-p")
            args.top_p = std::stof(next());
        else if (a == "--top-k")
            args.top_k = std::stoi(next());
        else if (a == "--seed")
            args.seed = static_cast<uint32_t>(std::stoul(next()));
        else if (a == "--chat")
            args.chat = true;
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
    if (args.model_dir.empty()) args.model_dir = fs::current_path() / "modelfiles" / GEMMA4_VARIANT;
    // The chat template is what places the image span, so an attachment needs it.
    if (!args.images.empty()) args.chat = true;
    return true;
}

geniex::GenerationConfig makeGenConfig(const Args& args) {
    geniex::GenerationConfig gen_cfg;
    gen_cfg.max_tokens = args.max_tokens;
    if (args.temperature > 0.0f) {
        gen_cfg.enable_sampling = true;
        gen_cfg.temperature     = args.temperature;
        gen_cfg.top_p           = args.top_p;
        gen_cfg.top_k           = args.top_k;
        gen_cfg.seed            = args.seed;
    }
    return gen_cfg;
}

void printMetrics(const geniex::GenerateResult& result, const Args& args) {
    if (args.verbose) {
        std::cout << "\033[1;36m=== Performance ===\033[0m\n"
                  << "Prompt tokens    : " << result.prompt_tokens << "\n"
                  << "Generated tokens : " << result.generated_tokens << "\n"
                  << "TTFT             : " << std::fixed << std::setprecision(1) << result.ttft_ms << " ms\n"
                  << "Decode speed     : " << std::fixed << std::setprecision(2) << result.tokens_per_second
                  << " tokens/s\n"
                  << "Stop reason      : " << result.stop_reason << "\n"
                  << "===================\n\n";
    } else {
        std::cout << "TTFT: " << std::fixed << std::setprecision(1) << result.ttft_ms << " ms"
                  << "  |  " << std::setprecision(2) << result.tokens_per_second << " tokens/s\n\n";
    }
}

bool isImageFile(const std::string& path) {
    std::string p = path;
    std::transform(p.begin(), p.end(), p.begin(), [](unsigned char c) { return static_cast<char>(::tolower(c)); });
    auto ends_with = [&p](const char* suffix) {
        const std::string s(suffix);
        return p.size() >= s.size() && p.compare(p.size() - s.size(), s.size(), s) == 0;
    };
    return ends_with(".jpg") || ends_with(".jpeg") || ends_with(".png") || ends_with(".bmp") || ends_with(".gif") ||
           ends_with(".webp");
}

// Splits an interactive line into text and any image paths it mentions.
void parseInput(const std::string& input, std::string& prompt_text, std::vector<std::string>& image_paths) {
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

bool streamToken(const char* piece) {
    std::cout << piece << std::flush;
    return true;
}

// Gemma4 turn framing for CONTINUATION turns.
//
// The chat template renders the FIRST turn (it also emits <bos>, which the
// pipeline would otherwise prepend itself). Continuation turns must not repeat
// <bos> or replay the history: the KV cache already holds it, so we only feed
// the delta — close the model's turn, then open the next user/model pair. Genie
// has to re-send the whole transcript every round because its "basic" dialog is
// stateless; keeping KV is why round 2 here prefills ~20 tokens instead of ~336.
//
// A continuation turn may attach its own images. Because we bypass the template
// here, we must emit the processor's image marker ourselves — one per image,
// ahead of the text, which is where the template puts them. generate() pairs the
// i-th marker with image_paths[i], and the processor expands each into that
// image's boi / soft-token run / eoi span.
std::string continuationPrompt(
    const std::string& user_text, const std::vector<std::string>& images, const std::string& image_marker) {
    std::string markers;
    for (size_t i = 0; i < images.size(); ++i) markers += image_marker + "\n";
    return "<turn|>\n<|turn>user\n" + markers + user_text + "<turn|>\n<|turn>model\n";
}

// ── Turn ────────────────────────────────────────────────────────────────────

void runTurn(geniex::VLMPipeline& pipe, const std::string& user_text, const std::vector<std::string>& images,
    const Args& args, bool first_turn) {
    std::string prompt = user_text;
    // An attachment always needs the template: it is what places the image span.
    if (args.chat || !images.empty()) {
        // The processor turns each mm_content entry into an image marker and then
        // expands that marker into the boi / soft-token run / eoi span, so the
        // image span is never assembled by hand here.
        geniex::ChatMessage user_msg{geniex::Role::User, user_text};
        for (const auto& img : images) user_msg.mm_content.push_back({geniex::Modality::Image, img});
        try {
            prompt = first_turn ? pipe.applyChatTemplate({std::move(user_msg)})
                                : continuationPrompt(user_text, images, pipe.imageMarker());
        } catch (const std::exception& e) {
            std::cerr << "Chat-template error: " << e.what() << "\n";
            return;
        }
    }

    std::cout << "\033[33m";
    const auto result = pipe.generate(prompt, images, makeGenConfig(args), streamToken);
    std::cout << "\033[0m\n";
    printMetrics(result, args);
}

}  // namespace

int main(int argc, char** argv) {
#ifdef _WIN32
    enable_utf8_io();
#endif

    Args args;
    if (!parseArgs(argc, argv, args)) return 1;

    if (!fs::exists(args.model_dir)) {
        std::cerr << "Model directory not found: " << args.model_dir.string() << "\n";
        return 1;
    }

    geniex::QnnRuntimeConfig runtime_cfg;

    geniex::ModelConfig model_cfg;
    try {
        model_cfg = geniex::modelConfigFromDirectory(args.model_dir);
    } catch (const std::exception& e) {
        std::cerr << "Failed to read bundle: " << e.what() << "\n";
        return 1;
    }

    // Every gemma_4_* bundle ships a vision encoder, so this is a hard
    // requirement rather than a mode switch.
    const fs::path veg = args.mmproj.empty() ? args.model_dir / "vision_encoder.bin" : args.mmproj;
    if (!fs::exists(veg)) {
        std::cerr << "No vision encoder found at " << veg.string()
                  << ".\nEvery Gemma4 bundle ships one; pass --mmproj <path> if it lives elsewhere.\n";
        return 1;
    }

    std::cout << "\033[1;36mLoading " GEMMA4_VARIANT_LABEL " from " << args.model_dir.string() << "...\033[0m\n";

    geniex::VLMConfig config;
    config.llm_config                    = model_cfg;
    config.vision_config.model_paths     = {veg.string()};
    config.vision_config.htp_config_path = model_cfg.htp_config_path;

    auto pipe = geniex::gemma4::makeVLMPipeline(runtime_cfg, config);
    if (!pipe) {
        std::cerr << "Failed to create the pipeline. See logs for details.\n";
        return 1;
    }
    std::cout << "\033[1;32mModel loaded.\033[0m\n\n";

    if (!args.turns.empty()) {
        // Scripted conversation: one KV cache carried across every round.
        //
        // A turn may carry its own image by naming the file inside the --turn
        // text (same convention as the interactive prompt), so a conversation can
        // attach a different image each round. Any --image flags attach to the
        // first turn, matching --prompt --image.
        args.chat = true;
        for (size_t i = 0; i < args.turns.size(); ++i) {
            std::string              text;
            std::vector<std::string> images;
            parseInput(args.turns[i], text, images);
            if (i == 0) images.insert(images.begin(), args.images.begin(), args.images.end());

            std::cout << "\033[1;35m=== Round " << (i + 1) << " — user: " << text;
            for (const auto& img : images) std::cout << " [" << img << "]";
            std::cout << "\033[0m\n";
            runTurn(*pipe, text, images, args, /*first=*/i == 0);
        }
        return 0;
    }

    if (!args.prompt.empty()) {
        runTurn(*pipe, args.prompt, args.images, args, /*first=*/true);
        return 0;
    }

    // Interactive: also a conversation, so KV is kept and only the first turn
    // goes through the chat template.
    bool first = true;
    while (true) {
        std::cout << "Enter your prompt (type 'exit' to quit, 'reset' to clear history): ";
        std::string input;
        if (!std::getline(std::cin, input) || input == "exit" || input == "quit") break;
        if (input == "reset") {
            pipe->reset();
            first = true;
            std::cout << "\033[1;35m[history cleared]\033[0m\n";
            continue;
        }

        std::string              text;
        std::vector<std::string> images;
        parseInput(input, text, images);
        runTurn(*pipe, text, images, args, first);
        first = false;
    }
    return 0;
}
