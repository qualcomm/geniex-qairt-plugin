// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause
//
// Family-free LLM REPL. The model + chat template are loaded entirely from
// the bundle directory; no per-family header is needed.
//
// All loading and generation logic lives in auto_llm.h. This file is just
// CLI parsing + REPL plumbing.

#include "auto_llm.h"

// Same vendored JSON header core/ uses (`utils/detail/json.hpp`, re-namespaced
// to `qualla`). Deliberately NOT <nlohmann/json.hpp> — geniex-proc vendors a
// different nlohmann version, and including both trips its ABI-mismatch guard.
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "utils/detail/json.hpp"

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
    // One-shot mode: generate a single reply and exit instead of entering the
    // REPL. `prompt` is the literal text; `prompt_file` reads it from a UTF-8
    // file (avoids shell quoting/encoding damage on long or non-ASCII prompts).
    std::string prompt;
    std::string prompt_file;
    // Batch mode: a JSON file of prompts, all answered in one process so the
    // (multi-GB) bundle is loaded exactly once. KV cache is reset between
    // prompts so each answer is independent, as in one-shot mode.
    std::string batch_file;
    // When set, the run's answer + metrics are written to this path as JSON.
    // Machine-readable counterpart to --verbose, for benchmark harnesses. In
    // batch mode this is an array, rewritten after every prompt.
    std::string metrics_json;
    // Suppress the streamed answer + banners on stdout. Only meaningful with
    // --metrics-json, where the JSON file is the real output channel.
    bool quiet = false;
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
              << "  --prompt <text>             One-shot: answer this prompt and exit\n"
              << "  --prompt-file <path>        One-shot: read the prompt from a UTF-8 file\n"
              << "  --batch-file <path>         Batch: JSON [{id, prompt}, ...], model loaded once\n"
              << "  --metrics-json <path>       Write answer + TTFT/TPS/token counts as JSON\n"
              << "                              (an array, rewritten per prompt, in batch mode)\n"
              << "  --quiet                     Suppress streamed output and banners\n"
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
        else if (a == "--prompt")
            args.prompt = next();
        else if (a == "--prompt-file")
            args.prompt_file = next();
        else if (a == "--batch-file")
            args.batch_file = next();
        else if (a == "--metrics-json")
            args.metrics_json = next();
        else if (a == "--quiet")
            args.quiet = true;
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
    const int modes =
        (!args.prompt.empty() ? 1 : 0) + (!args.prompt_file.empty() ? 1 : 0) + (!args.batch_file.empty() ? 1 : 0);
    if (modes > 1) {
        std::cerr << "--prompt, --prompt-file and --batch-file are mutually exclusive\n";
        return false;
    }
    return true;
}

// Reads the whole file as raw bytes. The prompt file is expected to be UTF-8;
// we pass it through untouched so the tokenizer sees exactly what was written.
bool readPromptFile(const std::string& path, std::string& out) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        std::cerr << "cannot read --prompt-file: " << path << "\n";
        return false;
    }
    std::ostringstream buf;
    buf << in.rdbuf();
    out = buf.str();
    // Trim the trailing newline an editor or `Set-Content` inevitably adds;
    // it would otherwise land inside the user turn of the chat template.
    while (!out.empty() && (out.back() == '\n' || out.back() == '\r')) out.pop_back();
    return true;
}

// The per-run payload the benchmark harness reads: answer text plus the
// runtime's own TTFT / decode / throughput numbers (never wall-clock measured
// from outside, so process startup can't pollute the metric).
qualla::ordered_json resultToJson(const geniex::GenerateResult& r) {
    return qualla::ordered_json{
        {"answer", r.full_text},
        {"prompt_tokens", r.prompt_tokens},
        {"generated_tokens", r.generated_tokens},
        {"ttft_ms", r.ttft_ms},
        {"decode_ms", r.decode_ms},
        {"tokens_per_second", r.tokens_per_second},
        {"stop_reason", r.stop_reason},
    };
}

bool writeJsonFile(const std::string& path, const qualla::ordered_json& j) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) {
        std::cerr << "cannot write JSON to: " << path << "\n";
        return false;
    }
    out << j.dump(2) << "\n";
    return static_cast<bool>(out);
}

// A batch item read from --batch-file. `id` is opaque to us — it's echoed back
// so the harness can join results to its own prompt suite.
struct BatchItem {
    qualla::json id;
    std::string  prompt;
};

// --batch-file accepts either a bare JSON array or {"prompts": [...]}, where
// each element is {"id": <any>, "prompt": "<text>"}. A bare string element is
// also allowed, in which case the 0-based index becomes the id.
bool readBatchFile(const std::string& path, std::vector<BatchItem>& out) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        std::cerr << "cannot read --batch-file: " << path << "\n";
        return false;
    }
    qualla::json doc;
    try {
        in >> doc;
    } catch (const std::exception& e) {
        std::cerr << "--batch-file is not valid JSON: " << e.what() << "\n";
        return false;
    }
    const qualla::json* arr = &doc;
    if (doc.is_object() && doc.contains("prompts")) arr = &doc["prompts"];
    if (!arr->is_array()) {
        std::cerr << "--batch-file must be a JSON array (or an object with a \"prompts\" array)\n";
        return false;
    }
    size_t index = 0;
    for (const auto& el : *arr) {
        BatchItem item;
        if (el.is_string()) {
            item.id     = index;
            item.prompt = el.get<std::string>();
        } else if (el.is_object() && el.contains("prompt")) {
            item.id     = el.contains("id") ? el["id"] : qualla::json(index);
            item.prompt = el["prompt"].get<std::string>();
        } else {
            std::cerr << "--batch-file element " << index << " is neither a string nor {\"prompt\": ...}\n";
            return false;
        }
        ++index;
        if (item.prompt.empty()) {
            std::cerr << "--batch-file element " << (index - 1) << " has an empty prompt\n";
            return false;
        }
        out.push_back(std::move(item));
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

    // Non-interactive modes are selected by --prompt / --prompt-file (one-shot)
    // or --batch-file. Resolve the prompts up front so a bad path or malformed
    // JSON fails before the (slow) model load.
    const bool  one_shot        = !args.prompt.empty() || !args.prompt_file.empty();
    std::string one_shot_prompt = args.prompt;
    if (!args.prompt_file.empty() && !readPromptFile(args.prompt_file, one_shot_prompt)) return 1;
    if (one_shot && one_shot_prompt.empty()) {
        std::cerr << "the one-shot prompt is empty\n";
        return 1;
    }

    std::vector<BatchItem> batch;
    const bool             batch_mode = !args.batch_file.empty();
    if (batch_mode) {
        if (!readBatchFile(args.batch_file, batch)) return 1;
        if (batch.empty()) {
            std::cerr << "--batch-file contains no prompts\n";
            return 1;
        }
    }

    if (!one_shot && !batch_mode && !args.metrics_json.empty()) {
        std::cerr << "--metrics-json requires --prompt, --prompt-file or --batch-file\n";
        return 1;
    }

    if (!args.quiet) std::cout << "\033[1;36mLoading model from " << bundle_dir.string() << "\033[0m\n";

    geniex::ModelConfig model_cfg;
    try {
        model_cfg = geniex::modelConfigFromDirectory(bundle_dir);
        populateEmbeddingPathIfPresent(model_cfg, bundle_dir);
    } catch (const std::exception& e) {
        std::cerr << "Failed to read bundle: " << e.what() << "\n";
        return 1;
    }

    auto pipe_opt = geniex::auto_llm::makePipeline(geniex::QnnRuntimeConfig{}, model_cfg, args.tokenizer_config_path);
    if (!pipe_opt) {
        std::cerr << "Failed to create pipeline. See logs for details.\n";
        return 1;
    }
    auto& pipe = *pipe_opt;

    if (!args.quiet) std::cout << "\033[1;32mModel loaded.\033[0m\n\n";

    // The Pipeline's KV cache holds the prefix matching `messages`; never
    // resetKVCache() between turns.
    std::vector<geniex::ChatMessage> messages;
    if (!args.system_prompt.empty()) {
        geniex::ChatMessage sys;
        sys.role    = geniex::Role::System;
        sys.content = args.system_prompt;
        messages.push_back(std::move(sys));
    }

    geniex::GenerationConfig gen_cfg;
    gen_cfg.max_tokens    = args.max_tokens;
    gen_cfg.thinking_mode = args.enable_thinking;

    geniex::Tokenizer::ApplyChatTemplateOptions opts;
    if (args.enable_thinking) {
        opts.extra_context_json = R"({"enable_thinking":true})";
    }

    // Answers one independent prompt: a fresh [system?, user] history against a
    // cleared KV cache, so batch results match what a one-prompt-per-process
    // run would produce.
    auto answerOnce = [&](const std::string& text) {
        pipe.reset();
        std::vector<geniex::ChatMessage> turn = messages;  // system prompt, if any
        geniex::ChatMessage              user;
        user.role    = geniex::Role::User;
        user.content = text;
        turn.push_back(std::move(user));
        return pipe.generateChat(turn, gen_cfg, opts, [&](const char* piece) {
            if (!args.quiet) std::cout << piece << std::flush;
            return true;
        });
    };

    // ── One-shot: single turn, optional JSON metrics, then exit ──────────────
    if (one_shot) {
        const auto result = answerOnce(one_shot_prompt);
        if (!args.quiet) std::cout << "\n";

        // Write the metrics file even on error — the harness needs to record the
        // failed row rather than treating a missing file as a crash.
        bool wrote = true;
        if (!args.metrics_json.empty()) wrote = writeJsonFile(args.metrics_json, resultToJson(result));

        if (result.stop_reason == "error") {
            std::cerr << "generation failed (stop_reason=error). See logs for details.\n";
            return 1;
        }
        if (!wrote) return 1;
        if (!args.quiet) printPerfLine(result, args.verbose);
        return 0;
    }

    // ── Batch: every prompt in one process, model loaded once ────────────────
    if (batch_mode) {
        qualla::ordered_json results = qualla::ordered_json::array();
        size_t               failed  = 0;
        for (size_t i = 0; i < batch.size(); ++i) {
            const auto& item = batch[i];
            if (!args.quiet) {
                std::cout << "\033[1;36m[" << (i + 1) << "/" << batch.size() << "] id=" << item.id.dump()
                          << "\033[0m\n\033[33m";
            }
            const auto result = answerOnce(item.prompt);
            if (!args.quiet) std::cout << "\033[0m\n";

            auto row = resultToJson(result);
            row.push_back({"id", item.id});
            row.push_back({"prompt", item.prompt});
            results.push_back(std::move(row));
            if (result.stop_reason == "error") {
                ++failed;
                std::cerr << "prompt id=" << item.id.dump() << " failed (stop_reason=error)\n";
            }

            // Rewrite after every prompt: a long suite is expensive to lose, and
            // a killed run still leaves every completed answer on disk.
            if (!args.metrics_json.empty() && !writeJsonFile(args.metrics_json, results)) return 1;
            if (!args.quiet) printPerfLine(result, args.verbose);
        }
        if (!args.quiet) {
            std::cout << "Completed " << batch.size() << " prompts (" << failed << " failed)\n";
        }
        return failed == batch.size() ? 1 : 0;
    }

    while (true) {
        std::cout << "Enter your prompt (type 'exit' to quit): ";
        std::string input;
        if (!std::getline(std::cin, input) || input == "exit" || input == "quit") break;
        if (input.empty()) continue;

        geniex::ChatMessage user;
        user.role    = geniex::Role::User;
        user.content = input;
        messages.push_back(std::move(user));

        std::cout << "\033[33m";
        const auto result = pipe.generateChat(messages, gen_cfg, opts, [](const char* piece) {
            std::cout << piece << std::flush;
            return true;
        });
        std::cout << "\033[0m\n";

        if (result.stop_reason == "error") {
            // Drop the user turn whose generation failed and reset KV state
            // so the next turn starts clean.
            messages.pop_back();
            pipe.reset();
            continue;
        }

        geniex::ChatMessage assistant;
        assistant.role    = geniex::Role::Assistant;
        assistant.content = result.full_text;
        messages.push_back(std::move(assistant));

        printPerfLine(result, args.verbose);
    }

    return 0;
}
