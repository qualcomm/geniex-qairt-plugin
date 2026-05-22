// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#include <chrono>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "geniex-proc/tokenizer.h"
#include "qwen3_cb.h"
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
static int getTerminalWidth() {
    CONSOLE_SCREEN_BUFFER_INFO csbi;
    if (GetConsoleScreenBufferInfo(GetStdHandle(STD_OUTPUT_HANDLE), &csbi))
        return csbi.srWindow.Right - csbi.srWindow.Left + 1;
    return 120;
}
#else
#include <sys/ioctl.h>
#include <unistd.h>
static int getTerminalWidth() {
    struct winsize w;
    if (ioctl(STDOUT_FILENO, TIOCGWINSZ, &w) == 0 && w.ws_col > 0) return w.ws_col;
    return 120;
}
#endif

struct Args {
    int32_t max_tokens = 512;
    bool    verbose    = false;
};

static void printUsage(const char* prog) {
    std::cout << "Usage: " << prog << " [OPTIONS]\n"
              << "  --max-tokens <n>   Max tokens per session (default 512)\n"
              << "  --verbose          Print performance metrics\n"
              << "  --help\n";
}

static bool parseArgs(int argc, char** argv, Args& args) {
    for (int i = 1; i < argc; ++i) {
        std::string a    = argv[i];
        auto        next = [&]() -> std::string { return (i + 1 < argc) ? argv[++i] : std::string{}; };
        if (a == "--max-tokens")
            args.max_tokens = std::stoi(next());
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

static std::string applyTemplate(const std::string& user_text) {
    return "<|im_start|>user\n" + user_text + "<|im_end|>\n<|im_start|>assistant\n";
}

int main(int argc, char** argv) {
#ifdef _WIN32
    enable_utf8_io();
#endif

    Args args;
    if (!parseArgs(argc, argv, args)) return 1;

    const auto model_dir =
        std::filesystem::current_path() / "modelfiles" / "qwen3-4b-instruct-weights" / "qwen3_4b_instruct_2507";

    geniex::QnnRuntimeConfig runtime_cfg;

    geniex::ModelConfig model_cfg;
    model_cfg.model_paths = {
        (model_dir / "qwen3_4b_instruct_2507_part_1_of_4.bin").string(),
        (model_dir / "qwen3_4b_instruct_2507_part_2_of_4.bin").string(),
        (model_dir / "qwen3_4b_instruct_2507_part_3_of_4.bin").string(),
        (model_dir / "qwen3_4b_instruct_2507_part_4_of_4.bin").string(),
    };
    model_cfg.tokenizer_path  = (model_dir / "tokenizer.json").string();
    model_cfg.htp_config_path = (model_dir / "htp_backend_ext_config.json").string();

    std::cout << "\033[1;32m"
              << "   ______           _     _  __\n"
              << "  / ____/__  ____  (_)__ | |/ /\n"
              << " / / __/ _ \\/ __ \\/ / _ \\|   / \n"
              << "/ /_/ /  __/ / / / /  __/   |  \n"
              << "\\____/\\___/_/ /_/_/\\___/_/|_| \n"
              << "\033[0m"
              << "\033[1;33m  Continuous Batching\033[0m\n\n";

    std::cout << "\033[1;36mLoading model...\033[0m\n";
    auto model = geniex::qwen3_cb::makeModel(model_cfg);
    try {
        if (!model.initialize(runtime_cfg, model_cfg)) {
            std::cerr << "Failed to initialize model.\n";
            return 1;
        }
    } catch (const std::exception& e) {
        std::cerr << "Model initialization error: " << e.what() << "\n";
        return 1;
    }
    std::cout << "\033[1;32mModel loaded.\033[0m\n\n";

    auto tokenizer = geniex::Tokenizer::from_file(model_cfg.tokenizer_path);

    std::cout << "=== Continuous Batching Mode ===\n"
              << "Enter multiple prompts (one per line). Type 'go' to start generation.\n"
              << "Type 'exit' to quit.\n\n";

    std::vector<std::pair<std::string, std::vector<int32_t>>> pending;
    int                                                       session_counter = 0;

    while (true) {
        pending.clear();
        model.resetKVCache();

        while (true) {
            std::cout << "Prompt " << (pending.size() + 1) << " (or 'go'/'exit'): ";
            std::string input;
            if (!std::getline(std::cin, input)) return 0;
            if (input == "exit" || input == "quit") return 0;
            if (input == "go") break;
            if (input.empty()) continue;

            auto encoded = tokenizer->encode(applyTemplate(input));
            pending.push_back({input, {encoded.begin(), encoded.end()}});
        }

        if (pending.empty()) continue;

        geniex::cb::Scheduler      scheduler;
        geniex::cb::KVCacheManager kv_mgr;
        std::vector<std::string>   session_ids;
        for (size_t i = 0; i < pending.size(); ++i) {
            std::string sid = "s" + std::to_string(session_counter++);
            session_ids.push_back(sid);
            scheduler.addSession(sid, pending[i].second, args.max_tokens);
        }

        std::cout << "\n\033[1;36m--- Starting " << pending.size() << " sessions ---\033[0m\n";
        for (size_t i = 0; i < pending.size(); ++i) {
            std::cout << "  [" << session_ids[i] << "] \"" << pending[i].first << "\" (" << pending[i].second.size()
                      << " tokens)\n";
        }
        std::cout << "\n";

        // Streaming display: reserve a fixed line budget per session so the
        // area never grows and wrapped output doesn't trigger reflows.
        const size_t n_sessions          = pending.size();
        const int    term_width          = getTerminalWidth();
        const int    prefix_len          = 5;  // "[sN] " label width
        const int    text_width          = std::max(term_width - prefix_len, 20);
        const int    lines_per_session   = 4;  // pre-allocated lines per session
        const int    total_display_lines = static_cast<int>(n_sessions) * (lines_per_session + 1);  // +1 for label line

        std::vector<std::string> output_text(n_sessions);

        for (size_t i = 0; i < n_sessions; ++i) {
            std::cout << "\033[1;33m[" << session_ids[i] << "]\033[0m\n";
            for (int l = 0; l < lines_per_session; ++l) std::cout << "\n";
        }
        std::cout << std::flush;

        // Split text into rows, honouring both embedded newlines and the
        // terminal width.
        auto textToRows = [&](const std::string& text) -> std::vector<std::string> {
            std::vector<std::string> rows;
            size_t                   pos = 0;
            while (pos < text.size()) {
                size_t      nl           = text.find('\n', pos);
                std::string logical_line = (nl == std::string::npos) ? text.substr(pos) : text.substr(pos, nl - pos);
                pos                      = (nl == std::string::npos) ? text.size() : nl + 1;

                if (logical_line.empty()) {
                    rows.push_back("");
                    continue;
                }
                size_t lp = 0;
                while (lp < logical_line.size()) {
                    size_t chunk = std::min(static_cast<size_t>(text_width), logical_line.size() - lp);
                    rows.push_back(logical_line.substr(lp, chunk));
                    lp += chunk;
                }
            }
            return rows;
        };

        // Redraw the whole reserved area from the current text. Shows only
        // the last `lines_per_session` rows so output scrolls within the
        // fixed area instead of pushing it down.
        auto redraw = [&]() {
            std::cout << "\033[" << total_display_lines << "A";

            for (size_t i = 0; i < n_sessions; ++i) {
                std::cout << "\r\033[K\033[1;33m[" << session_ids[i] << "]\033[0m\n";

                auto   rows          = textToRows(output_text[i]);
                size_t start_row     = (rows.size() > static_cast<size_t>(lines_per_session))
                                           ? rows.size() - static_cast<size_t>(lines_per_session)
                                           : 0;
                int    printed_lines = 0;
                for (size_t r = start_row; r < rows.size() && printed_lines < lines_per_session; ++r) {
                    std::cout << "\r\033[K  " << rows[r] << "\n";
                    ++printed_lines;
                }
                for (; printed_lines < lines_per_session; ++printed_lines) std::cout << "\r\033[K\n";
            }
            std::cout << std::flush;
        };

        const auto t_start = std::chrono::high_resolution_clock::now();

        try {
            model.generateBatch(scheduler, kv_mgr, [&](const std::string& sid, int32_t tok) {
                size_t idx = 0;
                for (; idx < session_ids.size(); ++idx)
                    if (session_ids[idx] == sid) break;
                if (idx >= session_ids.size()) return;

                output_text[idx] += tokenizer->decode_token(tok);
                redraw();
            });
        } catch (const std::exception& e) {
            std::cerr << "\nGeneration error: " << e.what() << "\n";
            continue;
        }

        const auto t_end = std::chrono::high_resolution_clock::now();

        redraw();

        std::cout << "\n\033[1;36m--- Results ---\033[0m\n";
        size_t total_gen = 0;
        for (size_t i = 0; i < pending.size(); ++i) {
            auto*  s   = scheduler.getSession(session_ids[i]);
            size_t gen = s ? s->generated_tokens.size() : 0;
            total_gen += gen;
            std::cout << "\033[1;33m[" << session_ids[i] << "]\033[0m " << gen << " tokens\n";
        }

        if (args.verbose) {
            const double elapsed_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
            const double tps        = elapsed_ms > 0 ? total_gen / (elapsed_ms / 1000.0) : 0;
            std::cout << "\n\033[1;36m=== Performance ===\033[0m\n"
                      << "Sessions       : " << pending.size() << "\n"
                      << "Total tokens   : " << total_gen << "\n"
                      << "Elapsed        : " << std::fixed << std::setprecision(1) << elapsed_ms << " ms\n"
                      << "Throughput     : " << std::setprecision(2) << tps << " tokens/s\n"
                      << "===================\n";
        }
        std::cout << "\n";
    }

    return 0;
}
