// Copyright (c) 2026 Nexa AI. SPDX-License-Identifier: BSD-3-Clause
//
// Gemma4 E2B streaming demo on the NPU (X Elite / HTP v73).
//
// Drives the gemma4 prefill context binary directly (72 numbered inputs, see
// gemma4.h), doing CPU-side embedding + PLE lookup, dual RoPE, dual masks, and
// an autoregressive greedy/sampled decode loop that streams tokens as they are
// produced. Uses geniex Model (graph loading), Tokenizer, and Sampler.
//
// Because the demo binary is a single AR=S prefill graph (no AR=1 decode
// graph), each decode step re-runs the AR=S graph with the running token
// sequence (n_past=0, empty/zero KV, causal mask). Correct but O(S) per token.
//
// Usage:
//   gemma4_e2b_example --bundle <dir> --ctx <gemma4_ctx_v73.bin> \
//       --prompt "What is the capital of France?" [--max-tokens 32] [--temp 0]

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "gemma4/gemma4.h"
#include "geniex-proc/sampler.h"
#include "geniex-proc/tokenizer.h"
#include "types.h"

#ifdef _WIN32
#include <windows.h>
static void enable_utf8_io() {
    SetConsoleOutputCP(CP_UTF8);
    HANDLE hOut = GetStdHandle(STD_OUTPUT_HANDLE);
    DWORD  mode = 0;
    if (GetConsoleMode(hOut, &mode)) SetConsoleMode(hOut, mode | ENABLE_VIRTUAL_TERMINAL_PROCESSING);
}
#endif

namespace fs = std::filesystem;
using geniex::gemma4::Bf16Table;
using geniex::gemma4::Gemma4Arch;
using geniex::gemma4::Gemma4Model;
using geniex::gemma4::InputBuilder;

static int run(int argc, char** argv);

int main(int argc, char** argv) {
#ifdef _WIN32
    enable_utf8_io();
#endif
    try {
        return run(argc, argv);
    } catch (const std::exception& e) {
        std::cerr << "\n[FATAL] " << e.what() << std::endl;
        return 2;
    } catch (...) {
        std::cerr << "\n[FATAL] unknown exception" << std::endl;
        return 3;
    }
}

static int run(int argc, char** argv) {
    fs::path    bundle, ctx;
    std::string prompt = "What is the capital of France?";
    int         max_tokens = 32;
    float       temp = 0.0f;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto        nx = [&] { return (i + 1 < argc) ? argv[++i] : ""; };
        if (a == "--bundle") bundle = nx();
        else if (a == "--ctx") ctx = nx();
        else if (a == "--prompt") prompt = nx();
        else if (a == "--max-tokens") max_tokens = std::stoi(nx());
        else if (a == "--temp") temp = std::stof(nx());
    }
    if (bundle.empty() || ctx.empty()) {
        std::cerr << "usage: --bundle <dir> --ctx <ctx.bin> [--prompt ...] [--max-tokens N] [--temp T]\n";
        return 1;
    }

    std::cerr << "[dbg] args parsed; bundle=" << bundle << " ctx=" << ctx << std::endl;
    Gemma4Arch arch;
    // Must match the compiled graph. ar128-cl4096 production graph: S=128,
    // full-attn KV=4096-128=3968, sliding-attn KV=512-128=384. (Toy graph was 128/128/128.)
    const int  S = 128, KV_FULL = 3968, KV_SLIDE = 384;

    // ---- load embedding tables (bf16 raw) ----
    std::cerr << "[dbg] loading embed table ("
              << (double)arch.vocab * arch.hidden * 2 / 1e9 << " GB) + PLE ("
              << (double)arch.vocab * arch.num_layers * arch.ple_dim * 2 / 1e9 << " GB)..." << std::endl;
    std::cout << "[load] embedding tables...\n";
    Bf16Table embed, ple;
    embed.load((bundle / "embed_tokens_e2b.bf16.bin").string(), arch.vocab, arch.hidden);
    ple.load((bundle / "embed_tokens_per_layer_e2b.bf16.bin").string(), arch.vocab, arch.num_layers * arch.ple_dim);
    InputBuilder ib(arch, S, KV_FULL, KV_SLIDE, &embed, &ple);
    std::cerr << "[dbg] tables loaded" << std::endl;

    // ---- tokenizer + chat template ----
    auto tok = geniex::Tokenizer::from_file((bundle / "tokenizer.json").string(),
                                            (bundle / "tokenizer_config.json").string());
    if (!tok) { std::cerr << "tokenizer load failed\n"; return 1; }
    // gemma4 ships its chat template in a separate chat_template.jinja (not in
    // tokenizer_config.json), and it's a 386-line macro-heavy template. Build
    // the turn frame directly from known special tokens instead:
    //   <bos> <|turn> user \n {prompt} <turn|> \n <|turn> model \n
    // (<bos>=2, <|turn>=105, <turn|>=106). Verified to reproduce the reference
    // 16-token France prompt and next-token 818 ("The").
    const int32_t BOS = 2, TURN_OPEN = 105, TURN_CLOSE = 106;
    std::vector<int32_t> ids;
    auto append = [&](const std::vector<int32_t>& v) { ids.insert(ids.end(), v.begin(), v.end()); };
    ids.push_back(BOS);
    ids.push_back(TURN_OPEN);
    append(tok->encode("user\n", false));
    append(tok->encode(prompt, false));
    ids.push_back(TURN_CLOSE);
    append(tok->encode("\n", false));
    ids.push_back(TURN_OPEN);
    append(tok->encode("model\n", false));
    std::cout << "[tok] prompt framed to " << ids.size() << " tokens: ";
    for (int32_t t : ids) std::cout << t << " ";
    std::cout << "\n";
    if ((int)ids.size() > S) { std::cerr << "prompt too long for S=" << S << "\n"; return 1; }

    // ---- load context binary ----
    std::cout << "[load] context binary " << ctx.string() << " (NPU)...\n";
    geniex::QnnRuntimeConfig rt;  // auto-detect HTP paths from htp-files/
    geniex::ModelConfig      mc;
    mc.model_paths   = {ctx.string()};
    mc.tokenizer_path = (bundle / "tokenizer.json").string();
    Gemma4Model model;
    if (!model.initialize(rt, mc)) { std::cerr << "context binary load failed\n"; return 1; }
    // The context binary may hold multiple graphs (e.g. gemma4_ar128 prefill +
    // gemma4_ar1 decode when weight-shared). This prefill-rerun driver uses the
    // AR=S graph; pick it by name (== "gemma4_ar128" or the sole graph) rather
    // than assuming index 0.
    size_t gi = 0;
    for (size_t k = 0; k < model.graphCount(); ++k) {
        std::cout << "[load] found graph[" << k << "] '" << model.graph(k).name() << "'\n";
        if (model.graph(k).name().find("ar128") != std::string::npos) gi = k;
    }
    geniex::Graph& g = model.graph(gi);
    std::cout << "[load] using graph '" << g.name() << "': " << g.inputSpecs().size() << " inputs, "
              << g.outputSpecs().size() << " outputs\n";

    // ---- sampler (greedy if temp<=0) ----
    geniex_sampler_params sp;
    sp.temp = temp;
    sp.no_perf = true;
    for (int e : {1, 106, 50}) sp.eog_tokens.push_back(e);  // gemma4 eos ids
    geniex::Sampler sampler(sp);
    const bool greedy = (temp <= 0.0f);

    // ---- helper: write graph inputs by ROLE NAME for the running token seq ----
    // Graph I/O uses role names (export_onnx.py): input_embeds, per_layer_input_{i},
    // cos/sin_global/local, attention_mask_full/slide, past_key_{i}_in/past_value_{i}_in;
    // output "logits". KV inputs are zero-filled (n_past=0 prefill; mask masks the cache).
    // Optional: dump each CPU-built input tensor to <GEMMA4_DUMP>/<name>.raw
    // (float32) for byte comparison against the python reference raws. Set env
    // GEMMA4_DUMP=<dir> to enable. Used to isolate InputBuilder bugs.
    const char* dump_dir = std::getenv("GEMMA4_DUMP");
    auto dump = [&](const std::string& nm, const std::vector<float>& v) {
        if (!dump_dir) return;
        std::ofstream f((fs::path(dump_dir) / (nm + ".raw")).string(), std::ios::binary);
        f.write(reinterpret_cast<const char*>(v.data()), (std::streamsize)(v.size() * sizeof(float)));
    };

    auto writeInputs = [&](const std::vector<int32_t>& seq) {
        const int curr = (int)seq.size();
        auto embeds = ib.buildEmbeds(seq);
        g.write("input_embeds", embeds.data(), embeds.size());
        dump("input_embeds", embeds);
        auto per = ib.buildPerLayer(seq);
        for (int l = 0; l < arch.num_layers; ++l) {
            std::string nm = "per_layer_input_" + std::to_string(l);
            if (g.hasInput(nm)) g.write(nm, per[l].data(), per[l].size());
            if (l == 0) dump("per_layer_input_0", per[l]);
        }
        std::vector<float> cg, sg, cl, sl;
        ib.buildRope(/*sliding=*/false, seq, curr, cg, sg);
        ib.buildRope(/*sliding=*/true, seq, curr, cl, sl);
        if (g.hasInput("cos_global")) g.write("cos_global", cg.data(), cg.size());
        if (g.hasInput("sin_global")) g.write("sin_global", sg.data(), sg.size());
        if (g.hasInput("cos_local")) g.write("cos_local", cl.data(), cl.size());
        if (g.hasInput("sin_local")) g.write("sin_local", sl.data(), sl.size());
        dump("cos_global", cg); dump("sin_global", sg); dump("cos_local", cl); dump("sin_local", sl);
        auto mf = ib.buildMask(/*sliding=*/false, curr);
        auto ms = ib.buildMask(/*sliding=*/true, curr);
        if (g.hasInput("attention_mask_full")) g.write("attention_mask_full", mf.data(), mf.size());
        if (g.hasInput("attention_mask_slide")) g.write("attention_mask_slide", ms.data(), ms.size());
        dump("attention_mask_full", mf); dump("attention_mask_slide", ms);
        // KV caches: zero-filled to this graph's actual per-tensor size.
        for (const auto& spec : g.inputSpecs()) {
            if (spec.name.rfind("past_key_", 0) != 0 && spec.name.rfind("past_value_", 0) != 0) continue;
            std::vector<float> z(spec.elementCount(), 0.0f);
            g.write(spec.name, z.data(), z.size());
        }
    };

    // ---- decode loop (re-run prefill graph per step) ----
    std::cout << "\n\033[33m";
    std::vector<int32_t> gen;
    std::map<std::string, std::pair<double, uint16_t>> tl;
    std::vector<float>   logits(arch.vocab);
    auto                 t0 = std::chrono::steady_clock::now();
    double               ttft_ms = 0;
    for (int step = 0; step < max_tokens; ++step) {
        std::vector<int32_t> seq = ids;
        seq.insert(seq.end(), gen.begin(), gen.end());
        if ((int)seq.size() >= S) { std::cout << "\n[reached S limit]\n"; break; }
        writeInputs(seq);
        if (!g.execute(tl)) { std::cerr << "\ngraph execute failed\n"; break; }
        // read logits at the last real position
        const int last = (int)seq.size() - 1;
        g.read("logits", logits.data(), logits.size(), (size_t)last * arch.vocab);
        int32_t next = greedy ? (int32_t)(std::max_element(logits.begin(), logits.end()) - logits.begin())
                              : sampler.sample(logits);
        if (step == 0) ttft_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        // Stop on gemma4 end-of-turn (<turn|>=106), <eos>=1, or tokenizer eog.
        if (next == 106 || next == 1 || tok->is_eog(next)) { std::cout << "\n[eot]\n"; break; }
        // Stream the delta: decode the full generated seq and print only the new
        // tail. This applies SentencePiece '▁'→space rendering (decode_token's
        // stream-safe raw-byte mode leaves the '▁' marker visible).
        std::string prev = tok->decode(gen);
        gen.push_back(next);
        std::string cur  = tok->decode(gen);
        std::cout << cur.substr(std::min(prev.size(), cur.size())) << std::flush;
    }
    std::cout << "\033[0m\n";
    double total_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    std::cout << "\n[done] generated " << gen.size() << " tokens | TTFT " << (int)ttft_ms << " ms | "
              << (gen.empty() ? 0.0 : (gen.size() * 1000.0 / total_ms)) << " tok/s\n";
    std::cout << "[full] " << tok->decode(gen) << "\n";
    return 0;
}
