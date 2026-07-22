// Copyright (c) 2026 Nexa AI. SPDX-License-Identifier: BSD-3-Clause
//
// Gemma4 E2B streaming on the NPU with a REAL AR=1 decode graph.
//
// Unlike gemma4_e2b_example.cpp (which re-runs the AR=128 prefill graph every
// step, O(128*len)/token), this driver uses the two-graph weight-shared context
// binary (gemma4_ar128 prefill + gemma4_ar1 decode):
//   1. PREFILL once: run gemma4_ar128 on the padded prompt (n_past=0, zero KV),
//      read logits at the last real position, sample token #1, and capture the
//      prefill KV outputs.
//   2. Seed a running KV cache from those prefill KV outputs.
//   3. DECODE loop: run gemma4_ar1 on one token at position n_past, feeding the
//      running KV cache; read the single logits row, sample, append the new KV
//      into the cache, n_past++, repeat.
//
// This does 1 query row/step against cached KV instead of recomputing 128 — the
// tok/s lever. On X Elite v73 the decode sequence matches the prefill-rerun
// driver + HF float ("The capital of France is Paris.") at ~5 tok/s vs ~1 tok/s.
//
// KV requant note: seeding/appending the cache requantizes float KV into the
// decode graph's int16 KV-in. Graph::write's uFixed16 path TRUNCATES toward zero
// (deliberately, to bit-match Genie's RoPE tensors), but qnn-net-run — and the
// HTP itself — ROUND. On this argmax-sensitive KV graph that ~0.5-LSB bias is
// enough to flip the decoded token, so KV-in is written via quantizeRoundU16
// (round-to-nearest) instead. See writeKVIn.
//
// Usage:
//   gemma4_e2b_decode --bundle <dir> --ctx <2graph.bin> \
//       --prompt "What is the capital of France?" [--max-tokens 32] [--temp 0]

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "gemma4/gemma4.h"
#include "gemma4/gemma4_decode.h"
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
using geniex::gemma4::DecodeInputBuilder;
using geniex::gemma4::Gemma4Arch;
using geniex::gemma4::Gemma4Model;
using geniex::gemma4::InputBuilder;
using geniex::gemma4::KvCache;

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
    }
}

// Find a graph by EXACT name; -1 if absent. (Substring match is unsafe here:
// "ar1" is a substring of "ar128".)
static int findGraph(Gemma4Model& m, const std::string& exact) {
    for (size_t k = 0; k < m.graphCount(); ++k)
        if (m.graph(k).name() == exact) return (int)k;
    return -1;
}

// Quantize float -> uFixed16 with ROUND-TO-NEAREST, matching the QNN SDK's
// datautil::floatToTfN (std::round) and the on-device HTP quantize. geniex's
// Graph::write uFixed16 path truncates toward zero (to bit-match Genie's RoPE
// tensors); on this argmax-sensitive KV graph that ~0.5-LSB bias flips the
// decoded token, so KV-in is quantized here and raw-copied instead.
static void quantizeRoundU16(uint16_t* out, const float* in, int32_t offset, float scale, size_t n) {
    const double max_val = 65535.0;
    const double enc_min = (double)offset * (double)scale;
    const double range   = max_val * (double)scale;  // = enc_max - enc_min
    for (size_t i = 0; i < n; ++i) {
        double q = std::round(max_val * ((double)in[i] - enc_min) / range);
        if (q < 0.0) q = 0.0;
        else if (q > max_val) q = max_val;
        out[i] = (uint16_t)q;
    }
}

// uFixed8 twin of quantizeRoundU16, for int8 KV-cache I/O. On the coarser 8-bit
// grid the truncate-vs-round bias (Graph::write truncates; qnn-net-run + HTP round)
// is a larger fraction of an LSB, so round-to-nearest matters even more here.
static void quantizeRoundU8(uint8_t* out, const float* in, int32_t offset, float scale, size_t n) {
    const double max_val = 255.0;
    const double enc_min = (double)offset * (double)scale;
    const double range   = max_val * (double)scale;  // = enc_max - enc_min
    for (size_t i = 0; i < n; ++i) {
        double q = std::round(max_val * ((double)in[i] - enc_min) / range);
        if (q < 0.0) q = 0.0;
        else if (q > max_val) q = max_val;
        out[i] = (uint8_t)q;
    }
}

// Write a KV-in tensor from host float storage. Per-tensor uFixed16/uFixed8 KV-in
// is quantized round-to-nearest (quantizeRoundU16/U8) and raw-copied, bypassing
// Graph::write's truncating float path; anything else defers to Graph::write.
static void writeKVIn(geniex::Graph& g, const std::string& name, const std::vector<float>& v) {
    const auto& spec = g.inputSpec(name);
    if (spec.dtype == QNN_DATATYPE_UFIXED_POINT_16 && spec.axis_quant.empty()) {
        std::vector<uint16_t> q(v.size());
        quantizeRoundU16(q.data(), v.data(), spec.quant_offset, spec.quant_scale, v.size());
        g.write(name, static_cast<const void*>(q.data()), q.size() * sizeof(uint16_t));
    } else if (spec.dtype == QNN_DATATYPE_UFIXED_POINT_8 && spec.axis_quant.empty()) {
        std::vector<uint8_t> q(v.size());
        quantizeRoundU8(q.data(), v.data(), spec.quant_offset, spec.quant_scale, v.size());
        g.write(name, static_cast<const void*>(q.data()), q.size() * sizeof(uint8_t));
    } else {
        g.write(name, v.data(), v.size());
    }
}

// INCREMENTAL KV-in update — the tok/s lever. Quantize (round-to-nearest) ONLY
// the single new column this decode step produced and write it DIRECTLY into
// gd's PERSISTENT input buffer (the io_tensor RPC buffer cached in Graph::setup,
// reused across executes), instead of re-quantizing + re-uploading the ENTIRE
// KV cache every step (writeKVIn on the full [hd,KV]/[KV,hd] host buffers, which
// is O(context)/layer and was done twice per token). Prior columns already sit
// in the buffer from the one-time prefill seed + earlier in-place appends.
//   K is [hd, KV] column-major over positions -> new column at stride KV, index d*KV+pos.
//   V is [KV, hd] row-major over positions     -> new row contiguous at pos*hd .. pos*hd+hd.
// These offsets match KvCache::appendLayer, so the buffer stays byte-identical
// to what a full writeKVIn(cache) would have produced. Handles per-tensor
// uFixed16 (int16 KV) and uFixed8 (int8 KV I/O); throws otherwise so a
// layout/dtype change can't silently corrupt the cache.
static void writeKVColumn(geniex::Graph& g, const std::string& name, const float* col, int hd, int kv, int pos,
                          bool is_key) {
    const auto& spec = g.inputSpec(name);
    if (spec.axis_quant.empty() && spec.dtype == QNN_DATATYPE_UFIXED_POINT_16) {
        std::vector<uint16_t> q((size_t)hd);
        quantizeRoundU16(q.data(), col, spec.quant_offset, spec.quant_scale, (size_t)hd);
        uint16_t* buf = static_cast<uint16_t*>(g.inputPtr(name));
        if (is_key) { for (int d = 0; d < hd; ++d) buf[(size_t)d * kv + pos] = q[d]; }
        else        { uint16_t* row = buf + (size_t)pos * hd; for (int d = 0; d < hd; ++d) row[d] = q[d]; }
    } else if (spec.axis_quant.empty() && spec.dtype == QNN_DATATYPE_UFIXED_POINT_8) {
        std::vector<uint8_t> q((size_t)hd);
        quantizeRoundU8(q.data(), col, spec.quant_offset, spec.quant_scale, (size_t)hd);
        uint8_t* buf = static_cast<uint8_t*>(g.inputPtr(name));
        if (is_key) { for (int d = 0; d < hd; ++d) buf[(size_t)d * kv + pos] = q[d]; }
        else        { uint8_t* row = buf + (size_t)pos * hd; for (int d = 0; d < hd; ++d) row[d] = q[d]; }
    } else {
        throw std::runtime_error("writeKVColumn: expected per-tensor uFixed16/uFixed8 KV-in for '" + name + "'");
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
        std::cerr << "usage: --bundle <dir> --ctx <2graph.bin> [--prompt ...] [--max-tokens N] [--temp T]\n";
        return 1;
    }

    Gemma4Arch arch;
    // Prefill (ar128) geometry: S=128, full KV=4096-128=3968, sliding KV=512-128=384.
    // Decode  (ar1)  geometry: S=1,   full KV=4096-1=4095,    sliding KV=512-1=511.
    const int PS = 128, P_KVF = 3968, P_KVS = 384;
    const int D_KVF = 4095, D_KVS = 511;

    // ---- embedding tables ----
    std::cout << "[load] embedding tables...\n";
    Bf16Table embed, ple;
    embed.load((bundle / "embed_tokens_e2b.bf16.bin").string(), arch.vocab, arch.hidden);
    ple.load((bundle / "embed_tokens_per_layer_e2b.bf16.bin").string(), arch.vocab, arch.num_layers * arch.ple_dim);
    InputBuilder       prefill_ib(arch, PS, P_KVF, P_KVS, &embed, &ple);
    DecodeInputBuilder decode_ib(arch, D_KVF, D_KVS, &embed, &ple);

    // ---- tokenizer + chat frame (<bos><|turn>user\n...<turn|>\n<|turn>model\n) ----
    auto tok = geniex::Tokenizer::from_file((bundle / "tokenizer.json").string(),
                                            (bundle / "tokenizer_config.json").string());
    if (!tok) { std::cerr << "tokenizer load failed\n"; return 1; }
    const int32_t        BOS = 2, TURN_OPEN = 105, TURN_CLOSE = 106;
    std::vector<int32_t> ids;
    auto                 app = [&](const std::vector<int32_t>& v) { ids.insert(ids.end(), v.begin(), v.end()); };
    ids.push_back(BOS);
    ids.push_back(TURN_OPEN);
    app(tok->encode("user\n", false));
    app(tok->encode(prompt, false));
    ids.push_back(TURN_CLOSE);
    app(tok->encode("\n", false));
    ids.push_back(TURN_OPEN);
    app(tok->encode("model\n", false));
    const int prompt_len = (int)ids.size();
    std::cout << "[tok] prompt framed to " << prompt_len << " tokens\n";
    if (prompt_len > PS) { std::cerr << "prompt too long for prefill S=" << PS << "\n"; return 1; }

    // ---- load 2-graph context binary ----
    std::cout << "[load] context binary " << ctx.string() << " (NPU)...\n";
    geniex::QnnRuntimeConfig rt;
    geniex::ModelConfig      mc;
    mc.model_paths    = {ctx.string()};
    mc.tokenizer_path = (bundle / "tokenizer.json").string();
    // Apply the HTP backend-extension config (fp16_relaxed_precision, O, vtcm)
    // used to build the binary, matching every other model example. It sits next
    // to the binary (../htp_config.json); overridable via GEMMA4_HTP_CONFIG.
    {
        fs::path htp_cfg = ctx.parent_path() / ".." / "htp_config.json";
        if (const char* e = std::getenv("GEMMA4_HTP_CONFIG")) htp_cfg = e;
        if (fs::exists(htp_cfg)) {
            mc.htp_config_path = fs::absolute(htp_cfg).string();
            std::cout << "[load] htp_config: " << mc.htp_config_path << "\n";
        } else {
            std::cerr << "[warn] htp_config not found at " << htp_cfg << " — using defaults\n";
        }
    }
    Gemma4Model model;
    if (!model.initialize(rt, mc)) { std::cerr << "context binary load failed\n"; return 1; }
    int gi_pre = findGraph(model, "gemma4_ar128"), gi_dec = findGraph(model, "gemma4_ar1");
    if (gi_pre < 0 || gi_dec < 0) {
        std::cerr << "need both gemma4_ar128 and gemma4_ar1 graphs in the binary (found "
                  << model.graphCount() << " graphs)\n";
        return 1;
    }
    geniex::Graph& gp = model.graph(gi_pre);
    geniex::Graph& gd = model.graph(gi_dec);
    std::cout << "[load] prefill '" << gp.name() << "' (" << gp.inputSpecs().size() << " in), decode '"
              << gd.name() << "' (" << gd.inputSpecs().size() << " in)\n";

    // ---- sampler ----
    geniex_sampler_params sp;
    sp.temp    = temp;
    sp.no_perf = true;
    for (int e : {1, 106, 50}) sp.eog_tokens.push_back(e);
    geniex::Sampler sampler(sp);
    const bool      greedy = (temp <= 0.0f);
    auto            pick = [&](std::vector<float>& logits) -> int32_t {
        return greedy ? (int32_t)(std::max_element(logits.begin(), logits.end()) - logits.begin())
                                 : sampler.sample(logits);
    };

    std::map<std::string, std::pair<double, uint16_t>> tl;
    std::vector<float>                                 logits(arch.vocab);
    KvCache                                            cache(arch, D_KVF, D_KVS);
    auto                                               t0 = std::chrono::steady_clock::now();

    // ============================ PREFILL (once) ============================
    {
        auto embeds = prefill_ib.buildEmbeds(ids);
        gp.write("input_embeds", embeds.data(), embeds.size());
        auto per = prefill_ib.buildPerLayer(ids);
        for (int l = 0; l < arch.num_layers; ++l) {
            std::string nm = "per_layer_input_" + std::to_string(l);
            if (gp.hasInput(nm)) gp.write(nm, per[l].data(), per[l].size());
        }
        std::vector<float> cg, sg, cl, sl;
        prefill_ib.buildRope(false, ids, prompt_len, cg, sg);
        prefill_ib.buildRope(true, ids, prompt_len, cl, sl);
        gp.write("cos_global", cg.data(), cg.size());
        gp.write("sin_global", sg.data(), sg.size());
        gp.write("cos_local", cl.data(), cl.size());
        gp.write("sin_local", sl.data(), sl.size());
        auto mf = prefill_ib.buildMask(false, prompt_len);
        auto ms = prefill_ib.buildMask(true, prompt_len);
        gp.write("attention_mask_full", mf.data(), mf.size());
        gp.write("attention_mask_slide", ms.data(), ms.size());
        for (const auto& spec : gp.inputSpecs()) {
            if (spec.name.rfind("past_key_", 0) != 0 && spec.name.rfind("past_value_", 0) != 0) continue;
            std::vector<float> z(spec.elementCount(), 0.0f);
            gp.write(spec.name, z.data(), z.size());
        }
        if (!gp.execute(tl)) { std::cerr << "prefill execute failed\n"; return 1; }

        // sample first token from logits at last real prompt position
        const int last = prompt_len - 1;
        gp.read("logits", logits.data(), logits.size(), (size_t)last * arch.vocab);
        if (std::getenv("GEMMA4_DBG")) {
            int am = (int)(std::max_element(logits.begin(), logits.end()) - logits.begin());
            std::cerr << "[dbg] PREFILL argmax@" << last << " = " << am << "\n";
        }

        // Seed the DECODE graph's KV-in from the PREFILL graph's KV-out. KV-out is
        // dequantized to float (Graph::read) then requantized round-to-nearest into
        // the decode KV-in (writeKVIn); the unfilled tail is float 0.0 -> encoded
        // zero-point. Only the first num_layers-num_kv_shared layers own KV.
        for (int L = 0; L < cache.numPairs(); ++L) {
            int                hd = cache.hdOf(L);
            std::vector<float> ko((size_t)hd * PS), vo((size_t)hd * PS);
            gp.read("past_key_" + std::to_string(L) + "_out", ko.data(), ko.size());
            gp.read("past_value_" + std::to_string(L) + "_out", vo.data(), vo.size());
            cache.seedLayer(L, ko.data(), PS, vo.data(), prompt_len);
            writeKVIn(gd, "past_key_" + std::to_string(L) + "_in", cache.key(L));
            writeKVIn(gd, "past_value_" + std::to_string(L) + "_in", cache.val(L));
        }
    }
    double ttft_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();

    // ============================ DECODE (loop) =============================
    // KV-in update strategy. DEFAULT = incremental: the prefill seed above wrote
    // the full KV-in buffer once, and each step writes ONLY the one new column
    // directly into gd's persistent RPC buffer (writeKVColumn). Set
    // GEMMA4_FULL_KV_REWRITE=1 to restore the old behavior (re-quantize +
    // re-upload the ENTIRE cache twice/step via writeKVIn) for an on-device A/B —
    // the two paths are byte-identical, incremental is just O(hd) vs O(hd*KV).
    // The full rewrite starves the NPU (host requant of all 15 layers × full KV
    // width, done twice/step) → NPU-not-full + ~2× slower decode once SHA made
    // the NPU step cheap; incremental is the tok/s lever.
    const bool full_rewrite = std::getenv("GEMMA4_FULL_KV_REWRITE") != nullptr;
    std::cout << "[kv] update mode: " << (full_rewrite ? "FULL rewrite (legacy)" : "incremental (new)") << "\n";
    // GEMMA4_PROFILE=1 decomposes each decode step into host input-build vs NPU
    // execute vs logits-read vs KV-append, to see where the step time goes.
    const bool profile = std::getenv("GEMMA4_PROFILE") != nullptr;
    double     t_build = 0, t_exec = 0, t_logits = 0, t_kv = 0;
    auto       clk  = [&] { return std::chrono::steady_clock::now(); };
    auto       msec = [](auto a, auto b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
    std::cout << "\n\033[33m";
    std::vector<int32_t> gen;
    int                  n_past = prompt_len;  // number of KV positions already filled
    auto                 t_dec0 = std::chrono::steady_clock::now();
    for (int step = 0; step < max_tokens; ++step) {
        int32_t next = pick(logits);
        if (next == 106 || next == 1 || tok->is_eog(next)) { std::cout << "\n[eot]\n"; break; }
        std::string prev = tok->decode(gen);
        gen.push_back(next);
        std::string cur = tok->decode(gen);
        std::cout << cur.substr(std::min(prev.size(), cur.size())) << std::flush;
        if (n_past >= D_KVF) { std::cout << "\n[ctx full]\n"; break; }

        // ---- build AR=1 inputs for `next` at position n_past ----
        auto tb0 = clk();
        auto e1 = decode_ib.buildEmbeds(next);
        gd.write("input_embeds", e1.data(), e1.size());
        auto per1 = decode_ib.buildPerLayer(next);
        for (int l = 0; l < arch.num_layers; ++l) {
            std::string nm = "per_layer_input_" + std::to_string(l);
            if (gd.hasInput(nm)) gd.write(nm, per1[l].data(), per1[l].size());
        }
        std::vector<float> cg, sg, cl, sl;
        decode_ib.buildRope(false, n_past, cg, sg);
        decode_ib.buildRope(true, n_past, cl, sl);
        gd.write("cos_global", cg.data(), cg.size());
        gd.write("sin_global", sg.data(), sg.size());
        gd.write("cos_local", cl.data(), cl.size());
        gd.write("sin_local", sl.data(), sl.size());
        auto mf = decode_ib.buildMask(false, n_past);
        auto ms = decode_ib.buildMask(true, n_past);
        gd.write("attention_mask_full", mf.data(), mf.size());
        gd.write("attention_mask_slide", ms.data(), ms.size());
        // Legacy path only: re-write the whole running KV cache into gd's KV-in
        // before execute. Within a single graph each input tensor has its own RPC
        // buffer, so the embeds/RoPE/mask writes above cannot disturb KV-in — the
        // incremental path relies on that + the persistence of gd's KV-in buffer
        // across executes, so it skips this entirely.
        if (full_rewrite) {
            for (int L = 0; L < cache.numPairs(); ++L) {
                writeKVIn(gd, "past_key_" + std::to_string(L) + "_in", cache.key(L));
                writeKVIn(gd, "past_value_" + std::to_string(L) + "_in", cache.val(L));
            }
        }
        auto te0 = clk();
        if (profile) t_build += msec(tb0, te0);
        if (!gd.execute(tl)) { std::cerr << "\ndecode execute failed\n"; break; }
        auto tl0 = clk();
        if (profile) t_exec += msec(te0, tl0);

        // read single logits row
        gd.read("logits", logits.data(), logits.size(), 0);
        auto tk0 = clk();
        if (profile) t_logits += msec(tl0, tk0);
        if (std::getenv("GEMMA4_DBG")) {
            int am = (int)(std::max_element(logits.begin(), logits.end()) - logits.begin());
            std::cerr << "[dbg] decode step " << step << " n_past=" << n_past << " -> argmax " << am << "\n";
        }
        // Append this step's KV at column/row n_past. Incremental (default):
        // update the host cache AND write only the new column straight into gd's
        // persistent KV-in buffer (O(hd)/layer). Legacy: update the host cache,
        // then the next iteration's pre-execute loop re-uploads the whole thing.
        for (int L = 0; L < cache.numPairs(); ++L) {
            int                hd = cache.hdOf(L);
            std::vector<float> ko(hd), vo(hd);  // K-out [hd,1], V-out [1,hd]
            gd.read("past_key_" + std::to_string(L) + "_out", ko.data(), ko.size());
            gd.read("past_value_" + std::to_string(L) + "_out", vo.data(), vo.size());
            cache.appendLayer(L, ko.data(), vo.data(), n_past);
            if (!full_rewrite) {
                int kv = cache.kvOf(L);
                writeKVColumn(gd, "past_key_" + std::to_string(L) + "_in", ko.data(), hd, kv, n_past, true);
                writeKVColumn(gd, "past_value_" + std::to_string(L) + "_in", vo.data(), hd, kv, n_past, false);
            }
        }
        if (profile) t_kv += msec(tk0, clk());
        n_past++;
    }
    std::cout << "\033[0m\n";
    double dec_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_dec0).count();
    std::cout << "\n[done] " << gen.size() << " tokens | TTFT " << (int)ttft_ms << " ms | decode "
              << (gen.empty() ? 0.0 : (gen.size() * 1000.0 / dec_ms)) << " tok/s\n";
    if (profile && !gen.empty()) {
        int    n   = (int)gen.size();
        double tot = t_build + t_exec + t_logits + t_kv;
        printf("[profile] per-token avg over %d steps (ms):\n", n);
        printf("  input-build : %6.2f  (%4.1f%%)\n", t_build / n, 100 * t_build / tot);
        printf("  NPU execute : %6.2f  (%4.1f%%)\n", t_exec / n, 100 * t_exec / tot);
        printf("  logits-read : %6.2f  (%4.1f%%)  [dequant %d-vocab -> f32]\n", t_logits / n,
               100 * t_logits / tot, arch.vocab);
        printf("  KV-append   : %6.2f  (%4.1f%%)\n", t_kv / n, 100 * t_kv / tot);
        printf("  accounted   : %6.2f  /  measured step %.2f\n", tot / n, dec_ms / n);
    }
    std::cout << "[full] " << tok->decode(gen) << "\n";
    return 0;
}
