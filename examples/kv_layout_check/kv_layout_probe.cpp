// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause
//
// KV layout probe: runs one prefill chunk and dumps the graph's own KV OUTPUT
// bytes verbatim.
//
// The point is to recover the HMX tile layout from ground truth instead of
// guessing it. Run the same fixed prompt against a flat bundle and its
// ENABLE_NATIVE_KV twin: the flat dump gives the logical [head, head_dim, token]
// values, the native dump gives the same values permuted by the hardware layout.
// Comparing the two recovers the permutation exactly (see scripts/kv_solve.py).
//
//   kv_layout_probe --model-dir <bundle> --out kv_out.bin [--tensor past_key_0_out]
//
// Prompt is fixed (not configurable) so the two runs are directly comparable.

#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "geniex-proc/tokenizer.h"
#include "llm/llm_spec_loader.h"
#include "llm/speculative_llm_model.h"
#include "qwen3_eaglet.h"
#include "types.h"

namespace fs = std::filesystem;
using namespace geniex;

int main(int argc, char** argv) {
    std::string model_dir, out_path, tensor = "past_key_0_out";
    size_t      repeat_to = 0;  // pad the prompt to at least this many tokens
    for (int i = 1; i < argc; ++i) {
        std::string a    = argv[i];
        auto        next = [&]() -> std::string { return (i + 1 < argc) ? argv[++i] : std::string{}; };
        if (a == "--model-dir")
            model_dir = next();
        else if (a == "--out")
            out_path = next();
        else if (a == "--tensor")
            tensor = next();
        else if (a == "--repeat-to")
            repeat_to = std::strtoul(next().c_str(), nullptr, 10);
        else {
            std::cerr << "Usage: " << argv[0] << " --model-dir <dir> --out <file> [--tensor <name>]\n";
            return 1;
        }
    }
    if (model_dir.empty() || out_path.empty()) {
        std::cerr << "Usage: " << argv[0] << " --model-dir <dir> --out <file> [--tensor <name>]\n";
        return 1;
    }

    try {
        // Build the TARGET engine alone. Its export names the embedding tensor
        // something only the eaglet config knows, so reuse that config's binding
        // rather than the generic input_ids/input_embeds lookup -- exactly what
        // EagleModel::initialize does for the target.
        ModelConfig model_cfg = modelConfigFromDirectory(model_dir);
        const auto  bundle    = bundleDirOf(model_cfg);
        auto        gc        = parseGenieConfig(bundle);
        const auto  ecfg      = qwen3_eaglet::parseEagletConfig(bundle, gc);

        SpeculativeLLMModel tgt(buildSpecSkeleton(gc), gc);
        tgt.setEmbeddingBinding(ecfg.embedding_quant, /*table_path=*/"");
        if (!tgt.initialize(QnnRuntimeConfig{}, model_cfg)) {
            std::cerr << "target initialize failed\n";
            return 1;
        }

        auto tokenizer = Tokenizer::from_file(model_cfg.tokenizer_path, "");
        if (!tokenizer) {
            std::cerr << "tokenizer load failed\n";
            return 1;
        }
        // Fixed prompt, and short enough to be one prefill chunk: the cache is
        // fully masked during that chunk, so the KV OUTPUT depends only on the
        // graph's own compute and is directly comparable across bundles.
        auto tokens = tokenizer->encode("The capital of France is");
        // Repeat until the prompt spans more than one prefill chunk: only then does
        // a later chunk READ the KV cache we wrote, making the dump sensitive to
        // the kv_in layout. A single chunk sees a fully-masked cache and tells us
        // nothing.
        while (tokens.size() < repeat_to) {
            const auto n = tokens.size();
            for (size_t i = 0; i < n && tokens.size() < repeat_to; ++i) tokens.push_back(tokens[i]);
        }
        std::cout << "prefill tokens: " << tokens.size() << " (chunk size " << tgt.spec().seq_len_prefill << ")\n";
        // prefill(), not forwardLogits(): the latter resets the KV cache on exit,
        // which refills the buffers with the clear value before we can read them.
        tgt.resetKVCache();
        tgt.prefill(tokens, ecfg.rope_theta, /*feature_rows=*/nullptr, /*feature_row_bytes=*/0, /*feature_name=*/"");

        // `logits` lives on the last shard; KV tensors on the body shard.
        const size_t shard = (tensor == "logits") ? tgt.spec().shards.size() - 1 : 0;
        const size_t gi    = tgt.graphIndex(/*phase=*/0, shard, tgt.activeContextLengthIndex());
        const auto&  ts    = tgt.outputTensorSpec(gi, tensor);
        const void*  p     = tgt.outputBytes(gi, tensor);
        if (!p) {
            std::cerr << "no output tensor '" << tensor << "'\n";
            return 1;
        }

        std::string shape;
        for (size_t i = 0; i < ts.shape.size(); ++i) shape += (i ? "," : "") + std::to_string(ts.shape[i]);
        std::cout << "tensor '" << tensor << "' shape=[" << shape << "] bytes=" << ts.byteCount()
                  << " dataFormat=" << ts.data_format << "\n";

        std::ofstream f(out_path, std::ios::binary);
        f.write(static_cast<const char*>(p), static_cast<std::streamsize>(ts.byteCount()));
        if (!f) {
            std::cerr << "write failed\n";
            return 1;
        }
        std::cout << "wrote " << out_path << "\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}
