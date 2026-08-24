// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause
//
// KV cache layout inspector.
//
// Reports the physical byte layout of every KV tensor in a bundle -- which is
// what tells you whether an ENABLE_NATIVE_KV recipe actually emitted
// HMX_WEIGHT_LAYOUT, and whether the shapes it chose are representable in that
// layout (see kv::validateGeometry).
//
// Deliberately family-agnostic: it loads the graphs and reads tensor metadata
// only, with no spec inference, provider wiring or execution. That way it works
// on any bundle -- including eaglet/speculative ones whose generic spec loading
// needs a family-specific driver.
//
//   kv_layout_check --model-dir <bundle>
//
// See docs/native-kv-cache.md.

#include <algorithm>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include "llm/kv_layout.h"
#include "llm/llm_spec_loader.h"
#include "llm/llm_utils.h"  // isKVTensor
#include "model.h"
#include "types.h"

namespace fs = std::filesystem;
using namespace geniex;

namespace {

// Loads graphs and nothing else. Model::onInitialized() defaults to a no-op, so
// no KV pairs are resolved, no providers are built, and no shapes are inferred
// -- exactly what an inspector wants.
class GraphOnlyModel : public Model {
   public:
    GraphOnlyModel() = default;
};

void printUsage(const char* prog) {
    std::cout << "Usage: " << prog << " --model-dir <path>\n"
              << "  --model-dir <path>   QAIRT bundle directory (required)\n"
              << "  --help, -h\n";
}

std::string dtypeName(Qnn_DataType_t dt) {
    switch (dt) {
        case QNN_DATATYPE_UFIXED_POINT_8:
            return "ufixed8";
        case QNN_DATATYPE_SFIXED_POINT_8:
            return "sfixed8";
        case QNN_DATATYPE_UFIXED_POINT_16:
            return "ufixed16";
        case QNN_DATATYPE_SFIXED_POINT_16:
            return "sfixed16";
        case QNN_DATATYPE_FLOAT_16:
            return "fp16";
        case QNN_DATATYPE_FLOAT_32:
            return "fp32";
        default:
            return "dtype#" + std::to_string(static_cast<int>(dt));
    }
}

std::string shapeStr(const std::vector<uint32_t>& s) {
    std::string out = "[";
    for (size_t i = 0; i < s.size(); ++i) out += (i ? "," : "") + std::to_string(s[i]);
    return out + "]";
}

// A KV tensor's name tells us key vs value; the shape alone cannot.
bool isKeyTensor(const std::string& name) { return name.find("key") != std::string::npos; }

// One row per distinct (role, dtype, shape, format) descriptor -- a 36-layer
// model has 144 KV tensors per graph variant and they are all alike.
struct Row {
    std::string role, dtype, shape, format, sample;
    size_t      count = 0;
    bool        legal = true;
    std::string error;
};

void addRow(std::vector<Row>& rows, const std::string& name, const TensorSpec& ts, bool is_out) {
    const bool  is_key = isKeyTensor(name);
    const auto  geo    = kv::geometryOf(ts, is_key);
    std::string role   = std::string(is_out ? "out " : "in  ") + (is_key ? "key  " : "value");

    for (auto& r : rows) {
        if (r.role == role && r.dtype == dtypeName(ts.dtype) && r.shape == shapeStr(ts.shape) &&
            r.format == kv::formatName(geo.format)) {
            ++r.count;
            return;
        }
    }

    Row r{role, dtypeName(ts.dtype), shapeStr(ts.shape), kv::formatName(geo.format), name, 1, true, {}};
    try {
        kv::validateGeometry(geo, name);
    } catch (const std::exception& e) {
        r.legal = false;
        r.error = e.what();
    }
    rows.push_back(std::move(r));
}

}  // namespace

int main(int argc, char** argv) {
    std::string model_dir;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--model-dir" && i + 1 < argc) {
            model_dir = argv[++i];
        } else if (a == "--help" || a == "-h") {
            printUsage(argv[0]);
            return 0;
        } else {
            std::cerr << "Unknown or incomplete argument: " << a << "\n";
            printUsage(argv[0]);
            return 1;
        }
    }
    if (model_dir.empty()) {
        printUsage(argv[0]);
        return 1;
    }
    if (!fs::exists(model_dir)) {
        std::cerr << "model dir not found: " << model_dir << "\n";
        return 1;
    }

    try {
        const ModelConfig model_cfg = modelConfigFromDirectory(model_dir);
        std::cout << "bundle: " << model_dir << "\n";
        for (const auto& p : model_cfg.model_paths) std::cout << "  ctx-bin: " << fs::path(p).filename() << "\n";

        GraphOnlyModel   model;
        QnnRuntimeConfig runtime_cfg;
        if (!model.initialize(runtime_cfg, model_cfg)) {
            std::cerr << "model initialize failed\n";
            return 1;
        }

        size_t n_tiled = 0, n_flat = 0, n_illegal = 0;

        for (size_t gi = 0; gi < model.graphCount(); ++gi) {
            const Graph&     g = model.graph(gi);
            std::vector<Row> rows;
            for (const auto& ts : g.inputSpecs())
                if (isKVTensor(ts.name)) addRow(rows, ts.name, ts, /*is_out=*/false);
            for (const auto& ts : g.outputSpecs())
                if (isKVTensor(ts.name)) addRow(rows, ts.name, ts, /*is_out=*/true);
            if (rows.empty()) continue;

            std::cout << "\ngraph '" << g.name() << "'\n";
            for (const auto& r : rows) {
                std::cout << "  x" << r.count << "\t" << r.role << "  " << r.format << "  " << r.dtype << "  "
                          << r.shape << "\n";
                if (!r.legal) {
                    std::cout << "        NOT REPRESENTABLE: " << r.error << "\n";
                    ++n_illegal;
                }
                if (r.format == std::string("HMX_WEIGHT_LAYOUT"))
                    n_tiled += r.count;
                else
                    n_flat += r.count;
            }
        }

        std::cout << "\n== summary ==\n"
                  << "  HMX_WEIGHT_LAYOUT tensors : " << n_tiled << "\n"
                  << "  FLAT_BUFFER tensors       : " << n_flat << "\n";
        if (n_tiled == 0) {
            std::cout << "  -> flat bundle; the runtime takes the original strided-copy path\n";
        } else {
            std::cout << "  -> native KV bundle; cache writes go through the tiled addressing\n";
        }
        if (n_illegal) {
            std::cout << "  " << n_illegal << " descriptor(s) cannot be represented in the declared layout; the\n"
                      << "  runtime will refuse to load this bundle. Fix the recipe's kv_len.\n";
            return 2;
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}
