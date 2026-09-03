// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause
//
// Shared test doubles for LLMModel orchestration tests: a subclass that injects
// CPU graph fixtures in place of a QNN backend, plus an env guard that forces
// serial decode. Used by both llm_model_test and eagle_model_test.

#pragma once

#include <cstdlib>
#include <memory>
#include <utility>

#include "IOTensor.hpp"
#include "QnnApi.hpp"
#include "llm/llm_model.h"
#include "llm/speculative_llm_model.h"

namespace geniex::testing {

// Exposes the protected Model members so a test can wire in pre-built graphs
// and run the orchestration entry points without a QNN backend.
class TestableLLMModel : public geniex::LLMModel {
   public:
    explicit TestableLLMModel(geniex::LLMSpec spec) : geniex::LLMModel(std::move(spec)) {}

    // Moves the fixture's graphs into the model and runs onInitialized(). The
    // fixture (which owns the IOTensor / QnnApi / tensor buffers the graphs
    // point at) must outlive this model. Templated so any fixture exposing
    // `.io` and `.graphs` works.
    template <typename Fixture>
    bool initFromFixture(Fixture& fx) {
        api_       = std::make_unique<QnnApi>();
        io_tensor_ = std::shared_ptr<IOTensor>(std::shared_ptr<void>{}, &fx.io);  // non-owning alias
        for (auto& g : fx.graphs) graphs_.push_back(std::move(g));
        const bool ok = onInitialized();
        initialized_  = ok;
        return ok;
    }

    // Expose protected helpers for direct testing.
    using geniex::LLMModel::computeSlideDiscard;
    using geniex::LLMModel::discoverKVPairs;
    using geniex::LLMModel::isEndOfGeneration;
    using geniex::LLMModel::spec_;

    // Exposed for direct reshapeKV / promoteCL restride tests.
    using geniex::LLMModel::active_cl_idx_;
    using geniex::LLMModel::graph;
    using geniex::LLMModel::graphIndex;
    using geniex::LLMModel::requireKVStateBlock;
    using geniex::LLMModel::reshapeKV;

    // Exposed for scatter-cache (native KV) tests. kvLen() is already public.
    using geniex::LLMModel::kvScatter;
};

// Same fixture-injection harness for the speculative subclass, exposing the
// protected Model members its batched/tree-decode tests wire in.
class TestableSpeculativeLLMModel : public geniex::SpeculativeLLMModel {
   public:
    explicit TestableSpeculativeLLMModel(geniex::LLMSpec spec) : geniex::SpeculativeLLMModel(std::move(spec)) {}

    template <typename Fixture>
    bool initFromFixture(Fixture& fx) {
        api_       = std::make_unique<QnnApi>();
        io_tensor_ = std::shared_ptr<IOTensor>(std::shared_ptr<void>{}, &fx.io);  // non-owning alias
        for (auto& g : fx.graphs) graphs_.push_back(std::move(g));
        const bool ok = onInitialized();
        initialized_  = ok;
        return ok;
    }

    using geniex::LLMModel::active_cl_idx_;
    using geniex::LLMModel::graph;
    using geniex::LLMModel::graphIndex;
    using geniex::LLMModel::requireKVStateBlock;
    using geniex::LLMModel::spec_;
};

// Decode runs serially when no worker pool is created; the pool is only built
// when GENIEX_DECODE_WORKERS or the clock keeper is enabled. Force both off so
// updateKV happens inline and the loop is fully deterministic.
struct NoDecodePoolEnv {
    NoDecodePoolEnv() {
        _putenv_s("GENIEX_DECODE_WORKERS", "0");
        _putenv_s("GENIEX_CLOCK_KEEPER_THREADS", "0");
    }
    ~NoDecodePoolEnv() {
        _putenv_s("GENIEX_DECODE_WORKERS", "");
        _putenv_s("GENIEX_CLOCK_KEEPER_THREADS", "");
    }
};

}  // namespace geniex::testing
