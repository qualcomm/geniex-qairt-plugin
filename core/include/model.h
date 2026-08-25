// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <memory>
#include <vector>

#include "IOTensor.hpp"
#include "QnnApi.hpp"
#include "geniex_export.h"
#include "graph.h"
#include "types.h"

namespace geniex {

class GENIEX_API Model {
   public:
    virtual ~Model() = default;

    Model(const Model&)                = delete;
    Model& operator=(const Model&)     = delete;
    Model(Model&&) noexcept            = default;
    Model& operator=(Model&&) noexcept = default;

    // Sets up the QNN backend and loads all graphs. Must be called before any
    // subclass inference.
    virtual bool initialize(const QnnRuntimeConfig& runtime_cfg, const ModelConfig& model_cfg);

    bool isInitialized() const;

    size_t       graphCount() const;
    Graph&       graph(size_t idx);
    const Graph& graph(size_t idx) const;

    void         addSubModel(std::shared_ptr<Model> sub_model);
    Model&       subModel(size_t idx);
    const Model& subModel(size_t idx) const;

   protected:
    Model() = default;

    // Called by initialize() after all Graph objects have been set up.
    // Override to build connection lists, allocate KV buffers, etc.
    virtual bool onInitialized() { return true; }

    // Wires inter-graph connections by pointing each destination input buffer
    // directly at the source output buffer (zero-copy inter-shard data flow).
    void applyConnections(const std::vector<Connection>& connections);

    // Logs the device-reported NSP core count and, when model_cfg.num_cores > 1,
    // requests multicore graph execution (clamped to the device core count).
    // Called by initialize() after backend bring-up, before the first execute.
    void applyHtpNumCores(const ModelConfig& model_cfg);

    // Declaration order matters: graphs_ holds non-owning pointers into api_
    // and io_tensor_, so graphs_ must be destroyed first (it is listed last).
    ModelConfig               model_cfg_;
    std::unique_ptr<QnnApi>   api_;
    std::shared_ptr<IOTensor> io_tensor_;

    std::vector<Graph>                  graphs_;
    std::vector<std::shared_ptr<Model>> sub_models_;

    bool initialized_ = false;
};

}  // namespace geniex
