// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause
//
// Unit tests for core/src/model.cpp: graph/sub-model accessors and
// applyConnections(), plus ModelConfig defaults. Model::initialize() is device-only and not tested here.
// TestableModel injects pre-built graphs to exercise the CPU-reachable paths.

#include "model.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <memory>
#include <vector>

#include "IOTensor.hpp"
#include "graph.h"
#include "testing/graph_info_builder.hpp"
#include "types.h"

namespace {

using geniex::testing::GraphInfoBuilder;

// Bypasses device initialize() by injecting graphs directly into the protected graphs_ vector.
class TestableModel : public geniex::Model {
   public:
    void addGraph(geniex::Graph g) { graphs_.push_back(std::move(g)); }
    using geniex::Model::applyConnections;
};

geniex::Graph makeGraph(GraphInfoBuilder& b, IOTensor& io) {
    geniex::Graph g(&b.graphInfo(), /*api=*/nullptr, &io);
    EXPECT_TRUE(g.setup(nullptr));
    return g;
}

}  // namespace

// Context-binary loading options must stay off unless a caller opts in, so existing bundles keep
// loading exactly as before.
TEST(ModelConfig, ContextBinaryLoadingOptionsDefaultOff) {
    const geniex::ModelConfig cfg;
    EXPECT_FALSE(cfg.mmap_context_bins);
    EXPECT_EQ(cfg.file_read_budget_mb, 0u);
}

// Accessors reflect the injected graph set.
TEST(Model, GraphAccessors) {
    GraphInfoBuilder b0("g0", {{"in", QNN_DATATYPE_FLOAT_32, {2}}}, {{"out", QNN_DATATYPE_FLOAT_32, {2}}});
    IOTensor         io(BufferAlloc::DEFAULT);
    TestableModel    model;
    EXPECT_EQ(model.graphCount(), 0u);
    EXPECT_FALSE(model.isInitialized());

    model.addGraph(makeGraph(b0, io));
    EXPECT_EQ(model.graphCount(), 1u);
    EXPECT_NO_THROW(model.graph(0));
    EXPECT_THROW(model.graph(5), std::out_of_range);
}

// Sub-model registration + accessor (mutable and const overloads).
TEST(Model, SubModelAccessors) {
    auto          child = std::make_shared<TestableModel>();
    TestableModel parent;
    parent.addSubModel(child);
    EXPECT_EQ(&parent.subModel(0), child.get());
    EXPECT_THROW(parent.subModel(3), std::out_of_range);

    // Exercise the const overload via a const reference, including bounds check.
    const TestableModel& cparent = parent;
    EXPECT_EQ(&cparent.subModel(0), child.get());
    EXPECT_THROW(cparent.subModel(3), std::out_of_range);
}

// applyConnections copies graph0's output buffer into graph1's input buffer.
TEST(Model, ApplyConnectionsCopiesAcrossGraphs) {
    GraphInfoBuilder b0("g0", {{"x", QNN_DATATYPE_FLOAT_32, {2}}}, {{"out", QNN_DATATYPE_FLOAT_32, {2}}});
    GraphInfoBuilder b1("g1", {{"in", QNN_DATATYPE_FLOAT_32, {2}}}, {{"y", QNN_DATATYPE_FLOAT_32, {2}}});
    IOTensor         io(BufferAlloc::DEFAULT);

    TestableModel model;
    geniex::Graph g0 = makeGraph(b0, io);
    geniex::Graph g1 = makeGraph(b1, io);

    // Seed graph0's output buffer directly (Graph::write targets inputs only).
    const std::vector<float> src     = {3.0f, 7.0f};
    auto*                    out_buf = const_cast<float*>(static_cast<const float*>(g0.outputPtr("out")));
    std::copy(src.begin(), src.end(), out_buf);

    model.addGraph(std::move(g0));
    model.addGraph(std::move(g1));

    const std::vector<geniex::Connection> conns = {{/*src_graph=*/0, "out", /*dst_graph=*/1, "in"}};
    model.applyConnections(conns);

    const auto* got = static_cast<const float*>(model.graph(1).inputPtr("in"));
    EXPECT_EQ(std::vector<float>(got, got + 2), src);
}

// A connection naming a missing tensor is wrapped in a descriptive throw.
TEST(Model, ApplyConnectionsThrowsOnMissingTensor) {
    GraphInfoBuilder b0("g0", {{"x", QNN_DATATYPE_FLOAT_32, {2}}}, {{"out", QNN_DATATYPE_FLOAT_32, {2}}});
    GraphInfoBuilder b1("g1", {{"in", QNN_DATATYPE_FLOAT_32, {2}}}, {{"y", QNN_DATATYPE_FLOAT_32, {2}}});
    IOTensor         io(BufferAlloc::DEFAULT);

    TestableModel model;
    model.addGraph(makeGraph(b0, io));
    model.addGraph(makeGraph(b1, io));

    const std::vector<geniex::Connection> conns = {{0, "does_not_exist", 1, "in"}};
    EXPECT_THROW(model.applyConnections(conns), std::runtime_error);
}
