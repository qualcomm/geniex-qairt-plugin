// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause
//
// Unit tests for core/src/graph.cpp — tensor write/read round-trips (fp32,
// fp16, quantized uint8/uint16, int32) and spec building. Drives the real
// Graph against a real ClientBuffer-backed IOTensor and a link-time QnnApi
// stub; no QNN runtime.

#include "graph.h"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "IOTensor.hpp"
#include "testing/graph_info_builder.hpp"
#include "utils.h"

namespace {

using geniex::testing::GraphInfoBuilder;
using geniex::testing::TensorDesc;

// Builds a single-input/single-output Graph over a fixture (with a stub QnnApi)
// and runs setup(). The stub's graphExecute copies input bytes to output, so
// write(in) -> execute -> read(out) is a true round-trip.
struct GraphFixture {
    QnnApi        api;
    IOTensor      io{BufferAlloc::DEFAULT};
    geniex::Graph graph;

    GraphFixture(GraphInfoBuilder& b) : graph(&b.graphInfo(), &api, &io) {
        EXPECT_TRUE(graph.setup(/*context=*/nullptr));
    }
};

}  // namespace

TEST(GraphSetup, BuildsSpecsFromGraphInfo) {
    GraphInfoBuilder b("g",
        {{"in", QNN_DATATYPE_FLOAT_32, {2, 3}}},
        {{"out", QNN_DATATYPE_UFIXED_POINT_16, {4}, /*scale=*/0.5f, /*offset=*/-2}});
    GraphFixture     f(b);
    geniex::Graph&   g = f.graph;

    EXPECT_EQ(g.name(), "g");
    ASSERT_TRUE(g.hasInput("in"));
    ASSERT_TRUE(g.hasOutput("out"));
    EXPECT_FALSE(g.hasInput("missing"));

    const auto& in = g.inputSpec("in");
    EXPECT_EQ(in.dtype, QNN_DATATYPE_FLOAT_32);
    EXPECT_EQ(in.shape, (std::vector<uint32_t>{2, 3}));

    const auto& out = g.outputSpec("out");
    EXPECT_EQ(out.dtype, QNN_DATATYPE_UFIXED_POINT_16);
    EXPECT_FLOAT_EQ(out.quant_scale, 0.5f);
    EXPECT_EQ(out.quant_offset, -2);
}

TEST(GraphIO, Float32RoundTrip) {
    GraphInfoBuilder b("g", {{"in", QNN_DATATYPE_FLOAT_32, {4}}}, {{"out", QNN_DATATYPE_FLOAT_32, {4}}});
    GraphFixture     f(b);

    const std::vector<float> src = {-1.5f, 0.0f, 3.25f, 100.0f};
    f.graph.write("in", src.data(), src.size());
    geniex::TimeLog log;
    ASSERT_TRUE(f.graph.execute(log));

    std::vector<float> got(4);
    f.graph.read("out", got.data(), got.size());
    EXPECT_EQ(got, src);
}

// Lossy; check within fp16 precision.
TEST(GraphIO, Float16RoundTrip) {
    GraphInfoBuilder b("g", {{"in", QNN_DATATYPE_FLOAT_16, {5}}}, {{"out", QNN_DATATYPE_FLOAT_16, {5}}});
    GraphFixture     f(b);

    const std::vector<float> src = {0.5f, -2.0f, 1.0f, 0.125f, -8.0f};
    f.graph.write("in", src.data(), src.size());
    geniex::TimeLog log;
    ASSERT_TRUE(f.graph.execute(log));

    std::vector<float> got(5);
    f.graph.read("out", got.data(), got.size());
    for (size_t i = 0; i < src.size(); ++i)
        EXPECT_NEAR(got[i], src[i], std::abs(src[i]) * 1e-3f + 1e-7f) << "index " << i;
}

TEST(GraphIO, UFixed8RoundTrip) {
    const float      scale  = 0.1f;
    const int32_t    offset = -10;
    GraphInfoBuilder b("g",
        {{"in", QNN_DATATYPE_UFIXED_POINT_8, {4}, scale, offset}},
        {{"out", QNN_DATATYPE_UFIXED_POINT_8, {4}, scale, offset}});
    GraphFixture     f(b);

    const std::vector<float> src = {0.0f, 1.0f, 2.5f, 5.0f};
    f.graph.write("in", src.data(), src.size());
    geniex::TimeLog log;
    ASSERT_TRUE(f.graph.execute(log));

    std::vector<float> got(4);
    f.graph.read("out", got.data(), got.size());
    for (size_t i = 0; i < src.size(); ++i)
        EXPECT_NEAR(got[i], src[i], scale) << "index " << i;  // within one quant step
}

TEST(GraphIO, UFixed16RoundTrip) {
    const float      scale  = 0.001f;
    const int32_t    offset = 0;
    GraphInfoBuilder b("g",
        {{"in", QNN_DATATYPE_UFIXED_POINT_16, {3}, scale, offset}},
        {{"out", QNN_DATATYPE_UFIXED_POINT_16, {3}, scale, offset}});
    GraphFixture     f(b);

    const std::vector<float> src = {0.5f, 1.25f, 10.0f};
    f.graph.write("in", src.data(), src.size());
    geniex::TimeLog log;
    ASSERT_TRUE(f.graph.execute(log));

    std::vector<float> got(3);
    f.graph.read("out", got.data(), got.size());
    for (size_t i = 0; i < src.size(); ++i) EXPECT_NEAR(got[i], src[i], scale) << "index " << i;
}

TEST(GraphIO, Int32RoundTrip) {
    GraphInfoBuilder b("g", {{"in", QNN_DATATYPE_INT_32, {4}}}, {{"out", QNN_DATATYPE_INT_32, {4}}});
    GraphFixture     f(b);

    const std::vector<int32_t> src = {-5, 0, 42, 1000};
    f.graph.write("in", src.data(), src.size());
    geniex::TimeLog log;
    ASSERT_TRUE(f.graph.execute(log));

    std::vector<float> got(4);
    f.graph.read("out", got.data(), got.size());
    EXPECT_EQ(got, (std::vector<float>{-5.0f, 0.0f, 42.0f, 1000.0f}));
}

TEST(GraphIO, WriteOverflowThrows) {
    GraphInfoBuilder b("g", {{"in", QNN_DATATYPE_FLOAT_32, {2}}}, {{"out", QNN_DATATYPE_FLOAT_32, {2}}});
    GraphFixture     f(b);

    const std::vector<float> too_many = {1.f, 2.f, 3.f, 4.f};
    EXPECT_THROW(f.graph.write("in", too_many.data(), too_many.size()), std::runtime_error);
}

TEST(GraphExecute, StubReportsSuccess) {
    GraphInfoBuilder b("g", {{"in", QNN_DATATYPE_FLOAT_32, {2}}}, {{"out", QNN_DATATYPE_FLOAT_32, {2}}});
    GraphFixture     f(b);

    geniex::TimeLog log;
    EXPECT_TRUE(f.graph.execute(log));
}
