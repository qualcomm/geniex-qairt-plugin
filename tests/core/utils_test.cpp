// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause
//
// Unit tests for core/src/utils.cpp — fp16 conversion and TimeLog helpers.
// Pure CPU; no QNN runtime or NPU involved.

#include "utils.h"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

#include "QnnTypeMacros.hpp"
#include "QnnTypes.h"

namespace {

// IEEE-754 half bit patterns for a few exact, representable values.
constexpr uint16_t kHalfZero    = 0x0000;
constexpr uint16_t kHalfNegZero = 0x8000;
constexpr uint16_t kHalfOne     = 0x3C00;
constexpr uint16_t kHalfTwo     = 0x4000;
constexpr uint16_t kHalfHalf    = 0x3800;  // 0.5
constexpr uint16_t kHalfNegOne  = 0xBC00;
constexpr uint16_t kHalfPosInf  = 0x7C00;
constexpr uint16_t kHalfNegInf  = 0xFC00;

uint16_t toHalf(float f) {
    uint16_t h = 0;
    geniex::floatToFloat16(&h, &f, 1);
    return h;
}

float fromHalf(uint16_t h) {
    float f = 0.f;
    geniex::float16ToFloat(&f, &h, 1);
    return f;
}

}  // namespace

// ─── floatToFloat16: exact, representable values ─────────────────────────────

TEST(FloatToFloat16, ExactValues) {
    EXPECT_EQ(toHalf(0.0f), kHalfZero);
    EXPECT_EQ(toHalf(-0.0f), kHalfNegZero);
    EXPECT_EQ(toHalf(1.0f), kHalfOne);
    EXPECT_EQ(toHalf(2.0f), kHalfTwo);
    EXPECT_EQ(toHalf(0.5f), kHalfHalf);
    EXPECT_EQ(toHalf(-1.0f), kHalfNegOne);
}

TEST(FloatToFloat16, OverflowSaturatesToInf) {
    EXPECT_EQ(toHalf(1.0e30f), kHalfPosInf);
    EXPECT_EQ(toHalf(-1.0e30f), kHalfNegInf);
    EXPECT_EQ(toHalf(std::numeric_limits<float>::infinity()), kHalfPosInf);
}

TEST(FloatToFloat16, UnderflowFlushesToZero) {
    // Far below the smallest subnormal half (2^-24 ≈ 5.96e-8).
    EXPECT_EQ(toHalf(1.0e-12f), kHalfZero);
    EXPECT_EQ(toHalf(-1.0e-12f), kHalfNegZero);
}

// A value in the subnormal-half range (2^-24 .. 2^-14) exercises the subnormal
// encode path; it must round-trip back to (approximately) itself.
TEST(FloatToFloat16, SubnormalEncodeRoundTrips) {
    const float v    = std::ldexp(1.0f, -20);  // 2^-20, a representable subnormal half
    const float back = fromHalf(toHalf(v));
    EXPECT_NEAR(back, v, std::ldexp(1.0f, -24));  // within one subnormal step
    EXPECT_GT(back, 0.0f);                        // not flushed to zero
}

// RNE: 1 + 2^-11 is the exact midpoint between 1.0 and 1.0+2^-10; the even
// neighbour (1.0, mantissa LSB 0) must win.
TEST(FloatToFloat16, RoundsToNearestEven) {
    const float midpoint = 1.0f + std::ldexp(1.0f, -11);
    EXPECT_EQ(toHalf(midpoint), kHalfOne);
}

// ─── float16ToFloat: inverse direction ───────────────────────────────────────

TEST(Float16ToFloat, ExactValues) {
    EXPECT_FLOAT_EQ(fromHalf(kHalfZero), 0.0f);
    EXPECT_FLOAT_EQ(fromHalf(kHalfOne), 1.0f);
    EXPECT_FLOAT_EQ(fromHalf(kHalfTwo), 2.0f);
    EXPECT_FLOAT_EQ(fromHalf(kHalfHalf), 0.5f);
    EXPECT_FLOAT_EQ(fromHalf(kHalfNegOne), -1.0f);
}

TEST(Float16ToFloat, Infinities) {
    EXPECT_TRUE(std::isinf(fromHalf(kHalfPosInf)));
    EXPECT_GT(fromHalf(kHalfPosInf), 0.0f);
    EXPECT_TRUE(std::isinf(fromHalf(kHalfNegInf)));
    EXPECT_LT(fromHalf(kHalfNegInf), 0.0f);
}

TEST(Float16ToFloat, Subnormal) {
    // Smallest positive subnormal half = 2^-24.
    const float v = fromHalf(0x0001);
    EXPECT_FLOAT_EQ(v, std::ldexp(1.0f, -24));
}

// ─── round-trip ──────────────────────────────────────────────────────────────

TEST(Float16RoundTrip, RepresentableValuesAreExact) {
    const std::vector<float> vals = {0.0f, 1.0f, -1.0f, 0.5f, 2.0f, 4.0f, 0.25f, -8.0f, 100.0f};
    for (float v : vals) {
        EXPECT_FLOAT_EQ(fromHalf(toHalf(v)), v) << "value " << v;
    }
}

TEST(Float16RoundTrip, BatchConversionMatchesScalar) {
    const std::vector<float> in = {0.1f, -0.2f, 3.14159f, -2.71828f, 0.0f, 65504.0f};
    std::vector<uint16_t>    halves(in.size());
    std::vector<float>       out(in.size());

    geniex::floatToFloat16(halves.data(), in.data(), in.size());
    geniex::float16ToFloat(out.data(), halves.data(), in.size());

    for (size_t i = 0; i < in.size(); ++i) {
        // fp16 has ~3 decimal digits; tolerance scales with magnitude.
        EXPECT_NEAR(out[i], in[i], std::abs(in[i]) * 1e-3f + 1e-7f) << "index " << i;
    }
}

// ─── TimeLog helpers ─────────────────────────────────────────────────────────

TEST(TotalMs, SumsDurationField) {
    // Note: the value field holds microseconds; totalMs sums it as-is.
    geniex::TimeLog log;
    log["a"] = {10.0, 1};
    log["b"] = {25.5, 3};
    EXPECT_DOUBLE_EQ(geniex::totalMs(log), 35.5);
}

TEST(TotalMs, EmptyIsZero) {
    geniex::TimeLog log;
    EXPECT_DOUBLE_EQ(geniex::totalMs(log), 0.0);
}

TEST(MergeTimeLogs, AccumulatesDurationAndCount) {
    geniex::TimeLog dst;
    dst["graph0"] = {100.0, 2};

    geniex::TimeLog src;
    src["graph0"] = {50.0, 1};  // existing key → accumulate
    src["graph1"] = {30.0, 4};  // new key → inserted

    geniex::mergeTimeLogs(dst, src);

    EXPECT_DOUBLE_EQ(dst["graph0"].first, 150.0);
    EXPECT_EQ(dst["graph0"].second, 3);
    EXPECT_DOUBLE_EQ(dst["graph1"].first, 30.0);
    EXPECT_EQ(dst["graph1"].second, 4);
}

TEST(MergeTimeLogs, EmptySrcLeavesDstUnchanged) {
    geniex::TimeLog dst;
    dst["x"] = {7.0, 1};
    geniex::mergeTimeLogs(dst, geniex::TimeLog{});
    EXPECT_DOUBLE_EQ(dst["x"].first, 7.0);
    EXPECT_EQ(dst["x"].second, 1);
}

// printTimings formats and logs; it must not throw on populated or empty logs.
TEST(PrintTimings, DoesNotThrow) {
    geniex::TimeLog log;
    log["graph0"] = {12.5, 3};
    log["graph1"] = {0.0, 0};
    EXPECT_NO_THROW(geniex::printTimings(log));
    EXPECT_NO_THROW(geniex::printTimings(geniex::TimeLog{}));
}

// ─── Quantize / dequantize (scale-offset) ──────────────────────────────────

// uint8 quant→dequant round-trip stays within one quantization step.
TEST(FloatToTfN, Uint8RoundTripWithinStep) {
    const float              scale  = 0.1f;
    const int32_t            offset = -10;  // representable ≈ [-1.0, 24.5]
    const std::vector<float> src    = {0.0f, 1.0f, 2.5f, 5.0f};

    std::vector<uint8_t> q(src.size());
    geniex::floatToTfN(q.data(), src.data(), offset, scale, src.size());

    std::vector<float> deq(src.size());
    geniex::tfNToFloat(deq.data(), q.data(), offset, scale, q.size());

    for (size_t i = 0; i < src.size(); ++i) EXPECT_NEAR(deq[i], src[i], scale) << "index " << i;
}

// uint16 gives finer resolution than uint8.
TEST(FloatToTfN, Uint16RoundTripWithinStep) {
    const float              scale  = 0.001f;
    const int32_t            offset = 0;
    const std::vector<float> src    = {0.5f, 1.25f, 10.0f};

    std::vector<uint16_t> q(src.size());
    geniex::floatToTfN(q.data(), src.data(), offset, scale, src.size());

    std::vector<float> deq(src.size());
    geniex::tfNToFloat(deq.data(), q.data(), offset, scale, q.size());

    for (size_t i = 0; i < src.size(); ++i) EXPECT_NEAR(deq[i], src[i], scale) << "index " << i;
}

// Values outside the representable range clamp to [0, max].
TEST(FloatToTfN, ClampsToRange) {
    const float              scale  = 0.1f;
    const int32_t            offset = 0;  // representable [0, 25.5]
    const std::vector<float> src    = {-5.0f, 100.0f};

    std::vector<uint8_t> q(src.size());
    geniex::floatToTfN(q.data(), src.data(), offset, scale, src.size());

    EXPECT_EQ(q[0], 0);    // below range → 0
    EXPECT_EQ(q[1], 255);  // above range → max
}

// offset==0, scale==1 maps integers in [0,255] to themselves.
TEST(FloatToTfN, IdentityEncoding) {
    const std::vector<float> src = {0.0f, 1.0f, 200.0f, 255.0f};
    std::vector<uint8_t>     q(src.size());
    geniex::floatToTfN(q.data(), src.data(), /*offset=*/0, /*scale=*/1.0f, src.size());
    EXPECT_EQ(q, (std::vector<uint8_t>{0, 1, 200, 255}));
}

// RoundingMode::Nearest rounds to nearest instead of toward zero. Truncating
// biases every element down by up to a full LSB (mean -0.5) instead of a
// zero-mean +/-0.5, which shows up as a constant offset once a whole tensor is
// quantized through this path -- and disagrees with Genie / the QNN SDK's own
// datautil::floatToTfN. Callers whose graph is calibrated this way (e.g. Gemma4
// vision embeddings) select it explicitly.
TEST(FloatToTfN, RoundsToNearest) {
    const std::vector<float> src = {0.4f, 0.6f, 1.5f, 2.49f, 200.7f};
    std::vector<uint8_t>     q(src.size());
    geniex::floatToTfN(q.data(), src.data(), /*offset=*/0, /*scale=*/1.0f, src.size(), geniex::RoundingMode::Nearest);
    EXPECT_EQ(q, (std::vector<uint8_t>{0, 1, 2, 2, 201}));
}

// The default rounding mode is TowardZero: it truncates the fractional part,
// preserving the byte-for-byte behavior callers were written against (e.g. RoPE
// position_ids_cos/sin). Passing no mode must match an explicit TowardZero.
TEST(FloatToTfN, DefaultRoundingIsTowardZero) {
    const std::vector<float> src = {0.4f, 0.6f, 1.5f, 2.49f, 200.7f};
    std::vector<uint8_t>     defaulted(src.size());
    std::vector<uint8_t>     explicit_tz(src.size());
    geniex::floatToTfN(defaulted.data(), src.data(), /*offset=*/0, /*scale=*/1.0f, src.size());
    geniex::floatToTfN(
        explicit_tz.data(), src.data(), /*offset=*/0, /*scale=*/1.0f, src.size(), geniex::RoundingMode::TowardZero);
    EXPECT_EQ(defaulted, (std::vector<uint8_t>{0, 0, 1, 2, 200}));
    EXPECT_EQ(defaulted, explicit_tz);
}

// RoundingMode::TowardZero truncates the fractional part. This is the default,
// and callers whose graph is calibrated against truncation (e.g. RoPE
// position_ids_cos/sin) rely on it; RoundingMode::Nearest would shift those
// codes by up to 1 LSB.
TEST(FloatToTfN, TowardZeroTruncates) {
    const std::vector<float> src = {0.4f, 0.6f, 1.5f, 2.49f, 200.7f};
    std::vector<uint8_t>     q(src.size());
    geniex::floatToTfN(
        q.data(), src.data(), /*offset=*/0, /*scale=*/1.0f, src.size(), geniex::RoundingMode::TowardZero);
    EXPECT_EQ(q, (std::vector<uint8_t>{0, 0, 1, 2, 200}));
}

// The two rounding modes disagree by exactly 1 LSB on values whose quantized
// magnitude has a fractional part >= 0.5 -- the divergence that regressed
// on-device RoPE. Guards against silently collapsing the two modes into one.
TEST(FloatToTfN, NearestAndTowardZeroDifferByOneLsb) {
    // scale 1, offset 0: q == the source value, so 2.5 sits exactly mid-code.
    const float              scale  = 1.0f;
    const int32_t            offset = 0;
    const std::vector<float> src    = {2.5f};

    uint16_t nearest = 0, toward_zero = 0;
    geniex::floatToTfN(&nearest, src.data(), offset, scale, 1, geniex::RoundingMode::Nearest);
    geniex::floatToTfN(&toward_zero, src.data(), offset, scale, 1, geniex::RoundingMode::TowardZero);
    EXPECT_EQ(nearest, 3) << "nearest=" << nearest;
    EXPECT_EQ(toward_zero, 2) << "toward_zero=" << toward_zero;
    EXPECT_EQ(nearest - toward_zero, 1) << "nearest=" << nearest << " toward_zero=" << toward_zero;
}

TEST(TfNToFloat, AppliesScaleOffset) {
    const std::vector<uint8_t> q = {0, 10, 20};
    std::vector<float>         out(q.size());
    geniex::tfNToFloat(out.data(), q.data(), /*offset=*/-5, /*scale=*/0.5f, q.size());
    // (q + offset) * scale
    EXPECT_EQ(out, (std::vector<float>{-2.5f, 2.5f, 7.5f}));
}

// ─── Element-wise casts ────────────────────────────────────────────────

TEST(CastToFloat, FromIntegerTypes) {
    const std::vector<int32_t> in = {-3, 0, 7, 1000};
    std::vector<float>         out(in.size());
    geniex::castToFloat(out.data(), in.data(), in.size());
    EXPECT_EQ(out, (std::vector<float>{-3.0f, 0.0f, 7.0f, 1000.0f}));
}

TEST(CastFromFloat, ToInt32Truncates) {
    const std::vector<float> in = {-1.9f, 0.0f, 2.7f, 42.0f};
    std::vector<int32_t>     out(in.size());
    geniex::castFromFloat(out.data(), in.data(), in.size());
    EXPECT_EQ(out, (std::vector<int32_t>{-1, 0, 2, 42}));  // truncation toward zero
}

// ─── tensorByteSize ───────────────────────────────────────────────────

namespace {

// Minimal Qnn_Tensor_t with a dtype + dims, for byte-size queries.
Qnn_Tensor_t makeTensor(Qnn_DataType_t dtype, std::vector<uint32_t>& dims) {
    Qnn_Tensor_t t = QNN_TENSOR_INIT;
    QNN_TENSOR_SET_DATA_TYPE(t, dtype);
    QNN_TENSOR_SET_RANK(t, static_cast<uint32_t>(dims.size()));
    QNN_TENSOR_SET_DIMENSIONS(t, dims.data());
    return t;
}

}  // namespace

TEST(TensorByteSize, ProductOfDimsTimesDtypeSize) {
    std::vector<uint32_t> d23 = {2, 3};
    auto                  f32 = makeTensor(QNN_DATATYPE_FLOAT_32, d23);
    EXPECT_EQ(geniex::tensorByteSize(&f32), 2u * 3u * 4u);

    std::vector<uint32_t> d4 = {4};
    auto                  u8 = makeTensor(QNN_DATATYPE_UFIXED_POINT_8, d4);
    EXPECT_EQ(geniex::tensorByteSize(&u8), 4u * 1u);

    std::vector<uint32_t> d5  = {5};
    auto                  f16 = makeTensor(QNN_DATATYPE_FLOAT_16, d5);
    EXPECT_EQ(geniex::tensorByteSize(&f16), 5u * 2u);
}

TEST(TensorByteSize, UnsupportedDtypeThrows) {
    std::vector<uint32_t> d1  = {1};
    auto                  bad = makeTensor(QNN_DATATYPE_UNDEFINED, d1);
    EXPECT_THROW(geniex::tensorByteSize(&bad), std::runtime_error);
}
