// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#include "utils.h"

#include <cstring>
#include <iomanip>
#include <numeric>
#include <sstream>

#include "logging.h"

namespace geniex {

static uint16_t floatToHalf(float f) {
    // fp32 → fp16 with IEEE round-to-nearest-even, matching the reference
    // runtime's fp16_ieee_from_fp32_value. Plain truncation (mant >> 13) biases
    // every value low by up to ~1 ULP and shows up as a per-element mismatch in
    // fp16 graph inputs (e.g. the RoPE cos/sin table); rounding removes that.
    uint32_t x;
    std::memcpy(&x, &f, 4);
    const uint32_t sign = (x >> 16) & 0x8000;
    const int32_t  exp  = static_cast<int32_t>((x >> 23) & 0xFF) - 127 + 15;
    const uint32_t mant = x & 0x7FFFFF;

    if (exp >= 31) {  // overflow / inf / NaN → ±inf (NaN mantissa not preserved)
        return static_cast<uint16_t>(sign | 0x7C00);
    }
    if (exp <= 0) {
        // Subnormal half or underflow to ±0. Build the 10-bit mantissa with the
        // implicit leading 1, then shift right by (1 - exp) with RNE rounding.
        if (exp < -10) return static_cast<uint16_t>(sign);  // too small even for subnormal
        uint32_t       m         = mant | 0x800000;         // 24-bit significand
        const int32_t  shift     = 14 - exp;                // 13 + (1 - exp)
        const uint32_t half      = 1u << (shift - 1);
        const uint32_t round_bit = (m & half) != 0;
        const uint32_t sticky    = (m & (half - 1)) != 0;
        uint32_t       q         = m >> shift;
        if (round_bit && (sticky || (q & 1))) ++q;  // round-to-nearest-even
        return static_cast<uint16_t>(sign | q);
    }

    // Normal range. Round the 23-bit mantissa down to 10 bits, RNE.
    const uint32_t round_bit = (mant & 0x1000) != 0;  // bit 12 (the one being dropped's MSB)
    const uint32_t sticky    = (mant & 0x0FFF) != 0;  // bits 0..11
    uint32_t       q         = (static_cast<uint32_t>(exp) << 10) | (mant >> 13);
    q |= sign;
    if (round_bit && (sticky || (q & 1))) ++q;  // carry may ripple exp; that's correct
    return static_cast<uint16_t>(q);
}

static float halfToFloat(uint16_t h) {
    const uint32_t sign = (h & 0x8000) << 16;
    uint32_t       exp  = (h >> 10) & 0x1F;
    uint32_t       mant = h & 0x3FF;
    uint32_t       f;
    if (exp == 0) {
        if (mant == 0) {
            f = sign;  // ±0
        } else {
            // Subnormal half → normalized fp32. Shift the mantissa left until
            // its leading 1 reaches the implicit-bit position, adjusting the
            // exponent. Flushing these to zero (the previous behavior) drops
            // tiny but real graph-output values and breaks bit-exact matches.
            int32_t e = -1;
            do {
                mant <<= 1;
                ++e;
            } while ((mant & 0x400) == 0);
            mant &= 0x3FF;
            f = sign | (static_cast<uint32_t>(127 - 15 - e) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        f = sign | 0x7F800000 | (mant << 13);  // inf / NaN
    } else {
        f = sign | ((exp - 15 + 127) << 23) | (mant << 13);
    }
    float out;
    std::memcpy(&out, &f, 4);
    return out;
}

void floatToFloat16(uint16_t* out, const float* in, size_t n) {
    for (size_t i = 0; i < n; ++i) out[i] = floatToHalf(in[i]);
}

void float16ToFloat(float* out, const uint16_t* in, size_t n) {
    for (size_t i = 0; i < n; ++i) out[i] = halfToFloat(in[i]);
}

double totalMs(const TimeLog& log) {
    double sum = 0.0;
    for (const auto& kv : log) sum += kv.second.first;
    return sum;
}

void mergeTimeLogs(TimeLog& dst, const TimeLog& src) {
    for (const auto& kv : src) {
        dst[kv.first].first += kv.second.first;
        dst[kv.first].second += kv.second.second;
    }
}

void printTimings(const TimeLog& log) {
    std::ostringstream oss;
    for (const auto& kv : log) {
        oss << "  " << std::left << std::setw(60) << kv.first << std::right << std::setw(10) << std::fixed
            << std::setprecision(2) << kv.second.first << " us"
            << "  (" << kv.second.second << " calls)\n";
    }
    GENIEX_LOG_INFO("{}", oss.str());
}

}  // namespace geniex
