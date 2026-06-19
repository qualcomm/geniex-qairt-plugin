// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

#include "QnnTypeMacros.hpp"
#include "QnnTypes.h"

namespace geniex {

// ── Convention for this header ────────────────────────────────────────────
// • Header-inline (defined here): templates, and small pure leaf helpers with no
//   cross-DLL ABI concern. Each consumer instantiates its own copy, so there is
//   no exported symbol and no `geniex_core` ↔ `geniex_vlm` ABI contract — and
//   they remain unit-testable by simple inclusion. Do NOT mark these GENIEX_API.
// • Concrete in utils.cpp (declared here): non-template functions with a single
//   shared definition inside geniex_core. Mark GENIEX_API only once a consumer in
//   another module (e.g. geniex_vlm) actually calls them across the DLL boundary;
//   exporting on spec only widens the ABI surface with no caller to justify it.

// Bit-level float16 ↔ float32 conversion; no hardware fp16 instructions required.
void floatToFloat16(uint16_t* out, const float* in, size_t n);
void float16ToFloat(float* out, const uint16_t* in, size_t n);

// ── Tensor numeric kernels (header-inline; geniex_core-internal) ─────────────
// Quantize / dequantize (scale-offset) and element-wise casts used by Graph I/O.
// Header-inline by design: templated and pure, so no DLL export is needed.

// Byte size of a QNN tensor = product(dims) * sizeof(dtype).
inline size_t tensorByteSize(const Qnn_Tensor_t* t) {
    auto dtypeBytes = [](Qnn_DataType_t d) -> size_t {
        switch (d) {
            case QNN_DATATYPE_INT_8:
            case QNN_DATATYPE_UINT_8:
            case QNN_DATATYPE_SFIXED_POINT_8:
            case QNN_DATATYPE_UFIXED_POINT_8:
            case QNN_DATATYPE_BOOL_8:
                return 1;
            case QNN_DATATYPE_INT_16:
            case QNN_DATATYPE_UINT_16:
            case QNN_DATATYPE_FLOAT_16:
            case QNN_DATATYPE_SFIXED_POINT_16:
            case QNN_DATATYPE_UFIXED_POINT_16:
                return 2;
            case QNN_DATATYPE_INT_32:
            case QNN_DATATYPE_UINT_32:
            case QNN_DATATYPE_FLOAT_32:
            case QNN_DATATYPE_SFIXED_POINT_32:
            case QNN_DATATYPE_UFIXED_POINT_32:
                return 4;
            case QNN_DATATYPE_INT_64:
            case QNN_DATATYPE_UINT_64:
            case QNN_DATATYPE_FLOAT_64:
                return 8;
            default:
                throw std::runtime_error("tensorByteSize: unsupported dtype");
        }
    };
    size_t n = 1;
    for (uint32_t d = 0; d < QNN_TENSOR_GET_RANK(t); ++d) n *= QNN_TENSOR_GET_DIMENSIONS(t)[d];
    return n * dtypeBytes(QNN_TENSOR_GET_DATA_TYPE(t));
}

// Quantize float → unsigned fixed-point (T = uint8/uint16) via scale-offset,
// truncating toward zero and clamping to [0, 2^bits - 1].
template <typename T, typename Src>
void floatToTfN(T* out, const Src* in, int32_t offset, float scale, size_t n) {
    static_assert(std::is_unsigned<T>::value, "floatToTfN: unsigned types only");
    static_assert(std::is_floating_point<Src>::value, "floatToTfN: src must be floating-point");
    const double max_val      = static_cast<double>((T)-1);  // 2^bits - 1
    const double encoding_min = offset * static_cast<double>(scale);
    const double encoding_max = (max_val + offset) * static_cast<double>(scale);
    const double range        = encoding_max - encoding_min;

    for (size_t i = 0; i < n; ++i) {
        double q = max_val * (static_cast<double>(in[i]) - encoding_min) / range;
        if (q < 0.0)
            q = 0.0;
        else if (q > max_val)
            q = max_val;
        out[i] = static_cast<T>(q);  // truncation toward zero
    }
}

// Dequantize unsigned fixed-point (T = uint8/uint16) → float via scale-offset.
template <typename T>
void tfNToFloat(float* out, const T* in, int32_t offset, float scale, size_t n) {
    static_assert(std::is_unsigned<T>::value, "tfNToFloat: unsigned types only");
    for (size_t i = 0; i < n; ++i)
        out[i] = static_cast<float>((static_cast<double>(in[i]) + static_cast<double>(offset)) * scale);
}

// Element-wise cast to float.
template <typename T>
void castToFloat(float* out, const T* in, size_t n) {
    for (size_t i = 0; i < n; ++i) out[i] = static_cast<float>(in[i]);
}

// Element-wise cast from a floating-point source to Dst.
template <typename Dst, typename Src>
void castFromFloat(Dst* out, const Src* in, size_t n) {
    static_assert(std::is_floating_point<Src>::value, "castFromFloat: src must be floating-point");
    for (size_t i = 0; i < n; ++i) out[i] = static_cast<Dst>(in[i]);
}

// Timing map produced by Graph::execute().
// key = graph/op name, value = { cumulative_duration_us, call_count }
using TimeLog = std::map<std::string, std::pair<double, uint16_t>>;

double totalMs(const TimeLog& log);
void   mergeTimeLogs(TimeLog& dst, const TimeLog& src);

void printTimings(const TimeLog& log);

}  // namespace geniex
