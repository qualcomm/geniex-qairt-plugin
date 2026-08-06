// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "IOTensor.hpp"
#include "QnnApi.hpp"
#include "QnnTypes.h"
#include "QnnWrapperUtils.hpp"
#include "geniex_export.h"
#include "types.h"
#include "utils.h"

namespace geniex {

class GENIEX_API Graph {
   public:
    // All pointers are non-owning and must outlive this Graph.
    Graph(qnn_wrapper_api::GraphInfo_t* graph_info, QnnApi* api, IOTensor* io_tensor);

    ~Graph();

    Graph(const Graph&)            = delete;
    Graph& operator=(const Graph&) = delete;
    Graph(Graph&&) noexcept;
    Graph& operator=(Graph&&) noexcept;

    // Must be called once before any write / execute / read.
    // QnnApi::initializeHtp() already allocated a fused RPC buffer; all CL/AR
    // variants of the same shard share the same physical buffer, so no ownership
    // is assumed here and tearDownTensors is never called.
    bool setup(Qnn_ContextHandle_t context);

    bool hasInput(const std::string& name) const;
    bool hasOutput(const std::string& name) const;

    const TensorSpec& inputSpec(const std::string& name) const;
    const TensorSpec& outputSpec(const std::string& name) const;

    const std::vector<TensorSpec>& inputSpecs() const;
    const std::vector<TensorSpec>& outputSpecs() const;

    const std::string& name() const;

    // Converts src to the tensor's native dtype and writes it into the named
    // input buffer.
    //   float / double:  memcpy for FLOAT_32 (narrowing for double),
    //                    fp16-narrow for FLOAT_16,
    //                    scale-offset quant for UFIXED_POINT_8/16 (rounding per
    //                    `rounding`; default truncate toward zero),
    //                    plain cast for INT_32.
    //                    The double overload preserves full precision through
    //                    the quant pipeline.
    //   int32_t:         direct memcpy, no quantization.
    void write(const std::string& name, const float* src, size_t element_count,
        RoundingMode rounding = RoundingMode::TowardZero);
    void write(const std::string& name, const double* src, size_t element_count,
        RoundingMode rounding = RoundingMode::TowardZero);
    void write(const std::string& name, const int32_t* src, size_t element_count);

    // Copies bytes verbatim with no type conversion.
    void write(const std::string& name, const void* src, size_t byte_count);
    void read(const std::string& name, void* dst, size_t byte_count) const;

    // De-quantises or casts the named output buffer into float32.
    // elem_offset: number of elements to skip before reading (for multi-row outputs).
    void read(const std::string& name, float* dst, size_t element_count, size_t elem_offset = 0) const;

    void*       inputPtr(const std::string& name);
    const void* inputPtr(const std::string& name) const;
    const void* outputPtr(const std::string& name) const;

    bool execute(std::map<std::string, std::pair<double, uint16_t>>& time_log);

   private:
    void buildSpecs();

    qnn_wrapper_api::GraphInfo_t* graph_info_ = nullptr;
    QnnApi*                       api_        = nullptr;
    IOTensor*                     io_tensor_  = nullptr;

    Qnn_Tensor_t* inputs_  = nullptr;
    Qnn_Tensor_t* outputs_ = nullptr;

    std::vector<TensorSpec> input_specs_;
    std::vector<TensorSpec> output_specs_;

    std::unordered_map<std::string, uint32_t> input_index_;
    std::unordered_map<std::string, uint32_t> output_index_;

    std::unordered_map<std::string, void*> input_buffer_ptrs_;
    std::unordered_map<std::string, void*> output_buffer_ptrs_;

    std::unordered_map<std::string, size_t> input_tensors_size_;
    std::unordered_map<std::string, size_t> output_tensors_size_;

    std::string name_;

    bool setup_done_ = false;
};

}  // namespace geniex
