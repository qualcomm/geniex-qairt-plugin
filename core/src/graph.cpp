// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#include "graph.h"

#include <cmath>
#include <cstring>
#include <stdexcept>
#include <vector>

#include "QnnTypeMacros.hpp"
#include "QnnTypes.h"
#include "utils.h"

namespace geniex {

namespace {

// Shared dispatch for Graph::write(name, float*|double*, n). Templated on Src
// so the caller controls the input precision.
template <typename Src>
static void writeFloatLike(const std::string& tensor_name, const std::string& graph_name, const Qnn_Tensor_t& t,
    void* buf, const Src* src, size_t n) {
    static_assert(std::is_floating_point<Src>::value, "writeFloatLike: src must be floating-point");

    const size_t buf_bytes  = tensorByteSize(&t);
    const auto   dtype      = QNN_TENSOR_GET_DATA_TYPE(t);
    const size_t elem_bytes = (dtype == QNN_DATATYPE_FLOAT_32 || dtype == QNN_DATATYPE_INT_32)            ? 4
                              : (dtype == QNN_DATATYPE_FLOAT_16 || dtype == QNN_DATATYPE_UFIXED_POINT_16) ? 2
                                                                                                          : 1;
    const size_t needed     = n * elem_bytes;
    if (needed > buf_bytes) {
        throw std::runtime_error("Graph::write overflow on graph '" + graph_name + "' tensor '" + tensor_name +
                                 "': caller passed n=" + std::to_string(n) + " elements (" + std::to_string(needed) +
                                 " bytes) but buffer is only " + std::to_string(buf_bytes) + " bytes (" +
                                 std::to_string(buf_bytes / elem_bytes) + " elements)");
    }

    switch (dtype) {
        case QNN_DATATYPE_FLOAT_32:
            if constexpr (std::is_same_v<Src, float>) {
                std::memcpy(buf, src, n * sizeof(float));
            } else {
                // double -> float at the buffer boundary; downstream HTP graphs read float32.
                castFromFloat(static_cast<float*>(buf), src, n);
            }
            break;
        case QNN_DATATYPE_FLOAT_16: {
            // floatToFloat16 takes float; narrow once if Src is double.
            if constexpr (std::is_same_v<Src, float>) {
                floatToFloat16(static_cast<uint16_t*>(buf), src, n);
            } else {
                std::vector<float> tmp(n);
                for (size_t i = 0; i < n; ++i) tmp[i] = static_cast<float>(src[i]);
                floatToFloat16(static_cast<uint16_t*>(buf), tmp.data(), n);
            }
            break;
        }
        case QNN_DATATYPE_UFIXED_POINT_16: {
            const auto qp = QNN_TENSOR_GET_QUANT_PARAMS(t);
            if (qp.quantizationEncoding == QNN_QUANTIZATION_ENCODING_SCALE_OFFSET)
                floatToTfN(
                    static_cast<uint16_t*>(buf), src, qp.scaleOffsetEncoding.offset, qp.scaleOffsetEncoding.scale, n);
            else
                castFromFloat(static_cast<uint16_t*>(buf), src, n);
            break;
        }
        case QNN_DATATYPE_UFIXED_POINT_8: {
            const auto qp = QNN_TENSOR_GET_QUANT_PARAMS(t);
            if (qp.quantizationEncoding == QNN_QUANTIZATION_ENCODING_SCALE_OFFSET)
                floatToTfN(
                    static_cast<uint8_t*>(buf), src, qp.scaleOffsetEncoding.offset, qp.scaleOffsetEncoding.scale, n);
            else
                castFromFloat(static_cast<uint8_t*>(buf), src, n);
            break;
        }
        case QNN_DATATYPE_INT_32:
            castFromFloat(static_cast<int32_t*>(buf), src, n);
            break;
        default:
            throw std::runtime_error("Graph::write: unsupported dtype for '" + tensor_name + "'");
    }
}

}  // namespace

Graph::Graph(qnn_wrapper_api::GraphInfo_t* graph_info, QnnApi* api, IOTensor* io_tensor)
    : graph_info_(graph_info), api_(api), io_tensor_(io_tensor), name_(graph_info ? graph_info->graphName : "") {}

Graph::~Graph() {
    // inputs_ / outputs_ point into graph_info_->inputTensors / outputTensors,
    // which are owned by QnnApi.  Nothing to free here.
}

Graph::Graph(Graph&& other) noexcept
    : graph_info_(other.graph_info_),
      api_(other.api_),
      io_tensor_(other.io_tensor_),
      inputs_(other.inputs_),
      outputs_(other.outputs_),
      input_specs_(std::move(other.input_specs_)),
      output_specs_(std::move(other.output_specs_)),
      input_index_(std::move(other.input_index_)),
      output_index_(std::move(other.output_index_)),
      input_buffer_ptrs_(std::move(other.input_buffer_ptrs_)),
      output_buffer_ptrs_(std::move(other.output_buffer_ptrs_)),
      input_tensors_size_(std::move(other.input_tensors_size_)),
      output_tensors_size_(std::move(other.output_tensors_size_)),
      name_(std::move(other.name_)),
      setup_done_(other.setup_done_) {
    other.inputs_     = nullptr;
    other.outputs_    = nullptr;
    other.graph_info_ = nullptr;
    other.setup_done_ = false;
}

Graph& Graph::operator=(Graph&& other) noexcept {
    if (this != &other) {
        graph_info_          = other.graph_info_;
        api_                 = other.api_;
        io_tensor_           = other.io_tensor_;
        inputs_              = other.inputs_;
        outputs_             = other.outputs_;
        input_specs_         = std::move(other.input_specs_);
        output_specs_        = std::move(other.output_specs_);
        input_index_         = std::move(other.input_index_);
        output_index_        = std::move(other.output_index_);
        input_buffer_ptrs_   = std::move(other.input_buffer_ptrs_);
        output_buffer_ptrs_  = std::move(other.output_buffer_ptrs_);
        input_tensors_size_  = std::move(other.input_tensors_size_);
        output_tensors_size_ = std::move(other.output_tensors_size_);
        name_                = std::move(other.name_);
        setup_done_          = other.setup_done_;

        other.inputs_     = nullptr;
        other.outputs_    = nullptr;
        other.graph_info_ = nullptr;
        other.setup_done_ = false;
    }
    return *this;
}

bool Graph::setup(Qnn_ContextHandle_t /*context*/) {
    if (setup_done_) return true;

    inputs_  = graph_info_->inputTensors;
    outputs_ = graph_info_->outputTensors;

    for (uint32_t i = 0; i < graph_info_->numInputTensors; ++i) {
        const Qnn_Tensor_t& t  = graph_info_->inputTensors[i];
        std::string         n  = QNN_TENSOR_GET_NAME(t);
        input_tensors_size_[n] = tensorByteSize(&t);
        input_buffer_ptrs_[n]  = io_tensor_->getBuffer(&inputs_[i]);
    }
    for (uint32_t i = 0; i < graph_info_->numOutputTensors; ++i) {
        const Qnn_Tensor_t& t   = graph_info_->outputTensors[i];
        std::string         n   = QNN_TENSOR_GET_NAME(t);
        output_tensors_size_[n] = tensorByteSize(&t);
        output_buffer_ptrs_[n]  = io_tensor_->getBuffer(&outputs_[i]);
    }

    buildSpecs();
    setup_done_ = true;
    return true;
}

void Graph::buildSpecs() {
    auto makeSpec = [](const Qnn_Tensor_t& t) -> TensorSpec {
        TensorSpec spec;
        spec.name  = QNN_TENSOR_GET_NAME(t);
        spec.dtype = QNN_TENSOR_GET_DATA_TYPE(t);

        const uint32_t  rank = QNN_TENSOR_GET_RANK(t);
        const uint32_t* dims = QNN_TENSOR_GET_DIMENSIONS(t);
        spec.shape.assign(dims, dims + rank);

        const auto qp = QNN_TENSOR_GET_QUANT_PARAMS(t);
        if (qp.quantizationEncoding == QNN_QUANTIZATION_ENCODING_SCALE_OFFSET) {
            spec.quant_scale  = qp.scaleOffsetEncoding.scale;
            spec.quant_offset = qp.scaleOffsetEncoding.offset;
        }
        return spec;
    };

    input_specs_.resize(graph_info_->numInputTensors);
    for (uint32_t i = 0; i < graph_info_->numInputTensors; ++i) {
        input_specs_[i]                    = makeSpec(graph_info_->inputTensors[i]);
        input_index_[input_specs_[i].name] = i;
    }

    output_specs_.resize(graph_info_->numOutputTensors);
    for (uint32_t i = 0; i < graph_info_->numOutputTensors; ++i) {
        output_specs_[i]                     = makeSpec(graph_info_->outputTensors[i]);
        output_index_[output_specs_[i].name] = i;
    }
}

bool Graph::hasInput(const std::string& name) const { return input_index_.count(name) > 0; }

bool Graph::hasOutput(const std::string& name) const { return output_index_.count(name) > 0; }

const TensorSpec& Graph::inputSpec(const std::string& name) const { return input_specs_.at(input_index_.at(name)); }

const TensorSpec& Graph::outputSpec(const std::string& name) const { return output_specs_.at(output_index_.at(name)); }

const std::vector<TensorSpec>& Graph::inputSpecs() const { return input_specs_; }
const std::vector<TensorSpec>& Graph::outputSpecs() const { return output_specs_; }

const std::string& Graph::name() const { return name_; }

void Graph::write(const std::string& name, const float* src, size_t n) {
    void*               buf = input_buffer_ptrs_.at(name);
    const Qnn_Tensor_t& t   = inputs_[input_index_.at(name)];
    writeFloatLike(name, name_, t, buf, src, n);
}

void Graph::write(const std::string& name, const double* src, size_t n) {
    void*               buf = input_buffer_ptrs_.at(name);
    const Qnn_Tensor_t& t   = inputs_[input_index_.at(name)];
    writeFloatLike(name, name_, t, buf, src, n);
}

void Graph::write(const std::string& name, const int32_t* src, size_t n) {
    std::memcpy(input_buffer_ptrs_.at(name), src, n * sizeof(int32_t));
}

void Graph::write(const std::string& name, const void* src, size_t byte_count) {
    std::memcpy(input_buffer_ptrs_.at(name), src, byte_count);
}

void Graph::read(const std::string& name, void* dst, size_t byte_count) const {
    std::memcpy(dst, output_buffer_ptrs_.at(name), byte_count);
}

void Graph::read(const std::string& name, float* dst, size_t n, size_t elem_offset) const {
    const void*         buf = output_buffer_ptrs_.at(name);
    const Qnn_Tensor_t& t   = outputs_[output_index_.at(name)];

    switch (QNN_TENSOR_GET_DATA_TYPE(t)) {
        case QNN_DATATYPE_FLOAT_32: {
            const auto* p = static_cast<const float*>(buf) + elem_offset;
            std::memcpy(dst, p, n * sizeof(float));
            break;
        }
        case QNN_DATATYPE_FLOAT_16: {
            const auto* p = static_cast<const uint16_t*>(buf) + elem_offset;
            float16ToFloat(dst, p, n);
            break;
        }
        case QNN_DATATYPE_UFIXED_POINT_16: {
            const auto  qp = QNN_TENSOR_GET_QUANT_PARAMS(t);
            const auto* p  = static_cast<const uint16_t*>(buf) + elem_offset;
            if (qp.quantizationEncoding == QNN_QUANTIZATION_ENCODING_SCALE_OFFSET)
                tfNToFloat(dst, p, qp.scaleOffsetEncoding.offset, qp.scaleOffsetEncoding.scale, n);
            else
                castToFloat(dst, p, n);
            break;
        }
        case QNN_DATATYPE_UFIXED_POINT_8: {
            const auto  qp = QNN_TENSOR_GET_QUANT_PARAMS(t);
            const auto* p  = static_cast<const uint8_t*>(buf) + elem_offset;
            if (qp.quantizationEncoding == QNN_QUANTIZATION_ENCODING_SCALE_OFFSET)
                tfNToFloat(dst, p, qp.scaleOffsetEncoding.offset, qp.scaleOffsetEncoding.scale, n);
            else
                castToFloat(dst, p, n);
            break;
        }
        case QNN_DATATYPE_INT_32: {
            const auto* p = static_cast<const int32_t*>(buf) + elem_offset;
            castToFloat(dst, p, n);
            break;
        }
        default:
            throw std::runtime_error("Graph::read(float*): unsupported dtype for '" + name + "'");
    }
}

void* Graph::inputPtr(const std::string& name) { return input_buffer_ptrs_.at(name); }

const void* Graph::outputPtr(const std::string& name) const { return output_buffer_ptrs_.at(name); }

bool Graph::execute(std::map<std::string, std::pair<double, uint16_t>>& time_log) {
    return api_->graphExecute(graph_info_, inputs_, outputs_, time_log);
}

}  // namespace geniex
