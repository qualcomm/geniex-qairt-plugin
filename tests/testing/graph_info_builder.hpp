// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause
//
// Builds GraphInfo_t / Qnn_Tensor_t fixtures on the heap for Graph tests, with
// no QNN runtime. Uses the public QNN_TENSOR_SET_* macros to stay independent
// of the tensor union layout. Each tensor gets a CPU client buffer so a real
// IOTensor (default ClientBuffer mode) hands the pointer back via getBuffer().
// Owns all backing storage.

#pragma once

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <string>
#include <vector>

#include "QnnTypeMacros.hpp"
#include "QnnTypes.h"
#include "QnnWrapperUtils.hpp"

namespace geniex::testing {

inline size_t dtypeByteSize(Qnn_DataType_t dtype) {
    switch (dtype) {
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
        default:
            return 8;
    }
}

struct TensorDesc {
    std::string           name;
    Qnn_DataType_t        dtype;
    std::vector<uint32_t> dims;
    float                 scale  = 1.0f;  // scale-offset quant; used for *FIXED_POINT* dtypes
    int32_t               offset = 0;
};

// Owns the GraphInfo_t and every buffer it points at.
class GraphInfoBuilder {
   public:
    GraphInfoBuilder(
        std::string graph_name, const std::vector<TensorDesc>& inputs, const std::vector<TensorDesc>& outputs)
        : name_(std::move(graph_name)) {
        buildTensors(inputs, input_tensors_, input_dims_);
        buildTensors(outputs, output_tensors_, output_dims_);

        info_.graph            = nullptr;
        info_.graphName        = name_.data();
        info_.inputTensors     = input_tensors_.data();
        info_.numInputTensors  = static_cast<uint32_t>(input_tensors_.size());
        info_.outputTensors    = output_tensors_.data();
        info_.numOutputTensors = static_cast<uint32_t>(output_tensors_.size());
    }

    ~GraphInfoBuilder() {
        for (void* p : client_bufs_) std::free(p);
    }

    GraphInfoBuilder(const GraphInfoBuilder&)            = delete;
    GraphInfoBuilder& operator=(const GraphInfoBuilder&) = delete;

    qnn_wrapper_api::GraphInfo_t&       graphInfo() { return info_; }
    const qnn_wrapper_api::GraphInfo_t& graphInfo() const { return info_; }

   private:
    void buildTensors(const std::vector<TensorDesc>& descs, std::vector<Qnn_Tensor_t>& out,
        std::deque<std::vector<uint32_t>>& dim_store) {
        out.resize(descs.size());

        for (size_t i = 0; i < descs.size(); ++i) {
            const TensorDesc& d = descs[i];
            names_.push_back(d.name);
            dim_store.push_back(d.dims);

            size_t elems = 1;
            for (uint32_t dim : d.dims) elems *= dim;
            const size_t bytes = elems * dtypeByteSize(d.dtype);

            void* buf = std::malloc(bytes);
            std::memset(buf, 0, bytes);
            client_bufs_.push_back(buf);

            Qnn_Tensor_t t = QNN_TENSOR_INIT;
            QNN_TENSOR_SET_ID(t, static_cast<uint32_t>(client_bufs_.size() - 1));
            QNN_TENSOR_SET_NAME(t, names_.back().c_str());
            QNN_TENSOR_SET_TYPE(t, QNN_TENSOR_TYPE_APP_READWRITE);
            QNN_TENSOR_SET_DATA_FORMAT(t, QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER);
            QNN_TENSOR_SET_DATA_TYPE(t, d.dtype);
            QNN_TENSOR_SET_RANK(t, static_cast<uint32_t>(dim_store.back().size()));
            QNN_TENSOR_SET_DIMENSIONS(t, dim_store.back().data());
            QNN_TENSOR_SET_MEM_TYPE(t, QNN_TENSORMEMTYPE_RAW);

            Qnn_ClientBuffer_t cb{buf, static_cast<uint32_t>(bytes)};
            QNN_TENSOR_SET_CLIENT_BUF(t, cb);

            Qnn_QuantizeParams_t qp       = QNN_QUANTIZE_PARAMS_INIT;
            qp.encodingDefinition         = QNN_DEFINITION_DEFINED;
            qp.quantizationEncoding       = QNN_QUANTIZATION_ENCODING_SCALE_OFFSET;
            qp.scaleOffsetEncoding.scale  = d.scale;
            qp.scaleOffsetEncoding.offset = d.offset;
            QNN_TENSOR_SET_QUANT_PARAMS(t, qp);

            out[i] = t;
        }
    }

    std::string                       name_;
    std::deque<std::string>           names_;  // deque: stable c_str() across pushes
    std::deque<std::vector<uint32_t>> input_dims_;
    std::deque<std::vector<uint32_t>> output_dims_;
    std::vector<Qnn_Tensor_t>         input_tensors_;
    std::vector<Qnn_Tensor_t>         output_tensors_;
    std::vector<void*>                client_bufs_;
    qnn_wrapper_api::GraphInfo_t      info_{};
};

}  // namespace geniex::testing
