// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause
//
// Link-time stub for QnnApi used by Graph tests; the real QnnApi.cpp is not
// linked. graphExecute is the only QnnApi method Graph calls.

#include <algorithm>
#include <cstring>

#include "BackendExtensions.hpp"
#include "QnnApi.hpp"
#include "QnnTypeMacros.hpp"

// QnnApi holds unique_ptr<BackendExtensions>; ~QnnApi needs this symbol even
// though the pointer is always null in stub-constructed instances.
BackendExtensions::~BackendExtensions() = default;

QnnApi::~QnnApi() = default;

bool QnnApi::graphExecute(qnn_wrapper_api::GraphInfo_t* graph_info, const Qnn_Tensor_t* input, Qnn_Tensor_t* output,
    std::map<std::string, std::pair<double, uint16_t>>& /*timeLogs*/) {
    // Identity: copy each input client buffer into the output at the same index
    // (byte-for-byte), so write→execute→read round-trips are deterministic.
    const uint32_t n = std::min(graph_info->numInputTensors, graph_info->numOutputTensors);
    for (uint32_t i = 0; i < n; ++i) {
        const Qnn_ClientBuffer_t& src   = QNN_TENSOR_GET_CLIENT_BUF(input[i]);
        const Qnn_ClientBuffer_t& dst   = QNN_TENSOR_GET_CLIENT_BUF(output[i]);
        const uint32_t            bytes = std::min(src.dataSize, dst.dataSize);
        if (src.data && dst.data) std::memcpy(dst.data, src.data, bytes);
    }
    return true;
}
