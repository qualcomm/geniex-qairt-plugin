// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause
//
// Link-time stub for QnnApi used by Graph tests; the real QnnApi.cpp is not
// linked. graphExecute is the only QnnApi method Graph calls.

#include "stub_qnnapi.hpp"

#include <algorithm>
#include <cstring>
#include <string>

#include "BackendExtensions.hpp"
#include "QnnApi.hpp"
#include "QnnTypeMacros.hpp"

// QnnApi holds unique_ptr<BackendExtensions>; ~QnnApi needs this symbol even
// though the pointer is always null in stub-constructed instances.
BackendExtensions::~BackendExtensions() = default;

QnnApi::~QnnApi() = default;

// Device bring-up methods referenced by Model::initialize(), which the tests
// never call (they inject graphs via the TestableLLMModel subclass instead).
// Present only to satisfy the linker.
bool QnnApi::initializeHtp(std::string, std::vector<std::string>, BackendExtensionsConfigs,
    qnn::tools::netrun::PerfProfile, std::vector<GraphConfigs>, bool, std::string, bool, int64_t, uint32_t, bool, bool,
    uint64_t, bool, bool, const std::vector<std::string>&, bool, bool, uint32_t, LogCallback) {
    return false;
}

bool QnnApi::setPerfProfile(qualla::PerformanceProfile&) { return false; }

uint32_t QnnApi::getHtpDeviceNumCores() { return 0; }

bool QnnApi::setHtpNumCores(uint32_t) { return false; }

// Static helpers referenced by qualla::QnnUtils::Tensor (pulled in via
// qnn-utils.cpp) but never exercised by the orchestration tests.
bool QnnApi::getTensorQuantParams(const Qnn_Tensor_t*, std::vector<qualla::QnnUtils::QuantParam>&) { return false; }

bool QnnApi::getTensorShape(std::vector<size_t>&, const Qnn_Tensor_t&) { return false; }

namespace {
// LLM-orchestration tests drive decode by telling the stub which token to emit;
// graphExecute then writes a one-hot logits peak there so sampleNextToken()'s
// argmax fast path is deterministic. -1 disables (pure identity copy).
int32_t  g_next_token = -1;
uint32_t g_vocab_size = 0;
}  // namespace

namespace geniex::testing {
void stubSetNextToken(int32_t token_id) { g_next_token = token_id; }
void stubSetVocabSize(uint32_t vocab_size) { g_vocab_size = vocab_size; }
}  // namespace geniex::testing

bool QnnApi::graphExecute(qnn_wrapper_api::GraphInfo_t* graph_info, const Qnn_Tensor_t* input, Qnn_Tensor_t* output,
    std::map<std::string, std::pair<double, uint16_t>>& /*timeLogs*/) {
    // Identity: copy each input client buffer into the output at the same index
    // (byte-for-byte), so write->execute->read round-trips are deterministic.
    const uint32_t n = std::min(graph_info->numInputTensors, graph_info->numOutputTensors);
    for (uint32_t i = 0; i < n; ++i) {
        const Qnn_ClientBuffer_t& src   = QNN_TENSOR_GET_CLIENT_BUF(input[i]);
        const Qnn_ClientBuffer_t& dst   = QNN_TENSOR_GET_CLIENT_BUF(output[i]);
        const uint32_t            bytes = std::min(src.dataSize, dst.dataSize);
        if (src.data && dst.data) std::memcpy(dst.data, src.data, bytes);
    }

    // Override any output named "logits" with a one-hot peak at g_next_token so
    // the LM head's argmax is the token the test asked for, on every row.
    if (g_next_token >= 0 && g_vocab_size > 0) {
        for (uint32_t i = 0; i < graph_info->numOutputTensors; ++i) {
            if (std::string(QNN_TENSOR_GET_NAME(output[i])) != "logits") continue;
            const Qnn_ClientBuffer_t& dst = QNN_TENSOR_GET_CLIENT_BUF(output[i]);
            if (!dst.data) continue;
            auto*        logits = static_cast<float*>(dst.data);
            const size_t rows   = (dst.dataSize / sizeof(float)) / g_vocab_size;
            for (size_t r = 0; r < rows; ++r) {
                float* row = logits + r * g_vocab_size;
                std::fill_n(row, g_vocab_size, 0.0f);
                row[g_next_token] = 1.0f;
            }
        }
    }
    return true;
}
