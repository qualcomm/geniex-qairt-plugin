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
    uint64_t, bool, bool, const std::vector<std::string>&, bool, bool, uint32_t, LogCallback,
    const std::vector<size_t>&) {
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
// Position-scripted peaks: peak = g_token_script[abs_pos]. Takes precedence over
// g_next_token when non-empty. See stubSetTokenScript.
std::vector<int32_t> g_token_script;
// Per-row absolute sequence position of the batch currently in flight, recovered
// from the most recent body graph's attention mask. A multi-shard model runs the
// body (which owns the mask) and the LM head (which owns `logits`) as separate
// graphExecutes, so the position must be carried across them: the body execute
// fills this, the LM-head execute (or a single-shard execute) reads it.
std::vector<size_t> g_row_abs_pos;
}  // namespace

namespace geniex::testing {
void stubSetNextToken(int32_t token_id) { g_next_token = token_id; }
void stubSetVocabSize(uint32_t vocab_size) { g_vocab_size = vocab_size; }
void stubSetTokenScript(std::vector<int32_t> script) {
    g_token_script = std::move(script);
    g_row_abs_pos.clear();
}
}  // namespace geniex::testing

namespace {
// Records each decode row's absolute sequence position from an attention mask
// input, keying scripted logits peaks. A row's committed-history length (n_past)
// is its leading run of attended (0.0f) columns before the first masked (-1e9)
// entry; the row's own position is n_past + its batch offset (a linear
// verify/decode chain places row r at depth r). The mask is [rows, row_len] and
// its dims come straight from the tensor, so this works for ar==1 decode, the
// batched verify pass, and the draft tree alike.
//
// A multi-shard model runs the body graph (which owns the mask) and the LM head
// (which owns `logits`) as separate graphExecutes, so the recovered positions
// are cached in g_row_abs_pos for the sibling LM-head execute to consume.
void recordAbsPositions(const Qnn_Tensor_t* input, uint32_t n_inputs) {
    for (uint32_t i = 0; i < n_inputs; ++i) {
        if (std::string(QNN_TENSOR_GET_NAME(input[i])) != "attention_mask") continue;
        const Qnn_ClientBuffer_t& buf  = QNN_TENSOR_GET_CLIENT_BUF(input[i]);
        const uint32_t            rank = QNN_TENSOR_GET_RANK(input[i]);
        const uint32_t*           dims = QNN_TENSOR_GET_DIMENSIONS(input[i]);
        if (!buf.data || rank < 2 || !dims) return;
        const auto*  mask    = static_cast<const float*>(buf.data);
        const size_t rows    = dims[rank - 2];
        const size_t row_len = dims[rank - 1];
        g_row_abs_pos.assign(rows, 0);
        for (size_t r = 0; r < rows; ++r) {
            size_t       n_past = 0;
            const size_t base   = r * row_len;
            for (size_t j = 0; j < row_len; ++j) {
                if (mask[base + j] != 0.0f) break;
                ++n_past;
            }
            g_row_abs_pos[r] = n_past + r;  // linear-chain row depth == batch offset
        }
        return;
    }
}
}  // namespace

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

    // A body execute carries the attention mask: cache its per-row positions so a
    // scripted logits write on this or a sibling (LM-head) execute can use them.
    recordAbsPositions(input, graph_info->numInputTensors);

    // Override any output named "logits" with a one-hot peak so the LM head's
    // argmax is deterministic. Scripted mode peaks at g_token_script[abs_pos]
    // (abs_pos recovered per row from the attention mask, cached across shards);
    // otherwise every row peaks at the single g_next_token.
    const bool scripted = !g_token_script.empty();
    if ((g_next_token >= 0 || scripted) && g_vocab_size > 0) {
        for (uint32_t i = 0; i < graph_info->numOutputTensors; ++i) {
            if (std::string(QNN_TENSOR_GET_NAME(output[i])) != "logits") continue;
            const Qnn_ClientBuffer_t& dst = QNN_TENSOR_GET_CLIENT_BUF(output[i]);
            if (!dst.data) continue;
            auto*        logits = static_cast<float*>(dst.data);
            const size_t rows   = (dst.dataSize / sizeof(float)) / g_vocab_size;

            for (size_t r = 0; r < rows; ++r) {
                float* row = logits + r * g_vocab_size;
                std::fill_n(row, g_vocab_size, 0.0f);
                int32_t peak = g_next_token;
                if (scripted) {
                    const size_t abs_pos = r < g_row_abs_pos.size() ? g_row_abs_pos[r] : r;
                    peak                 = g_token_script[std::min(abs_pos, g_token_script.size() - 1)];
                }
                if (peak >= 0 && static_cast<uint32_t>(peak) < g_vocab_size) row[peak] = 1.0f;
            }
        }
    }
    return true;
}
