// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause
//
// Test-side controls for the link-time QnnApi stub (testing/stub_qnnapi.cpp).

#pragma once

#include <cstdint>
#include <vector>

namespace geniex::testing {

// When >= 0, the stub's graphExecute writes a one-hot peak at this token id
// into any output tensor named "logits" (every row), instead of the default
// identity copy. -1 restores pure identity-copy behaviour.
void stubSetNextToken(int32_t token_id);

// Vocabulary size used to interpret each logits row; must match the fixture's
// logits tensor inner dimension. Defaults to 0 (no logits write).
void stubSetVocabSize(uint32_t vocab_size);

// Position-scripted mode: the peak for a logits row is script[abs_pos], where
// abs_pos is the row's absolute sequence position recovered from its additive
// attention_mask (the count of attended committed-KV columns, plus the row's
// offset within the batch). This makes the argmax vary per position instead of
// being a single fixed token, so a plain autoregressive run and a speculative
// run over the same fixture must emit an identical sequence -- any tree-mask,
// draft-seed, or KV/position desync diverges here. Empty script disables the
// mode and restores stubSetNextToken behaviour.
void stubSetTokenScript(std::vector<int32_t> script);

}  // namespace geniex::testing
