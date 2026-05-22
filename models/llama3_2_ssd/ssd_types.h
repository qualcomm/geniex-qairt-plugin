// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace geniex {

// Configuration for Self-Speculative Decoding (SSD).
// Drafts a tree of candidates in a single AR-32 pass, verifies greedily, and
// accepts as many as match — no separate draft model required.
struct SSDConfig {
    // Branching factor per level: branches[d] = children per parent at depth d.
    // branches={3,2} → root→3→2 = 10 nodes; 10*2 = 20 forecast tokens; 30 total per pass.
    std::vector<size_t> branches = {3, 2};

    // Number of pre-computed KV entries for forecast tokens, loaded from disk at init.
    // The KV cache is pre-filled to this offset before the first prefill.
    size_t forecast_prefix = 16;

    // Full path to the forecast prefix KV cache file.
    std::string forecast_prefix_path;

    // RoPE base frequency for tree-based position encoding during SSD decode.
    float rope_theta = 500000.0f;
};

// Binary file header for the QAIRT KV cache format used to load the forecast prefix.
#pragma pack(push, 1)
struct KVCacheFileHeader {
    uint32_t num_tensors;  // K+V tensors (2 per layer)
    uint32_t magic;        // 0xC0DE
    uint8_t  dtype;        // 0=uint8, 1=float16, 2=float32
    uint8_t  pad;
    uint16_t n_heads;      // KV heads
    uint16_t embed_dim;    // head_dim
    uint16_t update_size;  // KV entries stored
};
#pragma pack(pop)
static_assert(sizeof(KVCacheFileHeader) == 16, "KVCacheFileHeader must be 16 bytes");

}  // namespace geniex
