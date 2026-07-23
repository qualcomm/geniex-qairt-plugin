// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "geniex_export.h"
#include "llm/llm_types.h"

namespace geniex {

// A read-only, memory-mapped [vocab_size, row_width] embedding table whose
// elements are stored quantized.
//
// Why mmap rather than a std::vector<float>: these tables are the largest
// artifact in a large-vocab bundle. Gemma4-E2B's per-layer table is
// 262144 x 8960 int8 = 2.35 GB on disk and 9.4 GB dequantized -- more than
// the RAM of the machines we target. Decode only ever touches one row per
// token, so we let the OS page cache hold the working set and convert rows on
// demand. This mirrors what Genie does with `use-mmap`.
//
// Layout is row-major and headerless; vocab_size is derived from the file size.
class GENIEX_API QuantizedLut {
   public:
    QuantizedLut() = default;
    ~QuantizedLut();

    QuantizedLut(const QuantizedLut&)            = delete;
    QuantizedLut& operator=(const QuantizedLut&) = delete;
    QuantizedLut(QuantizedLut&&) noexcept;
    QuantizedLut& operator=(QuantizedLut&&) noexcept;

    // Maps `path`. `row_width` is the element count of one row. Throws if the
    // file is missing or its size is not a whole number of rows.
    void open(const std::string& path, size_t row_width, const QuantizedLutSpec& spec);

    bool isOpen() const { return base_ != nullptr; }

    size_t vocabSize() const { return vocab_size_; }
    size_t rowWidth() const { return row_width_; }
    size_t rowBytes() const { return row_bytes_; }

    const QuantizedLutSpec& spec() const { return spec_; }

    // Raw pointer to row `token`'s stored bytes, or nullptr if out of range.
    // Valid for rowBytes() bytes. Use with byte-copy fast paths.
    const void* rowBytesPtr(int32_t token) const;

    // Appends row `token` to `out`, dequantized to float. Out-of-range tokens
    // append `row_width` zeros so a bad id degrades to a zero embedding rather
    // than reading out of bounds.
    void gatherRow(int32_t token, std::vector<float>& out) const;

    // Dequantizes row `token` into `dst` (must hold row_width floats).
    void dequantizeRow(int32_t token, float* dst) const;

   private:
    void closeMapping();

    void*            base_       = nullptr;  // start of mapping
    size_t           map_bytes_  = 0;
    size_t           row_width_  = 0;
    size_t           row_bytes_  = 0;
    size_t           vocab_size_ = 0;
    QuantizedLutSpec spec_;

#ifdef _WIN32
    void* file_handle_    = nullptr;  // HANDLE
    void* mapping_handle_ = nullptr;  // HANDLE
#else
    int fd_ = -1;
#endif
};

}  // namespace geniex
