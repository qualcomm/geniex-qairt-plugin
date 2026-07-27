// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#include "llm/quantized_lut.h"

#include <stdexcept>
#include <utility>

#include "logging.h"

#ifdef _WIN32
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace geniex {

namespace {

// real = scale * (stored + offset)  -- QNN convention, offset = -zero_point.
template <typename T>
inline void dequantInto(const void* src, size_t n, float scale, int32_t offset, float* dst) {
    const T* q = static_cast<const T*>(src);
    for (size_t i = 0; i < n; ++i) {
        dst[i] = scale * (static_cast<float>(q[i]) + static_cast<float>(offset));
    }
}

}  // namespace

QuantizedLut::~QuantizedLut() { closeMapping(); }

QuantizedLut::QuantizedLut(QuantizedLut&& o) noexcept { *this = std::move(o); }

QuantizedLut& QuantizedLut::operator=(QuantizedLut&& o) noexcept {
    if (this == &o) return *this;
    closeMapping();
    base_       = o.base_;
    map_bytes_  = o.map_bytes_;
    row_width_  = o.row_width_;
    row_bytes_  = o.row_bytes_;
    vocab_size_ = o.vocab_size_;
    spec_       = std::move(o.spec_);
#ifdef _WIN32
    file_handle_      = o.file_handle_;
    mapping_handle_   = o.mapping_handle_;
    o.file_handle_    = nullptr;
    o.mapping_handle_ = nullptr;
#else
    fd_   = o.fd_;
    o.fd_ = -1;
#endif
    o.base_       = nullptr;
    o.map_bytes_  = 0;
    o.vocab_size_ = 0;
    return *this;
}

void QuantizedLut::closeMapping() {
#ifdef _WIN32
    if (base_) UnmapViewOfFile(base_);
    if (mapping_handle_) CloseHandle(static_cast<HANDLE>(mapping_handle_));
    if (file_handle_ && file_handle_ != INVALID_HANDLE_VALUE) CloseHandle(static_cast<HANDLE>(file_handle_));
    mapping_handle_ = nullptr;
    file_handle_    = nullptr;
#else
    if (base_) munmap(base_, map_bytes_);
    if (fd_ >= 0) close(fd_);
    fd_ = -1;
#endif
    base_      = nullptr;
    map_bytes_ = 0;
}

void QuantizedLut::open(const std::string& path, size_t row_width, const QuantizedLutSpec& spec) {
    closeMapping();

    if (row_width == 0) {
        throw std::runtime_error("QuantizedLut: row_width must be non-zero for " + path);
    }
    spec_      = spec;
    row_width_ = row_width;
    row_bytes_ = row_width * spec.elementBytes();

#ifdef _WIN32
    HANDLE fh = CreateFileA(
        path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (fh == INVALID_HANDLE_VALUE) {
        throw std::runtime_error("QuantizedLut: cannot open " + path);
    }
    LARGE_INTEGER size{};
    if (!GetFileSizeEx(fh, &size)) {
        CloseHandle(fh);
        throw std::runtime_error("QuantizedLut: cannot stat " + path);
    }
    map_bytes_ = static_cast<size_t>(size.QuadPart);

    HANDLE mh = CreateFileMappingA(fh, nullptr, PAGE_READONLY, 0, 0, nullptr);
    if (!mh) {
        CloseHandle(fh);
        throw std::runtime_error("QuantizedLut: CreateFileMapping failed for " + path);
    }
    void* view = MapViewOfFile(mh, FILE_MAP_READ, 0, 0, 0);
    if (!view) {
        CloseHandle(mh);
        CloseHandle(fh);
        throw std::runtime_error("QuantizedLut: MapViewOfFile failed for " + path);
    }
    file_handle_    = fh;
    mapping_handle_ = mh;
    base_           = view;
#else
    int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) {
        throw std::runtime_error("QuantizedLut: cannot open " + path);
    }
    struct stat st {};
    if (fstat(fd, &st) != 0) {
        ::close(fd);
        throw std::runtime_error("QuantizedLut: cannot stat " + path);
    }
    map_bytes_ = static_cast<size_t>(st.st_size);

    void* view = mmap(nullptr, map_bytes_, PROT_READ, MAP_SHARED, fd, 0);
    if (view == MAP_FAILED) {
        ::close(fd);
        throw std::runtime_error("QuantizedLut: mmap failed for " + path);
    }
    fd_   = fd;
    base_ = view;
#endif

    if (map_bytes_ % row_bytes_ != 0) {
        const size_t rb = row_bytes_, mb = map_bytes_;
        closeMapping();
        throw std::runtime_error("QuantizedLut: " + path + " size " + std::to_string(mb) +
                                 " is not a whole number of " + std::to_string(rb) + "-byte rows");
    }
    vocab_size_ = map_bytes_ / row_bytes_;

    // NOTE: geniex's logger pre-stringifies every argument through lp(), so only
    // plain "{}" works here -- a format spec like "{:.2f}" throws at runtime.
    GENIEX_LOG_DEBUG("QuantizedLut: mapped {} [{} x {}] {} scale={} offset={} ({} MB)",
        path,
        vocab_size_,
        row_width_,
        spec_.datatype,
        spec_.scale,
        spec_.offset,
        map_bytes_ / (1024 * 1024));
}

const void* QuantizedLut::rowBytesPtr(int32_t token) const {
    if (!base_ || token < 0 || static_cast<size_t>(token) >= vocab_size_) return nullptr;
    return static_cast<const uint8_t*>(base_) + static_cast<size_t>(token) * row_bytes_;
}

void QuantizedLut::dequantizeRow(int32_t token, float* dst) const {
    const void* src = rowBytesPtr(token);
    if (!src) {
        for (size_t i = 0; i < row_width_; ++i) dst[i] = 0.0f;
        return;
    }
    if (!spec_.quantized()) {
        const float* f = static_cast<const float*>(src);
        for (size_t i = 0; i < row_width_; ++i) dst[i] = f[i];
        return;
    }
    if (spec_.elementBytes() == 2) {
        if (spec_.isSigned())
            dequantInto<int16_t>(src, row_width_, spec_.scale, spec_.offset, dst);
        else
            dequantInto<uint16_t>(src, row_width_, spec_.scale, spec_.offset, dst);
    } else {
        if (spec_.isSigned())
            dequantInto<int8_t>(src, row_width_, spec_.scale, spec_.offset, dst);
        else
            dequantInto<uint8_t>(src, row_width_, spec_.scale, spec_.offset, dst);
    }
}

void QuantizedLut::gatherRow(int32_t token, std::vector<float>& out) const {
    const size_t base = out.size();
    out.resize(base + row_width_);
    dequantizeRow(token, out.data() + base);
}

}  // namespace geniex
