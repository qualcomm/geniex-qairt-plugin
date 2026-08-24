// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>

#include "QnnTypes.h"
#include "geniex_export.h"
#include "types.h"

// KV cache byte layouts.
//
// A KV tensor is logically [n_heads, 1, head_dim, kv_len] (keys) or
// [n_heads, 1, kv_len, head_dim] (values), but its *physical* bytes depend on
// the tensor's Qnn dataFormat:
//
//   FLAT_BUFFER       — row-major, what every exporter emitted until now.
//   HMX_WEIGHT_LAYOUT — tiled so the HTP's HMX units can consume the cache as a
//                       matmul weight operand with no on-device re-layout. Set
//                       by ENABLE_NATIVE_KV recipes. See docs/native-kv-cache.md.
//
// Everything here is a generic algorithm parameterized by tile sizes and tensor
// shapes; no model-specific constants. The authoritative definition of the tiled
// layout is Genie's fromFlatOffset(), qualla/engines/qnn-htp/KVCache/native-kv.cpp
// :53-81, which this file ports.
namespace geniex::kv {

// Nominal tile widths the QNN compiler uses for the HMX weight layout
// (native-kv.cpp:16-18). Every tile is this wide except a trailing partial one --
// see tileFor() and elementOffset().
constexpr size_t K_TILE_MAX = 256;  // key tiling along the kv_len (dout) axis
constexpr size_t V_TILE_MAX = 64;   // value tiling along the head_dim (dout) axis

// The tiled layout's innermost chunk is [din_1:8][dout_1:32][din_2:4] = 1024
// contiguous bytes spanning 32 din x 32 dout logical elements. It is therefore
// the unit of contiguity: a copy whose din/dout offsets and extents are
// 32-aligned degenerates to whole-block memcpys, which is what keeps restride()
// in the same cost class as the flat path.
constexpr size_t KV_BLOCK_BYTES = 1024;
constexpr size_t TILE_GRAIN     = 32;  // logical elements per axis per block

enum class KVFormat {
    Flat,     // QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER (row-major)
    HmxTiled  // QNN_TENSOR_DATA_FORMAT_HMX_WEIGHT_LAYOUT
};

GENIEX_API KVFormat    formatOf(const TensorSpec& spec);
GENIEX_API const char* formatName(KVFormat f);

// The compiler's nominal tile width along the dout axis.
//
// Genie writes min(DOUT, N_TILE), which is only correct when N_TILE divides DOUT
// -- true for its always-full-CL (scatter) caches. A non-scatter cache is not:
// the real Qwen3-4B natKV export carries kv_len 1920 and 2016, neither divisible
// by 256. There the LAST tile is simply narrower, and the din stride *within that
// tile* shrinks to match, which is what makes the layout fit the tensor exactly
// (1920 -> last byte 245759 of 245760; 2016 -> 258047 of 258048).
//
// GENIEX_KV_KEY_TILE / GENIEX_KV_VALUE_TILE override it, for probing an export
// whose tile choice differs.
GENIEX_API size_t tileFor(bool is_key);

// Physical description of one KV tensor. `is_key` selects which logical axis
// carries tokens, which differs between keys and values:
//
//   key:   din = head_dim, dout = kv_len    (tokens on dout, tile = K_TILE)
//   value: din = kv_len,   dout = head_dim  (tokens on din,  tile = V_TILE)
struct KVGeometry {
    size_t   n_heads   = 0;
    size_t   head_dim  = 0;
    size_t   kv_len    = 0;  // token capacity
    size_t   elem_size = 1;
    bool     is_key    = true;
    KVFormat format    = KVFormat::Flat;

    size_t din() const { return is_key ? head_dim : kv_len; }
    size_t dout() const { return is_key ? kv_len : head_dim; }

    // The compiler's nominal tile width along dout. Overridable for probing an
    // export whose choice differs (GENIEX_KV_KEY_TILE / GENIEX_KV_VALUE_TILE).
    size_t nominalTile() const { return tileFor(is_key); }
    // Width of the tile containing element `dout` -- short only for a trailing
    // partial tile.
    size_t tileWidthAt(size_t dout_el) const {
        const size_t n = nominalTile();
        const size_t h = (dout_el / n) * n;
        return std::min(n, dout() - h);
    }
    // Bytes per head. Identical in both layouts -- tiling permutes, never pads.
    size_t headStride() const { return head_dim * kv_len * elem_size; }
    size_t totalBytes() const { return n_heads * headStride(); }

    // Same tensor at a different token capacity (used by restride).
    KVGeometry withKvLen(size_t new_kv_len) const {
        KVGeometry g = *this;
        g.kv_len     = new_kv_len;
        return g;
    }
};

// Derives geometry from a graph tensor. `is_key` cannot be inferred from the
// shape alone (head_dim == kv_len is legal), so the caller passes it -- it
// already knows which half of the KVTensorPair it holds.
//
// Shapes: key [H, 1, head_dim, kv_len], value [H, 1, kv_len, head_dim].
GENIEX_API KVGeometry geometryOf(const TensorSpec& spec, bool is_key);

// Throws std::runtime_error naming `tensor_name` when the geometry cannot be
// represented in its declared format. HmxTiled requires:
//   - 1-byte elements                 (native-kv.cpp:23-25, "Native KV only supports uint8")
//   - din % 32 == 0 && dout % 32 == 0 (the two asserts in fromFlatOffset)
// and the derived layout must span exactly headStride() bytes.
GENIEX_API void validateGeometry(const KVGeometry& geo, const std::string& tensor_name);

// Byte offset, within one head, where the tile containing element `dout_el`
// starts. Every tile before the last is `nominalTile()` wide, so this is a plain
// product.
GENIEX_API size_t tileBase(const KVGeometry& geo, size_t dout_el);

// Byte offset of logical element (din, dout) within one head, in either layout.
GENIEX_API size_t elementOffset(const KVGeometry& geo, size_t din, size_t dout);

// How an empty cache slot is encoded. Tiled buffers clear to 0x00: HMX applies
// no zero-point offset to a native KV operand, so encoded zero is a literal 0
// rather than the dtype midpoint (native-kv.cpp:18-25, overriding SmartMask's
// 1 << 7). `supported == false` means "leave the buffer alone".
struct ZeroPattern {
    bool     supported = false;
    bool     wide      = false;  // true => 16-bit fill_n, false => memset
    uint8_t  byte_val  = 0;
    uint16_t u16_val   = 0;
};

GENIEX_API ZeroPattern zeroPatternFor(KVFormat format, Qnn_DataType_t dtype);
GENIEX_API void        fillZero(void* dst, size_t n_bytes, const ZeroPattern& z);

// Copies `n_tok` tokens from src[src_off ..] to dst[dst_off ..], translating
// between layouts as needed. `rebase` is added (mod 256) to every byte -- see
// deriveRebase. Geometries must agree on n_heads, head_dim, elem_size and
// is_key; kv_len and format may differ (that is the point).
//
// flat -> flat with rebase == 0 takes the original strided-memcpy path verbatim,
// so existing bundles are byte-for-byte unaffected.
GENIEX_API void copyTokens(const KVGeometry& dst, uint8_t* dst_buf, const KVGeometry& src, const uint8_t* src_buf,
    size_t src_off, size_t dst_off, size_t n_tok, int rebase = 0);

// Sets tokens [first_tok, first_tok + n_tok) to the encoded-zero pattern.
GENIEX_API void clearTokens(const KVGeometry& geo, uint8_t* buf, size_t first_tok, size_t n_tok, const ZeroPattern& z);

// Drops the oldest `shift` tokens, sliding the rest down to index 0 and clearing
// the vacated tail. Used by fixed-window (swa_*) caches once the window fills.
GENIEX_API void shiftLeft(const KVGeometry& geo, uint8_t* buf, size_t shift, const ZeroPattern& z);

// Re-strides a cache buffer in place from `old_kv_len` to `new_kv_len`,
// preserving the first `n_valid` tokens and clearing the remainder.
// `geo.kv_len` is ignored.
//
// Tiled restride is pure memmove when the tile extent is the same at both
// lengths: keys are then tile-major with a kv_len-independent tile stride, so
// growing merely appends tiles, and values need one memmove per (head, tile)
// region. When the derived tile CHANGES (e.g. keys going 1920 -> 2016, which
// tile at 128 and 32) the permutation itself changes and it re-tiles through a
// scratch buffer instead.
GENIEX_API void restride(
    const KVGeometry& geo, uint8_t* buf, size_t old_kv_len, size_t new_kv_len, size_t n_valid, const ZeroPattern& z);

// Writes the first `n_valid` tokens out in flat row-major form, for inspection
// and cross-run comparison (examples/kv_layout_check). `dst_flat` needs
// n_heads * head_dim * n_valid * elem_size bytes.
GENIEX_API void detile(const KVGeometry& geo, const uint8_t* src, uint8_t* dst_flat, size_t n_valid, int rebase = 0);

// Byte bias applied when moving a graph KV output into the cache buffer.
//
// A tiled cache holds zero-centred (signed) values while a flat uint8 KV output
// carries the +128 zero-point, so Genie subtracts 128 on that conversion
// (native-kv.cpp:322) and adds it back when dumping. When the output is itself
// tiled no rebase is needed (native-kv.cpp:345), and a flat cache never needs
// one. Derived from tensor metadata where possible; GENIEX_NATIVE_KV_REBASE=0|1
// forces it off/on.
GENIEX_API int deriveRebase(const TensorSpec& kv_in, const TensorSpec& kv_out);

}  // namespace geniex::kv
