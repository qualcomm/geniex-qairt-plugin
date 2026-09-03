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
//                       by ENABLE_NATIVE_KV recipes.
//
// Verified against a real native-kv bundle (Llama-3.2-3B-Instruct-SSD, w4a16).
// That bundle's caches are always full context length (a scatter cache:
// kv_len == CL), so K_TILE / V_TILE always divide the tiled axis; this file
// assumes the same and rejects a shape where they don't.
namespace geniex::kv {

// Tile widths the QNN compiler uses for the HMX weight layout. Each tensor is
// tiled into chunks of min(dout, tile).
constexpr size_t K_TILE = 256;  // key tiling along the kv_len (dout) axis
constexpr size_t V_TILE = 64;   // value tiling along the head_dim (dout) axis

// The tiled layout's innermost chunk is [din_1:8][dout_1:32][din_2:4] = 1024
// contiguous bytes spanning 32 din x 32 dout logical elements -- the unit of
// contiguity for a whole-block memcpy fast path.
constexpr size_t KV_BLOCK_BYTES = 1024;
constexpr size_t TILE_GRAIN     = 32;  // logical elements per axis per block

enum class KVFormat {
    Flat,     // QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER (row-major)
    HmxTiled  // QNN_TENSOR_DATA_FORMAT_HMX_WEIGHT_LAYOUT
};

GENIEX_API KVFormat    formatOf(const TensorSpec& spec);
GENIEX_API const char* formatName(KVFormat f);

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
    size_t tile() const { return std::min(dout(), is_key ? K_TILE : V_TILE); }
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
//   - 1-byte elements
//   - din % 32 == 0 && dout % 32 == 0
//   - dout % tile == 0 (a partial trailing tile is not addressable by the
//     fixed block stride below)
GENIEX_API void validateGeometry(const KVGeometry& geo, const std::string& tensor_name);

// Byte offset, within one head, of the 32x32 block containing logical element
// (din, dout). Blocks are KV_BLOCK_BYTES long and fully contiguous.
// `din_block` / `dout_block` are element indices divided by TILE_GRAIN.
GENIEX_API size_t blockBase(const KVGeometry& geo, size_t din_block, size_t dout_block);

// Offset of (din % 32, dout % 32) inside a 32x32 block. This bit-interleaving
// is fixed -- independent of shape and tile size:
// din_1 << 7 | dout_1 << 2 | din_2.
constexpr size_t lowOffset(size_t din_lo, size_t dout_lo) {
    return ((din_lo >> 2) << 7) | (dout_lo << 2) | (din_lo & 3);
}

// Byte offset of logical element (din, dout) within one head, in either layout.
GENIEX_API size_t elementOffset(const KVGeometry& geo, size_t din, size_t dout);

// How an empty cache slot is encoded. Tiled buffers clear to 0x00: HMX applies
// no zero-point offset to a native KV operand, so encoded zero is a literal 0
// rather than the dtype midpoint used for flat buffers. `supported == false`
// means "leave the buffer alone".
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
// Pure memmove when the tile extent is unchanged at both lengths (keys are
// tile-major with a kv_len-independent stride; values need one memmove per
// (head, tile)). Only re-tiles through a scratch buffer when a CL promotion
// crosses the tile's own size (e.g. CL 128 -> 512 for keys, tile 128 -> 256).
GENIEX_API void restride(
    const KVGeometry& geo, uint8_t* buf, size_t old_kv_len, size_t new_kv_len, size_t n_valid, const ZeroPattern& z);

// Writes the first `n_valid` tokens out in flat row-major form, for inspection
// and cross-run comparison (examples/kv_layout_check). `dst_flat` needs
// n_heads * head_dim * n_valid * elem_size bytes.
GENIEX_API void detile(const KVGeometry& geo, const uint8_t* src, uint8_t* dst_flat, size_t n_valid, int rebase = 0);

// Byte bias applied when moving a graph KV output into the cache buffer.
//
// A tiled cache holds zero-centred (signed) values while a flat uint8 KV
// output carries the +128 zero-point, so that conversion subtracts 128. When
// the output is itself tiled no rebase is needed, and a flat cache never
// needs one. Derived from tensor metadata where possible;
// GENIEX_NATIVE_KV_REBASE=0|1 forces it off/on.
GENIEX_API int deriveRebase(const TensorSpec& kv_in, const TensorSpec& kv_out);

}  // namespace geniex::kv
