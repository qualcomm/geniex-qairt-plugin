// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#include "llm/kv_layout.h"

#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace geniex::kv {

namespace {

// Rounds `n` up to the next multiple of `m`.
size_t roundUp(size_t n, size_t m) { return ((n + m - 1) / m) * m; }

std::string shapeStr(const KVGeometry& g) {
    return "n_heads=" + std::to_string(g.n_heads) + " head_dim=" + std::to_string(g.head_dim) +
           " kv_len=" + std::to_string(g.kv_len) + " elem_size=" + std::to_string(g.elem_size) +
           (g.is_key ? " (key)" : " (value)");
}

// Byte stride between consecutive `dout` values, holding `din` fixed and staying
// inside one 32x32 block. Flat: contiguous. Tiled: dout_1 sits at bit 2, so
// consecutive dout are 4 bytes apart (the gaps hold the other din_2 values).
inline size_t doutStride(const KVGeometry& g) { return g.format == KVFormat::Flat ? g.elem_size : 4; }

}  // namespace

KVFormat formatOf(const TensorSpec& spec) {
    return spec.data_format == QNN_TENSOR_DATA_FORMAT_HMX_WEIGHT_LAYOUT ? KVFormat::HmxTiled : KVFormat::Flat;
}

const char* formatName(KVFormat f) { return f == KVFormat::HmxTiled ? "HMX_WEIGHT_LAYOUT" : "FLAT_BUFFER"; }

KVGeometry geometryOf(const TensorSpec& spec, bool is_key) {
    if (spec.shape.size() < 4) {
        throw std::runtime_error("kv_layout: tensor '" + spec.name + "' has rank " + std::to_string(spec.shape.size()) +
                                 ", expected 4 ([H,1,head_dim,kv_len] for keys)");
    }
    KVGeometry g;
    g.n_heads   = spec.shape[0];
    g.head_dim  = is_key ? spec.shape[2] : spec.shape[3];
    g.kv_len    = is_key ? spec.shape[3] : spec.shape[2];
    g.elem_size = spec.elementSize();
    g.is_key    = is_key;
    g.format    = formatOf(spec);
    return g;
}

void validateGeometry(const KVGeometry& geo, const std::string& tensor_name) {
    if (geo.n_heads == 0 || geo.head_dim == 0 || geo.kv_len == 0 || geo.elem_size == 0) {
        throw std::runtime_error("kv_layout: tensor '" + tensor_name + "' has a zero extent: " + shapeStr(geo));
    }
    if (geo.format == KVFormat::Flat) return;

    // Genie asserts uint8 for native KV (native-kv.cpp:23-25): the tiled offset
    // arithmetic is expressed in bytes and HMX consumes 8-bit operands.
    if (geo.elem_size != 1) {
        throw std::runtime_error("kv_layout: tensor '" + tensor_name + "' is HMX_WEIGHT_LAYOUT but " +
                                 std::to_string(geo.elem_size) + " bytes per element; native KV is 8-bit only (" +
                                 shapeStr(geo) + ")");
    }
    if (geo.din() % TILE_GRAIN != 0 || geo.dout() % TILE_GRAIN != 0) {
        throw std::runtime_error("kv_layout: tensor '" + tensor_name +
                                 "' is HMX_WEIGHT_LAYOUT but din=" + std::to_string(geo.din()) +
                                 " dout=" + std::to_string(geo.dout()) + " are not both multiples of " +
                                 std::to_string(TILE_GRAIN) + " (" + shapeStr(geo) + ")");
    }
    // A partial trailing tile is NOT representable: Genie's din_0 stride is
    // (tile / 32) * KV_BLOCK_BYTES, reserving a full tile's worth of dout slots
    // per din block, so a half-used final tile would push the last element past
    // the tensor's own byte count. Every native-kv bundle we have seen is a
    // scatter cache (kv_len == a power-of-two CL), for which this always holds.
    if (geo.dout() % geo.tile() != 0) {
        throw std::runtime_error("kv_layout: tensor '" + tensor_name + "' is HMX_WEIGHT_LAYOUT with dout=" +
                                 std::to_string(geo.dout()) + ", not a multiple of the tile extent " +
                                 std::to_string(geo.tile()) + "; a partial trailing tile is not representable (" +
                                 shapeStr(geo) + ")");
    }
}

size_t blockBase(const KVGeometry& geo, size_t din_block, size_t dout_block) {
    const size_t tile            = geo.tile();
    const size_t blocks_per_tile = tile / TILE_GRAIN;               // dout blocks inside one tile
    const size_t tile_stride     = geo.din() * tile;                // bytes per tile
    const size_t din_block_stride = tile * TILE_GRAIN;              // bytes per din block inside a tile
    const size_t tile_idx        = dout_block / blocks_per_tile;    // Genie: dout / tile_size
    const size_t dout_0          = dout_block % blocks_per_tile;    // Genie: (dout % tile_size) >> 5
    return tile_idx * tile_stride + din_block * din_block_stride + dout_0 * KV_BLOCK_BYTES;
}

size_t elementOffset(const KVGeometry& geo, size_t din, size_t dout) {
    if (geo.format == KVFormat::Flat) return (din * geo.dout() + dout) * geo.elem_size;
    return blockBase(geo, din / TILE_GRAIN, dout / TILE_GRAIN) + lowOffset(din % TILE_GRAIN, dout % TILE_GRAIN);
}

ZeroPattern zeroPatternFor(KVFormat format, Qnn_DataType_t dtype) {
    // A tiled buffer is consumed by HMX, which applies no zero-point offset, so
    // its encoded zero is a literal 0 regardless of the declared dtype
    // (native-kv.cpp:18-25).
    if (format == KVFormat::HmxTiled) return {true, false, 0x00, 0};

    // Flat: the dtype midpoint, matching Genie's SmartMask (kvmanager.cpp:68-79).
    switch (dtype) {
        case QNN_DATATYPE_UFIXED_POINT_8:
        case QNN_DATATYPE_UINT_8:
            return {true, false, 0x80, 0};
        case QNN_DATATYPE_SFIXED_POINT_8:
        case QNN_DATATYPE_INT_8:
        case QNN_DATATYPE_BOOL_8:
            return {true, false, 0x00, 0};
        case QNN_DATATYPE_UFIXED_POINT_16:
        case QNN_DATATYPE_UINT_16:
            return {true, true, 0x00, 0x8000};
        case QNN_DATATYPE_SFIXED_POINT_16:
        case QNN_DATATYPE_INT_16:
        case QNN_DATATYPE_FLOAT_16:
            return {true, true, 0x00, 0x0000};
        case QNN_DATATYPE_FLOAT_32:
        case QNN_DATATYPE_INT_32:
        case QNN_DATATYPE_UINT_32:
        case QNN_DATATYPE_SFIXED_POINT_32:
        case QNN_DATATYPE_UFIXED_POINT_32:
            return {true, false, 0x00, 0};
        default:
            return {};
    }
}

void fillZero(void* dst, size_t n_bytes, const ZeroPattern& z) {
    if (!z.supported || n_bytes == 0) return;
    if (z.wide) {
        std::fill_n(static_cast<uint16_t*>(dst), n_bytes / 2, z.u16_val);
    } else {
        std::memset(dst, z.byte_val, n_bytes);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// copyTokens
// ─────────────────────────────────────────────────────────────────────────────

namespace {

void requireCompatible(const KVGeometry& dst, const KVGeometry& src) {
    if (dst.n_heads != src.n_heads || dst.head_dim != src.head_dim || dst.is_key != src.is_key ||
        dst.elem_size != src.elem_size) {
        throw std::runtime_error(
            "kv_layout: incompatible KV geometries, dst[" + shapeStr(dst) + "] src[" + shapeStr(src) + "]");
    }
}

// The original flat strided copy, kept verbatim so existing bundles are
// unaffected by this refactor.
//   key   [H, 1, head_dim, kv_len] -> one memcpy per (head, head_dim) row
//   value [H, 1, kv_len, head_dim] -> one memcpy per head, head_dim per token
void copyFlatToFlat(const KVGeometry& dst, uint8_t* dst_buf, const KVGeometry& src, const uint8_t* src_buf,
    size_t src_off, size_t dst_off, size_t n_tok) {
    size_t num_rows, token_size;
    if (dst.is_key) {
        num_rows   = dst.n_heads * dst.head_dim;
        token_size = dst.elem_size;
    } else {
        num_rows   = dst.n_heads;
        token_size = dst.head_dim * dst.elem_size;
    }
    for (size_t row = 0; row < num_rows; ++row) {
        std::memcpy(dst_buf + (row * dst.kv_len + dst_off) * token_size,
            src_buf + (row * src.kv_len + src_off) * token_size,
            n_tok * token_size);
    }
}

// Whole-block copy: both sides tiled, rebase-free, and every offset/count
// 32-aligned, so each 32x32 block moves as one contiguous KV_BLOCK_BYTES
// memcpy. This is what keeps aligned prefill write-back (chunk sizes are
// always AR, itself 32-aligned for a native-kv bundle) in the same cost class
// as the flat path, instead of the element-by-element general path below.
bool tryCopyWholeBlocks(const KVGeometry& dst, uint8_t* dst_buf, const KVGeometry& src, const uint8_t* src_buf,
    size_t src_off, size_t dst_off, size_t n_tok, int rebase) {
    if (dst.format != KVFormat::HmxTiled || src.format != KVFormat::HmxTiled) return false;
    if (rebase != 0) return false;
    if (src_off % TILE_GRAIN != 0 || dst_off % TILE_GRAIN != 0 || n_tok % TILE_GRAIN != 0) return false;

    // The non-token axis always spans its full extent (head_dim), which the
    // tiled format already guarantees is a multiple of TILE_GRAIN.
    const size_t other_blocks = dst.head_dim / TILE_GRAIN;
    const size_t tok_blocks   = n_tok / TILE_GRAIN;

    for (size_t h = 0; h < dst.n_heads; ++h) {
        const uint8_t* sh = src_buf + h * src.headStride();
        uint8_t*       dh = dst_buf + h * dst.headStride();
        for (size_t ob = 0; ob < other_blocks; ++ob) {
            for (size_t tb = 0; tb < tok_blocks; ++tb) {
                const size_t s_tok = src_off / TILE_GRAIN + tb;
                const size_t d_tok = dst_off / TILE_GRAIN + tb;
                const size_t s_off = dst.is_key ? blockBase(src, ob, s_tok) : blockBase(src, s_tok, ob);
                const size_t d_off = dst.is_key ? blockBase(dst, ob, d_tok) : blockBase(dst, d_tok, ob);
                std::memcpy(dh + d_off, sh + s_off, KV_BLOCK_BYTES);
            }
        }
    }
    return true;
}

}  // namespace

void copyTokens(const KVGeometry& dst, uint8_t* dst_buf, const KVGeometry& src, const uint8_t* src_buf, size_t src_off,
    size_t dst_off, size_t n_tok, int rebase) {
    if (n_tok == 0) return;
    requireCompatible(dst, src);
    if (src_off + n_tok > src.kv_len) {
        throw std::runtime_error("kv_layout: read [" + std::to_string(src_off) + ", " +
                                 std::to_string(src_off + n_tok) + ") exceeds src capacity " +
                                 std::to_string(src.kv_len));
    }
    if (dst_off + n_tok > dst.kv_len) {
        throw std::runtime_error("kv_layout: write [" + std::to_string(dst_off) + ", " +
                                 std::to_string(dst_off + n_tok) + ") exceeds dst capacity " +
                                 std::to_string(dst.kv_len));
    }

    if (dst.format == KVFormat::Flat && src.format == KVFormat::Flat && rebase == 0) {
        copyFlatToFlat(dst, dst_buf, src, src_buf, src_off, dst_off, n_tok);
        return;
    }

    if (tryCopyWholeBlocks(dst, dst_buf, src, src_buf, src_off, dst_off, n_tok, rebase)) return;

    // General path. Any tiled side implies 1-byte elements (validateGeometry), so
    // element copies are byte copies.
    if (dst.elem_size != 1) {
        throw std::runtime_error(
            "kv_layout: layout conversion or rebase requires 1-byte elements, got " + std::to_string(dst.elem_size));
    }

    // Walk (din, dout) with dout innermost: it is the axis that is contiguous
    // when flat and (within one 32-element block) stride-4 when tiled, in BOTH
    // key and value layouts.
    //
    //   key:   din spans head_dim,  dout carries the token range
    //   value: din carries the token range, dout spans head_dim
    const bool    key        = dst.is_key;
    const size_t  din_count  = key ? dst.head_dim : n_tok;
    const size_t  dout_count = key ? n_tok : dst.head_dim;
    const size_t  src_din0   = key ? 0 : src_off;
    const size_t  dst_din0   = key ? 0 : dst_off;
    const size_t  src_dout0  = key ? src_off : 0;
    const size_t  dst_dout0  = key ? dst_off : 0;

    const size_t s_stride = doutStride(src);
    const size_t d_stride = doutStride(dst);
    const auto   bias     = static_cast<uint8_t>(static_cast<unsigned>(rebase) & 0xffu);

    for (size_t h = 0; h < dst.n_heads; ++h) {
        const uint8_t* sh = src_buf + h * src.headStride();
        uint8_t*       dh = dst_buf + h * dst.headStride();

        for (size_t d = 0; d < din_count; ++d) {
            const size_t s_din = src_din0 + d;
            const size_t d_din = dst_din0 + d;

            size_t done = 0;
            while (done < dout_count) {
                const size_t s_dout = src_dout0 + done;
                const size_t d_dout = dst_dout0 + done;

                // A run may not cross a 32-element block boundary on either
                // side, since the block (and, for tiled, the byte stride) both
                // change there.
                size_t run = dout_count - done;
                if (src.format == KVFormat::HmxTiled) run = std::min(run, TILE_GRAIN - s_dout % TILE_GRAIN);
                if (dst.format == KVFormat::HmxTiled) run = std::min(run, TILE_GRAIN - d_dout % TILE_GRAIN);

                const uint8_t* sp = sh + elementOffset(src, s_din, s_dout);
                uint8_t*       dp = dh + elementOffset(dst, d_din, d_dout);

                if (rebase == 0 && s_stride == 1 && d_stride == 1) {
                    std::memcpy(dp, sp, run);
                } else {
                    for (size_t k = 0; k < run; ++k) {
                        dp[k * d_stride] = static_cast<uint8_t>(sp[k * s_stride] + bias);
                    }
                }
                done += run;
            }
        }
    }
}

void clearTokens(const KVGeometry& geo, uint8_t* buf, size_t first_tok, size_t n_tok, const ZeroPattern& z) {
    if (n_tok == 0 || !z.supported) return;
    if (first_tok + n_tok > geo.kv_len) {
        throw std::runtime_error("kv_layout: clear [" + std::to_string(first_tok) + ", " +
                                 std::to_string(first_tok + n_tok) + ") exceeds capacity " +
                                 std::to_string(geo.kv_len));
    }

    if (geo.format == KVFormat::Flat) {
        size_t num_rows, token_size;
        if (geo.is_key) {
            num_rows   = geo.n_heads * geo.head_dim;
            token_size = geo.elem_size;
        } else {
            num_rows   = geo.n_heads;
            token_size = geo.head_dim * geo.elem_size;
        }
        for (size_t row = 0; row < num_rows; ++row) {
            fillZero(buf + (row * geo.kv_len + first_tok) * token_size, n_tok * token_size, z);
        }
        return;
    }

    // Tiled. The token axis is dout for keys, din for values; the other axis is
    // always cleared in full. Whole 32x32 blocks are one memset each; a ragged
    // edge (token range not 32-aligned) falls back to one byte at a time --
    // there is no contiguous run to exploit there (see doutStride).
    const size_t other       = geo.is_key ? geo.din() : geo.dout();
    const bool   tok_aligned = first_tok % TILE_GRAIN == 0 && n_tok % TILE_GRAIN == 0;

    for (size_t h = 0; h < geo.n_heads; ++h) {
        uint8_t* hp = buf + h * geo.headStride();
        if (tok_aligned) {
            for (size_t ob = 0; ob < other / TILE_GRAIN; ++ob) {
                for (size_t tb = first_tok / TILE_GRAIN; tb < (first_tok + n_tok) / TILE_GRAIN; ++tb) {
                    const size_t off = geo.is_key ? blockBase(geo, ob, tb) : blockBase(geo, tb, ob);
                    std::memset(hp + off, z.byte_val, KV_BLOCK_BYTES);
                }
            }
        } else {
            for (size_t o = 0; o < other; ++o) {
                for (size_t t = first_tok; t < first_tok + n_tok; ++t) {
                    const size_t off = geo.is_key ? elementOffset(geo, o, t) : elementOffset(geo, t, o);
                    hp[off]          = z.byte_val;
                }
            }
        }
    }
}

void shiftLeft(const KVGeometry& geo, uint8_t* buf, size_t shift, const ZeroPattern& z) {
    if (shift == 0) return;
    if (shift >= geo.kv_len) {
        clearTokens(geo, buf, 0, geo.kv_len, z);
        return;
    }
    const size_t keep = geo.kv_len - shift;

    if (geo.format == KVFormat::Flat) {
        // memmove, not memcpy: source and destination overlap.
        size_t num_rows, token_size;
        if (geo.is_key) {
            num_rows   = geo.n_heads * geo.head_dim;
            token_size = geo.elem_size;
        } else {
            num_rows   = geo.n_heads;
            token_size = geo.head_dim * geo.elem_size;
        }
        for (size_t row = 0; row < num_rows; ++row) {
            uint8_t* base = buf + row * geo.kv_len * token_size;
            std::memmove(base, base + shift * token_size, keep * token_size);
        }
    } else {
        // Tiled: token t moves to t - shift, i.e. strictly downward in both the
        // block ordering (blockBase is monotonic in dout_block for fixed
        // din_block) and within a block, so an ascending in-place walk never
        // reads a slot it has already written.
        copyTokens(geo, buf, geo, buf, shift, 0, keep, 0);
    }
    clearTokens(geo, buf, keep, shift, z);
}

void restride(
    const KVGeometry& geo, uint8_t* buf, size_t old_kv_len, size_t new_kv_len, size_t n_valid, const ZeroPattern& z) {
    if (old_kv_len == new_kv_len) return;

    const KVGeometry old_geo = geo.withKvLen(old_kv_len);
    const KVGeometry new_geo = geo.withKvLen(new_kv_len);

    // Never read past old_kv_len nor write past new_kv_len.
    const size_t copy_len = std::min(n_valid, std::min(old_kv_len, new_kv_len));

    // Nothing to preserve: just clear the target extent. This is also the only
    // sane path when either length is 0 -- a CL == AR variant (bert-style, no
    // cached past) has no capacity at all, and the tile math below is undefined
    // for a zero extent.
    if (copy_len == 0) {
        fillZero(buf, new_geo.totalBytes(), z);
        return;
    }

    validateGeometry(old_geo, "<kv restride source>");
    validateGeometry(new_geo, "<kv restride target>");

    const size_t old_head = old_geo.headStride();
    const size_t new_head = new_geo.headStride();
    const bool   growing  = new_kv_len > old_kv_len;

    if (geo.format == KVFormat::Flat) {
        // key   [H, 1, head_dim, kv_len]: one row per (head, head_dim)
        // value [H, 1, kv_len, head_dim]: one row per head, head_dim per token
        const size_t token_size = geo.is_key ? geo.elem_size : geo.head_dim * geo.elem_size;
        const size_t n_rows     = geo.is_key ? geo.n_heads * geo.head_dim : geo.n_heads;
        const size_t copy_bytes = copy_len * token_size;
        const size_t old_row    = old_kv_len * token_size;
        const size_t new_row    = new_kv_len * token_size;

        if (growing) {
            // Backward, so a row's destination never overwrites an unread source.
            for (size_t i = n_rows; i-- > 0;) {
                std::memmove(buf + i * new_row, buf + i * old_row, copy_bytes);
                fillZero(buf + i * new_row + copy_bytes, new_row - copy_bytes, z);
            }
        } else {
            for (size_t i = 0; i < n_rows; ++i) {
                std::memmove(buf + i * new_row, buf + i * old_row, copy_bytes);
                fillZero(buf + i * new_row + copy_bytes, new_row - copy_bytes, z);
            }
        }
        return;
    }

    // Tiled. The memmove paths below rely on the tile extent being the same at
    // both lengths; when it is not, the whole intra-head permutation changes and
    // the only correct move is a full re-tile through a scratch buffer. This
    // only happens on a CL promotion that crosses the tile's own size threshold
    // (e.g. keys: CL 128 -> 512, tile 128 -> 256) -- a scatter cache's kv_len is
    // otherwise phase-independent, so restride is rarely even called.
    if (old_geo.tile() != new_geo.tile()) {
        // Per-thread so the concurrent decode-pool KV workers do not share it.
        thread_local std::vector<uint8_t> scratch;
        scratch.assign(geo.n_heads * geo.head_dim * copy_len, 0);

        KVGeometry flat = old_geo.withKvLen(copy_len);
        flat.format     = KVFormat::Flat;
        copyTokens(flat, scratch.data(), old_geo, buf, /*src_off=*/0, /*dst_off=*/0, copy_len);

        fillZero(buf, new_geo.totalBytes(), z);
        copyTokens(new_geo, buf, flat, scratch.data(), /*src_off=*/0, /*dst_off=*/0, copy_len);
        return;
    }

    if (geo.is_key) {
        // Keys are tile-major with tile_stride = head_dim * tile, independent of
        // kv_len, so a head's bytes are laid out identically at both lengths and
        // growing simply appends tiles. One memmove per head.
        const size_t tile       = old_geo.tile();
        const size_t tile_bytes = geo.head_dim * tile;
        // copy_len (n_valid) is an arbitrary runtime value, not necessarily
        // tile-aligned -- round UP so a partial final tile is still carried
        // over whole, capped so we never read/write past either buffer's
        // actual extent.
        const size_t keep_tiles = std::min(roundUp(copy_len, tile) / tile, std::min(old_kv_len, new_kv_len) / tile);
        const size_t copy_bytes = keep_tiles * tile_bytes;

        if (growing) {
            for (size_t h = geo.n_heads; h-- > 0;) {
                std::memmove(buf + h * new_head, buf + h * old_head, copy_bytes);
                fillZero(buf + h * new_head + copy_bytes, new_head - copy_bytes, z);
            }
        } else {
            for (size_t h = 0; h < geo.n_heads; ++h) {
                std::memmove(buf + h * new_head, buf + h * old_head, copy_bytes);
                fillZero(buf + h * new_head + copy_bytes, new_head - copy_bytes, z);
            }
        }
        return;
    }

    // Values are tiled along head_dim with tile_stride = kv_len * V_TILE, which
    // DOES scale with kv_len, so each (head, tile) region moves separately.
    // Inside a region the din (token) blocks are contiguous and in order, so the
    // live prefix is a contiguous byte range.
    const size_t tile            = old_geo.tile();
    const size_t n_tiles         = geo.head_dim / tile;
    const size_t old_tile_bytes  = old_kv_len * tile;
    const size_t new_tile_bytes  = new_kv_len * tile;
    const size_t din_block_bytes = tile * TILE_GRAIN;
    // Same rounding-up rationale as the key path above.
    const size_t keep_blocks =
        std::min(roundUp(copy_len, TILE_GRAIN) / TILE_GRAIN, std::min(old_kv_len, new_kv_len) / TILE_GRAIN);
    const size_t copy_bytes = keep_blocks * din_block_bytes;
    const size_t n_regions       = geo.n_heads * n_tiles;

    auto move_region = [&](size_t r) {
        const size_t h        = r / n_tiles;
        const size_t t        = r % n_tiles;
        uint8_t*     dst      = buf + h * new_head + t * new_tile_bytes;
        const size_t old_base = h * old_head + t * old_tile_bytes;
        std::memmove(dst, buf + old_base, copy_bytes);
        fillZero(dst + copy_bytes, new_tile_bytes - copy_bytes, z);
    };

    if (growing) {
        for (size_t r = n_regions; r-- > 0;) move_region(r);
    } else {
        for (size_t r = 0; r < n_regions; ++r) move_region(r);
    }
}

void detile(const KVGeometry& geo, const uint8_t* src, uint8_t* dst_flat, size_t n_valid, int rebase) {
    if (n_valid == 0) return;
    KVGeometry flat = geo.withKvLen(n_valid);
    flat.format     = KVFormat::Flat;
    copyTokens(flat, dst_flat, geo, src, /*src_off=*/0, /*dst_off=*/0, n_valid, rebase);
}

int deriveRebase(const TensorSpec& kv_in, const TensorSpec& kv_out) {
    if (const char* env = std::getenv("GENIEX_NATIVE_KV_REBASE")) {
        return (env[0] == '0') ? 0 : -128;
    }

    const KVFormat in_fmt  = formatOf(kv_in);
    const KVFormat out_fmt = formatOf(kv_out);

    // A flat cache keeps whatever encoding the graph emitted.
    if (in_fmt == KVFormat::Flat) return 0;
    // Tiled output already carries the cache's own encoding (native-kv.cpp:345).
    if (out_fmt == KVFormat::HmxTiled) return 0;

    auto isSigned   = [](Qnn_DataType_t dt) { return dt == QNN_DATATYPE_SFIXED_POINT_8 || dt == QNN_DATATYPE_INT_8; };
    auto isUnsigned = [](Qnn_DataType_t dt) { return dt == QNN_DATATYPE_UFIXED_POINT_8 || dt == QNN_DATATYPE_UINT_8; };

    // Declared-signedness disagreement makes the rebase explicit.
    if (isSigned(kv_in.dtype) && isUnsigned(kv_out.dtype)) return -128;
    if (isSigned(kv_in.dtype) && isSigned(kv_out.dtype)) return 0;

    // Both declared unsigned: HMX still reads the tiled operand as zero-centred
    // regardless of the declared dtype, so Genie's unconditional -128 applies
    // (native-kv.cpp:322). Override with GENIEX_NATIVE_KV_REBASE=0 if a recipe
    // ever emits an already-centred flat output.
    return -128;
}

}  // namespace geniex::kv
