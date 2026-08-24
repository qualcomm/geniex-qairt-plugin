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
// inside one 32x32 block. Flat: contiguous. Tiled: dout_1 sits at bit 2.
inline size_t doutStride(const KVGeometry& g) { return g.elem_size; }

}  // namespace

KVFormat formatOf(const TensorSpec& spec) {
    // GENIEX_NATIVE_KV=0 forces every KV tensor to be treated as flat, ignoring
    // the declared dataFormat. Kept as a diagnostic: an export can carry
    // HMX_WEIGHT_LAYOUT while its host-visible bytes are still row-major, and
    // this is how you tell the two apart on real hardware.
    static const bool force_flat = [] {
        const char* e = std::getenv("GENIEX_NATIVE_KV");
        return e != nullptr && e[0] == '0';
    }();
    if (force_flat) return KVFormat::Flat;
    return spec.data_format == QNN_TENSOR_DATA_FORMAT_HMX_WEIGHT_LAYOUT ? KVFormat::HmxTiled : KVFormat::Flat;
}

const char* formatName(KVFormat f) { return f == KVFormat::HmxTiled ? "HMX_WEIGHT_LAYOUT" : "FLAT_BUFFER"; }

size_t tileFor(bool is_key) {
    static const size_t key_override = [] {
        const char* e = std::getenv("GENIEX_KV_KEY_TILE");
        return e ? std::strtoul(e, nullptr, 10) : 0u;
    }();
    static const size_t val_override = [] {
        const char* e = std::getenv("GENIEX_KV_VALUE_TILE");
        return e ? std::strtoul(e, nullptr, 10) : 0u;
    }();

    return is_key ? (key_override ? key_override : K_TILE_MAX) : (val_override ? val_override : V_TILE_MAX);
}

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
    // A trailing partial tile IS representable: the tile is simply narrower, so
    // the layout still ends exactly at the tensor's last byte. Assert that
    // invariant rather than trusting the arithmetic.
    const size_t max_off = elementOffset(geo, geo.din() - 1, geo.dout() - 1) + 1;
    if (max_off != geo.headStride()) {
        throw std::runtime_error("kv_layout: tensor '" + tensor_name + "' tiled layout spans " +
                                 std::to_string(max_off) + " bytes per head but the tensor has " +
                                 std::to_string(geo.headStride()) + " (nominal tile " +
                                 std::to_string(geo.nominalTile()) + ", " + shapeStr(geo) + ")");
    }
}

size_t tileBase(const KVGeometry& geo, size_t dout_el) {
    // Every tile before the last is a full `nominal` wide, so their sizes are
    // uniform and the base is a plain product.
    return (dout_el / geo.nominalTile()) * geo.din() * geo.nominalTile();
}

size_t elementOffset(const KVGeometry& geo, size_t din, size_t dout) {
    if (geo.format == KVFormat::Flat) return (din * geo.dout() + dout) * geo.elem_size;
    const size_t nominal   = geo.nominalTile();
    const size_t tile_head = (dout / nominal) * nominal;
    const size_t width     = std::min(nominal, geo.dout() - tile_head);
    return tileBase(geo, dout) + din * width + (dout - tile_head);
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

    // General path. Any tiled side implies 1-byte elements (validateGeometry), so
    // element copies are byte copies.
    if (dst.elem_size != 1) {
        throw std::runtime_error(
            "kv_layout: layout conversion or rebase requires 1-byte elements, got " + std::to_string(dst.elem_size));
    }

    // Walk (din, dout) rather than (token, other): `dout` is the axis that is
    // contiguous when flat and stride-4 when tiled, in BOTH key and value
    // layouts, so keeping it innermost gives one tight strided loop per run.
    //
    //   key:   din spans head_dim,  dout carries the token range
    //   value: din carries the token range, dout spans head_dim
    const bool   key        = dst.is_key;
    const size_t din_count  = key ? dst.head_dim : n_tok;
    const size_t dout_count = key ? n_tok : dst.head_dim;
    const size_t src_din0   = key ? 0 : src_off;
    const size_t dst_din0   = key ? 0 : dst_off;
    const size_t src_dout0  = key ? src_off : 0;
    const size_t dst_dout0  = key ? dst_off : 0;

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

                // A run of consecutive dout is contiguous within one tile, so it
                // may not cross a tile boundary on either side.
                size_t run = dout_count - done;
                if (src.format == KVFormat::HmxTiled) {
                    const size_t n = src.nominalTile();
                    run            = std::min(run, n - s_dout % n);
                }
                if (dst.format == KVFormat::HmxTiled) {
                    const size_t n = dst.nominalTile();
                    run            = std::min(run, n - d_dout % n);
                }

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

    // Tiled: a run of consecutive dout inside one tile is contiguous, so clear it
    // with memset runs rather than element by element.
    for (size_t h = 0; h < geo.n_heads; ++h) {
        uint8_t* hp = buf + h * geo.headStride();
        for (size_t d = 0; d < geo.din(); ++d) {
            // Keys carry tokens on dout, values on din.
            if (!geo.is_key) continue;
            size_t done = 0;
            while (done < n_tok) {
                const size_t dout = first_tok + done;
                const size_t n    = geo.nominalTile();
                const size_t run  = std::min(n_tok - done, n - dout % n);
                fillZero(hp + elementOffset(geo, d, dout), run, z);
                done += run;
            }
        }
        if (!geo.is_key) {
            // value: tokens are din; each (token, tile) run spans the tile width.
            for (size_t t = first_tok; t < first_tok + n_tok; ++t) {
                size_t done = 0;
                while (done < geo.head_dim) {
                    const size_t n   = geo.nominalTile();
                    const size_t run = std::min(geo.head_dim - done, n - done % n);
                    fillZero(hp + elementOffset(geo, t, done), run, z);
                    done += run;
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
        // tile ordering (tileBase is monotonic in dout) and within a tile, so an
        // ascending in-place walk never reads a slot it has already written.
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
    // the only correct move is a full re-tile through a scratch buffer.
    //
    // This is the real Qwen3-4B natKV case for KEYS: kv_len 1920 tiles at 128 and
    // 2016 tiles at 32, so every prefill<->decode stride switch re-tiles. Values
    // tile along head_dim, which does not change, so they always take the fast
    // path below.
    const bool layout_shifts =
        geo.is_key || old_geo.nominalTile() != new_geo.nominalTile() ||
        old_geo.tileWidthAt(old_geo.dout() - TILE_GRAIN) != new_geo.tileWidthAt(new_geo.dout() - TILE_GRAIN);
    if (layout_shifts) {
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
        // Keys are tile-major with tile_stride = head_dim * K_TILE, independent
        // of kv_len, so a head's bytes are laid out identically at both lengths
        // and growing simply appends tiles. One memmove per head.
        const size_t tile       = old_geo.nominalTile();
        const size_t tile_bytes = geo.head_dim * tile;
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
    const size_t tile            = old_geo.nominalTile();
    const size_t n_tiles         = geo.head_dim / tile;
    const size_t old_tile_bytes  = old_kv_len * tile;
    const size_t new_tile_bytes  = new_kv_len * tile;
    const size_t din_block_bytes = tile * TILE_GRAIN;
    const size_t keep_blocks =
        std::min(roundUp(copy_len, TILE_GRAIN) / TILE_GRAIN, std::min(old_kv_len, new_kv_len) / TILE_GRAIN);
    const size_t copy_bytes = keep_blocks * din_block_bytes;
    const size_t n_regions  = geo.n_heads * n_tiles;

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
