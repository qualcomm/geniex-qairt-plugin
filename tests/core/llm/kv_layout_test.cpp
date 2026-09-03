// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause
//
// KV cache layout: HMX tiled addressing, layout conversion, restride, shift.
//
// The tiled offset function is cross-checked against a SEPARATE, independent
// reimplementation of the bit-interleaved addressing formula, not
// kv_layout.cpp's own factoring, so a mistake in the decomposition can't hide.
// Verified against a real native-kv bundle (Llama-3.2-3B-Instruct-SSD, w4a16).

#include "llm/kv_layout.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdlib>
#include <numeric>
#include <vector>

using namespace geniex;
using geniex::kv::KVFormat;
using geniex::kv::KVGeometry;

namespace {

// ── Independent reference implementation of the tiled offset formula ───────
int32_t refFromFlatOffset(int32_t DIN, int32_t DOUT, int32_t N_TILE, int32_t din, int32_t dout) {
    const int32_t tile_size   = std::min(DOUT, N_TILE);
    const int32_t tile_stride = DIN * tile_size;

    const int32_t tile_idx = dout / tile_size;
    const int32_t dout_0   = (dout % tile_size) >> 5;
    const int32_t dout_1   = dout & 0x1f;

    const int32_t din_0 = din >> 5;
    const int32_t din_1 = (din & 0x1f) >> 2;
    const int32_t din_2 = din & 0x3;

    static const int32_t bitshift[3] = {10, 7, 2};

    const int32_t din_0_stride = tile_size << 5;

    return tile_idx * tile_stride + din_0 * din_0_stride +
           (dout_0 << bitshift[0] | (din_1 << bitshift[1]) | (dout_1 << bitshift[2]) | din_2);
}

// Shapes from the real native-kv bundle (Llama-3.2-3B-Instruct-SSD w4a16): H=8,
// head_dim=128, CL=4096 (scatter, so kv_len == CL at every phase).
constexpr size_t kHeads   = 8;
constexpr size_t kHeadDim = 128;

KVGeometry geo(size_t kv_len, bool is_key, KVFormat fmt, size_t heads = kHeads) {
    KVGeometry g;
    g.n_heads   = heads;
    g.head_dim  = kHeadDim;
    g.kv_len    = kv_len;
    g.elem_size = 1;
    g.is_key    = is_key;
    g.format    = fmt;
    return g;
}

// Distinct, order-sensitive byte per (head, token, dim) so any permutation error
// shows up. Deliberately avoids 0 and 0x80 (the clear values).
uint8_t pattern(size_t h, size_t tok, size_t dim) {
    return static_cast<uint8_t>(1 + ((h * 131 + tok * 17 + dim * 7) % 254));
}

// Flat byte index of (head, token, dim).
size_t flatIdx(const KVGeometry& g, size_t h, size_t tok, size_t dim) {
    const size_t din  = g.is_key ? dim : tok;
    const size_t dout = g.is_key ? tok : dim;
    return h * g.head_dim * g.kv_len + din * (g.is_key ? g.kv_len : g.head_dim) + dout;
}

std::vector<uint8_t> makeFlat(const KVGeometry& g) {
    std::vector<uint8_t> buf(g.totalBytes(), 0);
    for (size_t h = 0; h < g.n_heads; ++h)
        for (size_t t = 0; t < g.kv_len; ++t)
            for (size_t d = 0; d < g.head_dim; ++d) buf[flatIdx(g, h, t, d)] = pattern(h, t, d);
    return buf;
}

// Reads token `tok` of a tiled buffer back into flat order, one head.
std::vector<uint8_t> readToken(const KVGeometry& g, const std::vector<uint8_t>& buf, size_t h, size_t tok) {
    std::vector<uint8_t> out(g.head_dim);
    for (size_t d = 0; d < g.head_dim; ++d) {
        const size_t din  = g.is_key ? d : tok;
        const size_t dout = g.is_key ? tok : d;
        out[d]            = buf[h * g.headStride() + kv::elementOffset(g, din, dout)];
    }
    return out;
}

void setEnv(const char* name, const char* value) {
#ifdef _WIN32
    _putenv_s(name, value ? value : "");
#else
    if (value)
        setenv(name, value, 1);
    else
        unsetenv(name);
#endif
}

TensorSpec spec(std::vector<uint32_t> shape, Qnn_DataType_t dt, Qnn_TensorDataFormat_t fmt) {
    TensorSpec s;
    s.name        = "past_key_0_in";
    s.shape       = std::move(shape);
    s.dtype       = dt;
    s.data_format = fmt;
    return s;
}

}  // namespace

// ── Addressing ───────────────────────────────────────────────────────────────

// Ground truth: elementOffset must match the reference formula bit-for-bit,
// over every (din, dout) pair, at the real bundle's shapes.
TEST(KVLayoutOffset, KeyOffsetMatchesReference) {
    const auto g = geo(/*kv_len=*/4096, /*is_key=*/true, KVFormat::HmxTiled);
    for (size_t din = 0; din < g.din(); ++din) {
        for (size_t dout = 0; dout < g.dout(); ++dout) {
            EXPECT_EQ(kv::elementOffset(g, din, dout),
                static_cast<size_t>(refFromFlatOffset(static_cast<int32_t>(g.din()),
                    static_cast<int32_t>(g.dout()),
                    static_cast<int32_t>(kv::K_TILE),
                    static_cast<int32_t>(din),
                    static_cast<int32_t>(dout))))
                << "din=" << din << " dout=" << dout;
        }
    }
}

TEST(KVLayoutOffset, ValueOffsetMatchesReference) {
    const auto g = geo(/*kv_len=*/4096, /*is_key=*/false, KVFormat::HmxTiled);
    for (size_t din = 0; din < g.din(); ++din) {
        for (size_t dout = 0; dout < g.dout(); ++dout) {
            EXPECT_EQ(kv::elementOffset(g, din, dout),
                static_cast<size_t>(refFromFlatOffset(static_cast<int32_t>(g.din()),
                    static_cast<int32_t>(g.dout()),
                    static_cast<int32_t>(kv::V_TILE),
                    static_cast<int32_t>(din),
                    static_cast<int32_t>(dout))))
                << "din=" << din << " dout=" << dout;
        }
    }
}

// KV OUTPUT tensors are narrower (dout == AR, e.g. 32 or 128, always <= the
// tile) than KV inputs (dout == CL). The formula must still match the
// reference for these single-tile shapes.
TEST(KVLayoutOffset, OutputShapesMatchReference) {
    for (size_t ar : {size_t{32}, size_t{128}}) {
        const auto g = geo(/*kv_len=*/ar, /*is_key=*/true, KVFormat::HmxTiled);
        for (size_t din = 0; din < g.din(); ++din)
            for (size_t dout = 0; dout < g.dout(); ++dout)
                EXPECT_EQ(kv::elementOffset(g, din, dout),
                    static_cast<size_t>(refFromFlatOffset(static_cast<int32_t>(g.din()),
                        static_cast<int32_t>(g.dout()),
                        static_cast<int32_t>(kv::K_TILE),
                        static_cast<int32_t>(din),
                        static_cast<int32_t>(dout))))
                    << "ar=" << ar << " din=" << din << " dout=" << dout;
    }
}

// Every logical element must land on a distinct byte inside the head, and the
// permutation must not address past the tensor's own size. 1792 and 4096 are
// both multiples of K_TILE (256) and V_TILE (64); 256 is the smallest legal
// key shape (exactly one tile).
TEST(KVLayoutOffset, TiledAddressingIsABijectionWithinAHead) {
    for (bool is_key : {true, false}) {
        for (size_t kv_len : {size_t{256}, size_t{1792}, size_t{4096}}) {
            const auto        g = geo(kv_len, is_key, KVFormat::HmxTiled);
            const size_t      n = g.headStride();
            std::vector<char> seen(n, 0);
            for (size_t din = 0; din < g.din(); ++din) {
                for (size_t dout = 0; dout < g.dout(); ++dout) {
                    const size_t off = kv::elementOffset(g, din, dout);
                    ASSERT_LT(off, n) << "out of bounds: is_key=" << is_key << " kv_len=" << kv_len;
                    ASSERT_EQ(seen[off], 0) << "collision at " << off << " is_key=" << is_key;
                    seen[off] = 1;
                }
            }
            EXPECT_EQ(std::count(seen.begin(), seen.end(), 1), static_cast<ptrdiff_t>(n));
        }
    }
}

// ── Validation ───────────────────────────────────────────────────────────────

// A partial trailing tile is not representable (see blockBase's doc comment):
// with a fixed din_0 stride, a dout not divisible by the tile would walk off
// the end of the tensor. Every real native-kv bundle is a scatter cache
// (kv_len == a power-of-two CL), for which this never arises.
TEST(KVLayoutValidate, TiledLayoutSpansExactlyTheTensor) {
    for (bool is_key : {true, false}) {
        for (size_t kv_len : {size_t{256}, size_t{1792}, size_t{4096}}) {
            const auto g = geo(kv_len, is_key, KVFormat::HmxTiled);
            EXPECT_NO_THROW(kv::validateGeometry(g, "t")) << "kv_len=" << kv_len << " is_key=" << is_key;
            // The last element must land on the head's last byte.
            EXPECT_EQ(kv::elementOffset(g, g.din() - 1, g.dout() - 1) + 1, g.headStride())
                << "kv_len=" << kv_len << " is_key=" << is_key;
        }
    }
}

TEST(KVLayoutValidate, RejectsPartialTileNon32MultipleAndWideElements) {
    // 1920 is not a multiple of K_TILE (256): a partial trailing tile.
    EXPECT_THROW(kv::validateGeometry(geo(1920, /*is_key=*/true, KVFormat::HmxTiled), "k"), std::runtime_error);
    EXPECT_THROW(kv::validateGeometry(geo(2000, /*is_key=*/false, KVFormat::HmxTiled), "v"), std::runtime_error);

    auto wide      = geo(4096, /*is_key=*/true, KVFormat::HmxTiled);
    wide.elem_size = 2;
    EXPECT_THROW(kv::validateGeometry(wide, "k"), std::runtime_error);
}

TEST(KVLayoutValidate, GeometryFromTensorSpec) {
    const auto key = kv::geometryOf(
        spec({8, 1, 128, 4096}, QNN_DATATYPE_UFIXED_POINT_8, QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER), /*is_key=*/true);
    EXPECT_EQ(key.n_heads, 8u);
    EXPECT_EQ(key.head_dim, 128u);
    EXPECT_EQ(key.kv_len, 4096u);
    EXPECT_EQ(key.format, KVFormat::Flat);

    const auto val =
        kv::geometryOf(spec({8, 1, 4096, 128}, QNN_DATATYPE_UFIXED_POINT_8, QNN_TENSOR_DATA_FORMAT_HMX_WEIGHT_LAYOUT),
            /*is_key=*/false);
    EXPECT_EQ(val.kv_len, 4096u);
    EXPECT_EQ(val.head_dim, 128u);
    EXPECT_EQ(val.format, KVFormat::HmxTiled);
}

// ── Layout conversion ────────────────────────────────────────────────────────

class KVLayoutCopy : public testing::TestWithParam<bool> {};

TEST_P(KVLayoutCopy, FlatToTiledRoundTripsForEveryAlignment) {
    const bool is_key = GetParam();
    // Source: a graph KV output holding one prefill chunk.
    const size_t n_tok   = 128;
    const auto   src     = geo(n_tok, is_key, KVFormat::Flat);
    const auto   src_buf = makeFlat(src);

    const auto dst = geo(2048, is_key, KVFormat::HmxTiled);
    // 0 is block-aligned; 5 and 100 are not, exercising the split-run path.
    for (size_t dst_off : {size_t{0}, size_t{5}, size_t{100}, size_t{1920}}) {
        std::vector<uint8_t> dst_buf(dst.totalBytes(), 0);
        kv::copyTokens(dst, dst_buf.data(), src, src_buf.data(), 0, dst_off, n_tok);

        for (size_t h = 0; h < dst.n_heads; ++h) {
            for (size_t t = 0; t < n_tok; ++t) {
                const auto got = readToken(dst, dst_buf, h, dst_off + t);
                for (size_t d = 0; d < dst.head_dim; ++d) {
                    ASSERT_EQ(got[d], pattern(h, t, d))
                        << "is_key=" << is_key << " dst_off=" << dst_off << " h=" << h << " t=" << t << " d=" << d;
                }
            }
        }
        // detile() must reproduce the flat source exactly.
        std::vector<uint8_t> back(dst.n_heads * dst.head_dim * (dst_off + n_tok), 0);
        kv::detile(dst, dst_buf.data(), back.data(), dst_off + n_tok);
        const auto flat_view = geo(dst_off + n_tok, is_key, KVFormat::Flat);
        for (size_t h = 0; h < dst.n_heads; ++h)
            for (size_t t = 0; t < n_tok; ++t)
                for (size_t d = 0; d < dst.head_dim; ++d)
                    ASSERT_EQ(back[flatIdx(flat_view, h, dst_off + t, d)], pattern(h, t, d));
    }
}

TEST_P(KVLayoutCopy, TiledToTiledAlignedFastPathMatchesUnalignedPath) {
    const bool   is_key = GetParam();
    const size_t n_tok  = 128;

    // Build a tiled source by converting from flat.
    const auto           flat_src  = geo(n_tok, is_key, KVFormat::Flat);
    const auto           flat_buf  = makeFlat(flat_src);
    const auto           tiled_src = geo(n_tok, is_key, KVFormat::HmxTiled);
    std::vector<uint8_t> src_buf(tiled_src.totalBytes(), 0);
    kv::copyTokens(tiled_src, src_buf.data(), flat_src, flat_buf.data(), 0, 0, n_tok);

    const auto dst = geo(2048, is_key, KVFormat::HmxTiled);

    // dst_off 256 hits the whole-block memcpy path; 257 forces the element path.
    // Both must produce the same logical tokens.
    for (size_t dst_off : {size_t{256}, size_t{257}}) {
        std::vector<uint8_t> dst_buf(dst.totalBytes(), 0);
        kv::copyTokens(dst, dst_buf.data(), tiled_src, src_buf.data(), 0, dst_off, n_tok);
        for (size_t h = 0; h < dst.n_heads; ++h)
            for (size_t t = 0; t < n_tok; ++t) {
                const auto got = readToken(dst, dst_buf, h, dst_off + t);
                for (size_t d = 0; d < dst.head_dim; ++d)
                    ASSERT_EQ(got[d], pattern(h, t, d)) << "dst_off=" << dst_off << " t=" << t << " d=" << d;
            }
    }
}

// Reproduces the real SSD bundle's forecast-prefix write exactly: an AR-128
// tiled output (tile=128) partially consumed (only 48 of its 128 tokens are
// real) written into a kv_len=4096 tiled cache (tile=256) starting at a
// non-32-aligned offset (16, the forecast-prefix boundary).
TEST_P(KVLayoutCopy, TiledToTiledPartialWriteAtSmallUnalignedOffset) {
    const bool   is_key  = GetParam();
    const size_t ar      = 128;
    const size_t n_tok   = 48;
    const size_t dst_off = 16;

    const auto           flat_src  = geo(ar, is_key, KVFormat::Flat);
    const auto           flat_buf  = makeFlat(flat_src);
    const auto           tiled_src = geo(ar, is_key, KVFormat::HmxTiled);
    std::vector<uint8_t> src_buf(tiled_src.totalBytes(), 0);
    kv::copyTokens(tiled_src, src_buf.data(), flat_src, flat_buf.data(), 0, 0, ar);

    const auto           dst = geo(4096, is_key, KVFormat::HmxTiled);
    std::vector<uint8_t> dst_buf(dst.totalBytes(), 0);
    kv::copyTokens(dst, dst_buf.data(), tiled_src, src_buf.data(), 0, dst_off, n_tok);

    for (size_t h = 0; h < dst.n_heads; ++h)
        for (size_t t = 0; t < n_tok; ++t) {
            const auto got = readToken(dst, dst_buf, h, dst_off + t);
            for (size_t d = 0; d < dst.head_dim; ++d)
                ASSERT_EQ(got[d], pattern(h, t, d)) << "dst_off=" << dst_off << " t=" << t << " d=" << d;
        }
}

TEST_P(KVLayoutCopy, RebaseIsAppliedPerByte) {
    const bool   is_key = GetParam();
    const size_t n_tok  = 32;
    const auto   src    = geo(n_tok, is_key, KVFormat::Flat);
    const auto   sbuf   = makeFlat(src);
    const auto   dst    = geo(256, is_key, KVFormat::HmxTiled);

    std::vector<uint8_t> dbuf(dst.totalBytes(), 0);
    kv::copyTokens(dst, dbuf.data(), src, sbuf.data(), 0, 0, n_tok, /*rebase=*/-128);
    for (size_t h = 0; h < dst.n_heads; ++h)
        for (size_t t = 0; t < n_tok; ++t) {
            const auto got = readToken(dst, dbuf, h, t);
            for (size_t d = 0; d < dst.head_dim; ++d) ASSERT_EQ(got[d], static_cast<uint8_t>(pattern(h, t, d) - 128));
        }
}

INSTANTIATE_TEST_SUITE_P(KeysAndValues, KVLayoutCopy, testing::Values(true, false),
    [](const testing::TestParamInfo<bool>& i) { return i.param ? "Key" : "Value"; });

// The pre-existing flat path must stay byte-identical: it is what every shipped
// bundle uses.
TEST(KVLayoutCopy, FlatToFlatMatchesHandWrittenStridedCopy) {
    for (bool is_key : {true, false}) {
        const size_t n_tok = 37, dst_off = 11, src_off = 3;
        const auto   src  = geo(64, is_key, KVFormat::Flat);
        const auto   sbuf = makeFlat(src);
        const auto   dst  = geo(512, is_key, KVFormat::Flat);

        std::vector<uint8_t> got(dst.totalBytes(), 0), want(dst.totalBytes(), 0);
        kv::copyTokens(dst, got.data(), src, sbuf.data(), src_off, dst_off, n_tok);

        const size_t rows = is_key ? dst.n_heads * dst.head_dim : dst.n_heads;
        const size_t tsz  = is_key ? 1 : dst.head_dim;
        for (size_t r = 0; r < rows; ++r)
            std::memcpy(want.data() + (r * dst.kv_len + dst_off) * tsz,
                sbuf.data() + (r * src.kv_len + src_off) * tsz,
                n_tok * tsz);
        EXPECT_EQ(got, want) << "is_key=" << is_key;
    }
}

TEST(KVLayoutCopy, RejectsOverlongWrite) {
    const auto           src = geo(128, true, KVFormat::Flat);
    const auto           dst = geo(256, true, KVFormat::HmxTiled);
    std::vector<uint8_t> s(src.totalBytes()), d(dst.totalBytes());
    EXPECT_THROW(kv::copyTokens(dst, d.data(), src, s.data(), 0, 200, 128), std::runtime_error);
    EXPECT_THROW(kv::copyTokens(dst, d.data(), src, s.data(), 64, 0, 128), std::runtime_error);
}

// ── clear / shift / restride ─────────────────────────────────────────────────

TEST(KVLayoutClear, TiledClearsExactlyTheRequestedTokens) {
    for (bool is_key : {true, false}) {
        // 64 is block-aligned, 7..7+20 is not.
        for (auto range : {std::pair<size_t, size_t>{64, 64}, std::pair<size_t, size_t>{7, 20}}) {
            const auto [first, count] = range;
            const auto flat           = geo(256, is_key, KVFormat::Flat);
            const auto fbuf           = makeFlat(flat);
            const auto g              = geo(256, is_key, KVFormat::HmxTiled);

            std::vector<uint8_t> buf(g.totalBytes(), 0);
            kv::copyTokens(g, buf.data(), flat, fbuf.data(), 0, 0, 256);
            kv::clearTokens(g, buf.data(), first, count, kv::zeroPatternFor(KVFormat::HmxTiled, QNN_DATATYPE_UINT_8));

            for (size_t h = 0; h < g.n_heads; ++h)
                for (size_t t = 0; t < g.kv_len; ++t) {
                    const auto got   = readToken(g, buf, h, t);
                    const bool clear = t >= first && t < first + count;
                    for (size_t d = 0; d < g.head_dim; ++d)
                        ASSERT_EQ(got[d], clear ? 0 : pattern(h, t, d))
                            << "is_key=" << is_key << " first=" << first << " t=" << t;
                }
        }
    }
}

TEST(KVLayoutShift, TiledShiftMatchesLogicalShift) {
    for (bool is_key : {true, false}) {
        for (size_t shift : {size_t{1}, size_t{32}, size_t{33}}) {
            const auto flat = geo(256, is_key, KVFormat::Flat);
            const auto fbuf = makeFlat(flat);
            const auto g    = geo(256, is_key, KVFormat::HmxTiled);

            std::vector<uint8_t> buf(g.totalBytes(), 0);
            kv::copyTokens(g, buf.data(), flat, fbuf.data(), 0, 0, 256);
            kv::shiftLeft(g, buf.data(), shift, kv::zeroPatternFor(KVFormat::HmxTiled, QNN_DATATYPE_UINT_8));

            for (size_t h = 0; h < g.n_heads; ++h)
                for (size_t t = 0; t < g.kv_len; ++t) {
                    const auto got = readToken(g, buf, h, t);
                    for (size_t d = 0; d < g.head_dim; ++d) {
                        const uint8_t want = (t + shift < g.kv_len) ? pattern(h, t + shift, d) : uint8_t{0};
                        ASSERT_EQ(got[d], want) << "is_key=" << is_key << " shift=" << shift << " t=" << t;
                    }
                }
        }
    }
}

TEST(KVLayoutRestride, TiledGrowThenShrinkPreservesValidTokens) {
    for (bool is_key : {true, false}) {
        const size_t old_len = 1792, new_len = 2048, n_valid = 1000;
        const auto   flat = geo(old_len, is_key, KVFormat::Flat);
        const auto   fbuf = makeFlat(flat);
        const auto   g    = geo(old_len, is_key, KVFormat::HmxTiled);
        const auto   zp   = kv::zeroPatternFor(KVFormat::HmxTiled, QNN_DATATYPE_UINT_8);

        // Buffer is allocated at the largest stride; both variants share it.
        const auto           grown = geo(new_len, is_key, KVFormat::HmxTiled);
        std::vector<uint8_t> buf(grown.totalBytes(), 0);
        kv::copyTokens(g, buf.data(), flat, fbuf.data(), 0, 0, old_len);

        kv::restride(g, buf.data(), old_len, new_len, n_valid, zp);
        for (size_t h = 0; h < grown.n_heads; ++h)
            for (size_t t = 0; t < n_valid; ++t) {
                const auto got = readToken(grown, buf, h, t);
                for (size_t d = 0; d < grown.head_dim; ++d)
                    ASSERT_EQ(got[d], pattern(h, t, d)) << "grow is_key=" << is_key << " h=" << h << " t=" << t;
            }

        kv::restride(grown, buf.data(), new_len, old_len, n_valid, zp);
        for (size_t h = 0; h < g.n_heads; ++h)
            for (size_t t = 0; t < n_valid; ++t) {
                const auto got = readToken(g, buf, h, t);
                for (size_t d = 0; d < g.head_dim; ++d)
                    ASSERT_EQ(got[d], pattern(h, t, d)) << "shrink is_key=" << is_key << " h=" << h << " t=" << t;
            }
    }
}

// The fixed tile (K_TILE=256 for keys) can still change size ACROSS a CL
// promotion: kv_len 128 tiles at 128 (dout <= tile), kv_len 512 tiles at 256.
// restride must re-tile through a scratch buffer rather than memmove when this
// happens. Round-trip both ways and check every valid token survives.
TEST(KVLayoutRestride, TiledRestrideAcrossATileSizeChange) {
    const size_t old_len = 128, new_len = 512, n_valid = 100;
    const auto   g_old = geo(old_len, /*is_key=*/true, KVFormat::HmxTiled);
    const auto   g_new = geo(new_len, /*is_key=*/true, KVFormat::HmxTiled);
    ASSERT_NE(g_old.tile(), g_new.tile()) << "this test only means something if the tile size changes";
    const auto zp = kv::zeroPatternFor(KVFormat::HmxTiled, QNN_DATATYPE_UINT_8);

    const auto           flat = geo(old_len, /*is_key=*/true, KVFormat::Flat);
    const auto           fbuf = makeFlat(flat);
    std::vector<uint8_t> buf(g_new.totalBytes(), 0);
    kv::copyTokens(g_old, buf.data(), flat, fbuf.data(), 0, 0, old_len);

    kv::restride(g_old, buf.data(), old_len, new_len, n_valid, zp);
    for (size_t h = 0; h < g_new.n_heads; ++h)
        for (size_t t = 0; t < n_valid; ++t) {
            const auto got = readToken(g_new, buf, h, t);
            for (size_t d = 0; d < g_new.head_dim; ++d)
                ASSERT_EQ(got[d], pattern(h, t, d)) << "grow h=" << h << " t=" << t << " d=" << d;
        }

    kv::restride(g_new, buf.data(), new_len, old_len, n_valid, zp);
    for (size_t h = 0; h < g_old.n_heads; ++h)
        for (size_t t = 0; t < n_valid; ++t) {
            const auto got = readToken(g_old, buf, h, t);
            for (size_t d = 0; d < g_old.head_dim; ++d)
                ASSERT_EQ(got[d], pattern(h, t, d)) << "shrink h=" << h << " t=" << t << " d=" << d;
        }
}

TEST(KVLayoutRestride, FlatGrowClearsTailWithDtypeMidpoint) {
    const size_t old_len = 128, new_len = 256, n_valid = 64;
    const auto   g   = geo(old_len, /*is_key=*/true, KVFormat::Flat);
    const auto   big = geo(new_len, /*is_key=*/true, KVFormat::Flat);
    const auto   zp  = kv::zeroPatternFor(KVFormat::Flat, QNN_DATATYPE_UFIXED_POINT_8);
    ASSERT_EQ(zp.byte_val, 0x80);

    std::vector<uint8_t> buf(big.totalBytes(), 0);
    const auto           fbuf = makeFlat(g);
    std::copy(fbuf.begin(), fbuf.end(), buf.begin());

    kv::restride(g, buf.data(), old_len, new_len, n_valid, zp);
    for (size_t h = 0; h < big.n_heads; ++h)
        for (size_t d = 0; d < big.head_dim; ++d) {
            for (size_t t = 0; t < n_valid; ++t)
                ASSERT_EQ(buf[flatIdx(big, h, t, d)], pattern(h, t, d)) << "h=" << h << " t=" << t;
            for (size_t t = n_valid; t < new_len; ++t)
                ASSERT_EQ(buf[flatIdx(big, h, t, d)], 0x80) << "tail h=" << h << " t=" << t;
        }
}

TEST(KVLayoutRestride, NoOpWhenLengthsMatch) {
    const auto           g = geo(256, true, KVFormat::HmxTiled);
    std::vector<uint8_t> buf(g.totalBytes(), 0x5a), copy = buf;
    kv::restride(g, buf.data(), 256, 256, 100, kv::zeroPatternFor(KVFormat::HmxTiled, QNN_DATATYPE_UINT_8));
    EXPECT_EQ(buf, copy);
}

// ── zero pattern / rebase ────────────────────────────────────────────────────

TEST(KVLayoutZero, TiledClearsToZeroRegardlessOfDtype) {
    // HMX applies no zero-point offset, so a tiled ufixed8 cache still clears to
    // 0x00, not 0x80.
    EXPECT_EQ(kv::zeroPatternFor(KVFormat::HmxTiled, QNN_DATATYPE_UFIXED_POINT_8).byte_val, 0x00);
    EXPECT_EQ(kv::zeroPatternFor(KVFormat::Flat, QNN_DATATYPE_UFIXED_POINT_8).byte_val, 0x80);
    EXPECT_EQ(kv::zeroPatternFor(KVFormat::Flat, QNN_DATATYPE_SFIXED_POINT_8).byte_val, 0x00);

    const auto u16 = kv::zeroPatternFor(KVFormat::Flat, QNN_DATATYPE_UFIXED_POINT_16);
    EXPECT_TRUE(u16.wide);
    EXPECT_EQ(u16.u16_val, 0x8000);
}

class KVLayoutRebase : public testing::Test {
   protected:
    void SetUp() override { setEnv("GENIEX_NATIVE_KV_REBASE", nullptr); }
    void TearDown() override { setEnv("GENIEX_NATIVE_KV_REBASE", nullptr); }
};

TEST_F(KVLayoutRebase, DerivedFromLayoutAndSignedness) {
    const auto flat_u8 = spec({8, 1, 128, 4096}, QNN_DATATYPE_UFIXED_POINT_8, QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER);
    const auto tiled_u8 =
        spec({8, 1, 128, 2048}, QNN_DATATYPE_UFIXED_POINT_8, QNN_TENSOR_DATA_FORMAT_HMX_WEIGHT_LAYOUT);
    const auto tiled_s8 =
        spec({8, 1, 128, 2048}, QNN_DATATYPE_SFIXED_POINT_8, QNN_TENSOR_DATA_FORMAT_HMX_WEIGHT_LAYOUT);
    const auto flat_out_u8 = spec({8, 1, 128, 128}, QNN_DATATYPE_UFIXED_POINT_8, QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER);
    const auto tiled_out_u8 =
        spec({8, 1, 128, 128}, QNN_DATATYPE_UFIXED_POINT_8, QNN_TENSOR_DATA_FORMAT_HMX_WEIGHT_LAYOUT);

    // Flat cache: never rebased.
    EXPECT_EQ(kv::deriveRebase(flat_u8, flat_out_u8), 0);
    // Tiled cache fed by a tiled output: already centred.
    EXPECT_EQ(kv::deriveRebase(tiled_u8, tiled_out_u8), 0);
    // Tiled cache fed by a flat unsigned output: rebase by -128.
    EXPECT_EQ(kv::deriveRebase(tiled_u8, flat_out_u8), -128);
    EXPECT_EQ(kv::deriveRebase(tiled_s8, flat_out_u8), -128);
    // Both sides declared signed: nothing to re-centre.
    const auto flat_out_s8 = spec({8, 1, 128, 128}, QNN_DATATYPE_SFIXED_POINT_8, QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER);
    EXPECT_EQ(kv::deriveRebase(tiled_s8, flat_out_s8), 0);
}

TEST_F(KVLayoutRebase, EnvOverrideWins) {
    const auto tiled_u8 =
        spec({8, 1, 128, 2048}, QNN_DATATYPE_UFIXED_POINT_8, QNN_TENSOR_DATA_FORMAT_HMX_WEIGHT_LAYOUT);
    const auto flat_out = spec({8, 1, 128, 128}, QNN_DATATYPE_UFIXED_POINT_8, QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER);
    const auto flat_in  = spec({8, 1, 128, 4096}, QNN_DATATYPE_UFIXED_POINT_8, QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER);

    setEnv("GENIEX_NATIVE_KV_REBASE", "0");
    EXPECT_EQ(kv::deriveRebase(tiled_u8, flat_out), 0);

    setEnv("GENIEX_NATIVE_KV_REBASE", "1");
    EXPECT_EQ(kv::deriveRebase(tiled_u8, flat_out), -128);
    EXPECT_EQ(kv::deriveRebase(flat_in, flat_out), -128);  // override applies unconditionally
}

// A CL == AR variant (bert-style, no cached past) has kv_len 0. restride must
// treat it as "nothing to preserve" rather than rejecting a zero extent --
// MultiCLEagleTargetFixture's cl8/ar8 graph hits exactly this.
TEST(KVLayoutRestride, ZeroLengthExtentIsHandled) {
    for (bool is_key : {true, false}) {
        for (auto fmt : {KVFormat::Flat, KVFormat::HmxTiled}) {
            const auto           big = geo(256, is_key, fmt);
            std::vector<uint8_t> buf(big.totalBytes(), 0x5a);
            const auto           zp = kv::zeroPatternFor(fmt, QNN_DATATYPE_UINT_8);

            // 0 -> 256: nothing to carry over, whole target cleared.
            EXPECT_NO_THROW(kv::restride(big, buf.data(), 0, 256, 100, zp));
            EXPECT_EQ(std::count(buf.begin(), buf.end(), zp.byte_val), static_cast<ptrdiff_t>(buf.size()));

            // 256 -> 0: no target extent to write.
            std::fill(buf.begin(), buf.end(), 0x5a);
            EXPECT_NO_THROW(kv::restride(big, buf.data(), 256, 0, 100, zp));

            // n_valid == 0 clears rather than preserving stale bytes.
            std::fill(buf.begin(), buf.end(), 0x5a);
            EXPECT_NO_THROW(kv::restride(big, buf.data(), 128, 256, 0, zp));
            EXPECT_EQ(std::count(buf.begin(), buf.end(), zp.byte_val), static_cast<ptrdiff_t>(buf.size()));
        }
    }
}
