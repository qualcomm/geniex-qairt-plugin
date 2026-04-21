#include "qwen2_omni.h"
#include "utils.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace geniex {
namespace qwen2_omni {

// ── Qwen2OmniVisionEncoder ────────────────────────────────────────────────────

std::vector<float> Qwen2OmniVisionEncoder::encode(const PixelData& pixel_data) {
    const auto inv_freq = qwen_vit::makeInvFreq();
    const size_t patch_elems = kPatchChannels * kPatchSize * kPatchSize;  // 1176
    const size_t sm_unit     = kSpatialMergeSize * kSpatialMergeSize;     // 4

    Graph& g_patch = graph(0);
    Graph& g_vit   = graph(1);

    // Infer per-call capacity from graph buffer sizes.
    // patch_embed input: [batch, 6, 14, 14]  →  batch = buf / (1176 * sizeof(float))
    // vit input:         [1, batch, 1280]    →  batch = buf / (1280 * sizeof(float))
    const size_t patch_batch = g_patch.inputSpecs()[0].byteCount() / (patch_elems * sizeof(float));
    const size_t vit_batch   = g_vit.inputSpecs()[0].byteCount()   / (kEmbedDim   * sizeof(float));

    std::vector<float> all_embeddings;
    size_t px_offset = 0;

    TimeLog tl;

    for (const auto& thw : pixel_data.image_grid_thw) {
        const int T = thw[0], H = thw[1], W = thw[2];
        const size_t seq_len  = static_cast<size_t>(T * H * W);
        const size_t n_groups = seq_len / sm_unit;

        auto window_index   = qwen_vit::computeWindowIndex(T, H, W, kSpatialMergeSize,
                                                  kWindowSize, kPatchSize);
        auto reverse_idx    = qwen_vit::reverseWindowIndex(window_index);
        auto [cos_v, sin_v] = qwen_vit::computeSpatialCosSin(T, H, W, kSpatialMergeSize,
                                                inv_freq, window_index);
        const size_t cos_emb_dim = cos_v.size() / seq_len;  // 40 = half_freq * 2

        // ── patch_embed: graph(0), chunked ───────────────────────────────────
        // patch_embed is a per-patch projection with no cross-patch attention;
        // safe to process in independent chunks of patch_batch patches.
        std::vector<float> patch_out(seq_len * kEmbedDim);
        for (size_t off = 0; off < seq_len; off += patch_batch) {
            const size_t n_this = std::min(patch_batch, seq_len - off);
            const float* src = pixel_data.pixel_values.data() + px_offset + off * patch_elems;

            if (n_this < patch_batch) {
                // Zero-pad last chunk to full batch size.
                std::vector<float> padded(patch_batch * patch_elems, 0.0f);
                std::memcpy(padded.data(), src, n_this * patch_elems * sizeof(float));
                g_patch.write(g_patch.inputSpecs()[0].name, padded.data(), patch_batch * patch_elems);
            } else {
                g_patch.write(g_patch.inputSpecs()[0].name, src, patch_batch * patch_elems);
            }
            if (!g_patch.execute(tl))
                throw std::runtime_error("Qwen2OmniVisionEncoder: patch_embed execute failed");

            std::vector<float> batch_out(patch_batch * kEmbedDim);
            g_patch.read(g_patch.outputSpecs()[0].name, batch_out.data(), patch_batch * kEmbedDim);
            std::memcpy(patch_out.data() + off * kEmbedDim,
                        batch_out.data(),
                        n_this * kEmbedDim * sizeof(float));
        }

        // ── CPU windowing ─────────────────────────────────────────────────────
        auto windowed = qwen_vit::windowReorder(patch_out, n_groups, sm_unit, kEmbedDim, window_index);

        // ── vit_model: graph(1), chunked ─────────────────────────────────────
        // After windowing, patches from the same window are contiguous.
        // vit_batch is always a multiple of the window size, so each chunk
        // contains complete windows and can be processed independently.
        std::vector<float> vit_out_full(n_groups * kOutputDim);
        size_t groups_written = 0;

        for (size_t off = 0; off < seq_len; off += vit_batch) {
            const size_t n_this  = std::min(vit_batch, seq_len - off);
            const size_t ng_this = n_this / sm_unit;

            auto write_with_pad = [&](Graph& g, size_t spec_idx,
                                      const float* data, size_t n_valid, size_t n_full) {
                if (n_valid < n_full) {
                    std::vector<float> padded(n_full, 0.0f);
                    std::memcpy(padded.data(), data, n_valid * sizeof(float));
                    g.write(g.inputSpecs()[spec_idx].name, padded.data(), n_full);
                } else {
                    g.write(g.inputSpecs()[spec_idx].name, data, n_full);
                }
            };

            write_with_pad(g_vit, 0,
                           windowed.data() + off * kEmbedDim,
                           n_this * kEmbedDim, vit_batch * kEmbedDim);
            write_with_pad(g_vit, 1,
                           cos_v.data() + off * cos_emb_dim,
                           n_this * cos_emb_dim, vit_batch * cos_emb_dim);
            write_with_pad(g_vit, 2,
                           sin_v.data() + off * cos_emb_dim,
                           n_this * cos_emb_dim, vit_batch * cos_emb_dim);

            if (!g_vit.execute(tl))
                throw std::runtime_error("Qwen2OmniVisionEncoder: vit_model execute failed");

            const size_t chunk_groups = vit_batch / sm_unit;
            std::vector<float> chunk_out(chunk_groups * kOutputDim);
            g_vit.read(g_vit.outputSpecs()[0].name, chunk_out.data(), chunk_out.size());

            std::memcpy(vit_out_full.data() + groups_written * kOutputDim,
                        chunk_out.data(),
                        ng_this * kOutputDim * sizeof(float));
            groups_written += ng_this;
        }

        // ── Apply reverse indices ─────────────────────────────────────────────
        std::vector<float> ordered(n_groups * kOutputDim);
        for (size_t i = 0; i < reverse_idx.size(); ++i)
            std::memcpy(ordered.data() + i * kOutputDim,
                        vit_out_full.data() + reverse_idx[i] * kOutputDim,
                        kOutputDim * sizeof(float));

        all_embeddings.insert(all_embeddings.end(), ordered.begin(), ordered.end());
        px_offset += seq_len * patch_elems;
    }

    return all_embeddings;
}

// ── Qwen2OmniAudioEncoder ─────────────────────────────────────────────────────

Qwen2OmniAudioEncoder::Qwen2OmniAudioEncoder()
    : positional_embedding_(makeSinusoidalPositionalEmbedding(kMaxSourcePositions, kEmbedDim)) {}

std::vector<float> Qwen2OmniAudioEncoder::encode(const AudioData& audio_data) {
    if (audio_data.audio_features.empty()) return {};

    const int num_frames  = audio_data.num_frames;
    const int num_mel     = audio_data.num_mel_bins;
    const int chunk_len   = kNWindow * 2;  // 200 frames per helper0 chunk

    // Compute number of chunks
    const int valid_len    = num_frames;
    const int num_chunks   = (valid_len + chunk_len - 1) / chunk_len;
    const int padded_embed_len = (chunk_len - 1) / 2 + 1;  // = 100

    // Extract positional embedding slice [1, padded_embed_len, kEmbedDim]
    std::vector<float> pos_emb(padded_embed_len * kEmbedDim);
    std::memcpy(pos_emb.data(),
                positional_embedding_.data(),
                padded_embed_len * kEmbedDim * sizeof(float));

    TimeLog tl;

    // ── Stage 1: helper0 ─────────────────────────────────────────────────────
    // For each chunk of [num_mel, chunk_len]: helper0 → [1, 100, 1280]
    // Accumulate all chunk outputs: [num_chunks * 100, 1280]
    std::vector<float> all_helper0_out;
    all_helper0_out.reserve(num_chunks * kNWindow * kEmbedDim);

    Graph& g_h0 = graph(0);
    const std::string& h0_feat  = g_h0.inputSpecs()[0].name;
    const std::string& h0_mask  = g_h0.inputSpecs()[1].name;
    const std::string& h0_posem = g_h0.inputSpecs()[2].name;
    const std::string& h0_out   = g_h0.outputSpecs()[0].name;

    for (int c = 0; c < num_chunks; ++c) {
        const int start = c * chunk_len;
        const int end   = std::min(start + chunk_len, valid_len);
        const int clen  = end - start;

        // Feature chunk [num_mel, chunk_len] — zero-padded to [num_mel, chunk_len]
        std::vector<float> feat_chunk(num_mel * chunk_len, 0.0f);
        for (int m = 0; m < num_mel; ++m)
            std::memcpy(feat_chunk.data() + m * chunk_len,
                        audio_data.audio_features.data() + m * num_frames + start,
                        clen * sizeof(float));

        // Attention mask [1, 1, num_mel, chunk_len]: 1 for valid, 0 for padding
        std::vector<float> mask_chunk(chunk_len, 0.0f);
        std::fill(mask_chunk.begin(), mask_chunk.begin() + clen, 1.0f);

        g_h0.write(h0_feat,  feat_chunk.data(),   feat_chunk.size());
        g_h0.write(h0_mask,  mask_chunk.data(),   mask_chunk.size());
        g_h0.write(h0_posem, pos_emb.data(),       pos_emb.size());
        if (!g_h0.execute(tl))
            throw std::runtime_error("Qwen2OmniAudioEncoder: helper0 execute failed");

        std::vector<float> h0_result(kNWindow * kEmbedDim);
        g_h0.read(h0_out, h0_result.data(), h0_result.size());
        all_helper0_out.insert(all_helper0_out.end(), h0_result.begin(), h0_result.end());
    }

    // all_helper0_out: [num_chunks * 100, 1280]
    const size_t total_frames_h0 = num_chunks * kNWindow;

    // ── Stage 2: audio_encoder ────────────────────────────────────────────────
    // Each chunk: [1, 1280, 100, 1] → [1, 1280, 100, 1]
    // Process all frames in 100-frame windows.
    Graph& g_enc = graph(1);
    const std::string& enc_in  = g_enc.inputSpecs()[0].name;
    const std::string& enc_out = g_enc.outputSpecs()[0].name;

    std::vector<float> all_enc_out;
    all_enc_out.reserve(total_frames_h0 * kEmbedDim);

    for (size_t i = 0; i < total_frames_h0; i += kNWindow) {
        // Transpose [100, 1280] → [1280, 100], then add dims → [1, 1280, 100, 1]
        std::vector<float> enc_chunk(kEmbedDim * kNWindow);
        for (int d = 0; d < kEmbedDim; ++d)
            for (int t = 0; t < kNWindow; ++t)
                enc_chunk[d * kNWindow + t] =
                    all_helper0_out[(i + t) * kEmbedDim + d];

        g_enc.write(enc_in, enc_chunk.data(), enc_chunk.size());
        if (!g_enc.execute(tl))
            throw std::runtime_error("Qwen2OmniAudioEncoder: encoder execute failed");

        std::vector<float> enc_result(kEmbedDim * kNWindow);
        g_enc.read(enc_out, enc_result.data(), enc_result.size());

        // Transpose back [1280, 100] → [100, 1280]
        for (int t = 0; t < kNWindow; ++t)
            for (int d = 0; d < kEmbedDim; ++d)
                all_enc_out.push_back(enc_result[d * kNWindow + t]);
    }

    // ── Post-processing: extract valid, avg-pool by 2 ────────────────────────
    // valid_len_after_helper0 = ((valid_len - 1) / 2 + 1 - 2) / 2 + 1
    const int vlen_cnn1 = (valid_len - 1) / 2 + 1;
    const int vlen_out  = (vlen_cnn1 - 2) / 2 + 1;

    // Extract valid tokens from all_enc_out
    std::vector<float> valid_enc(vlen_out * kEmbedDim);
    std::memcpy(valid_enc.data(), all_enc_out.data(), vlen_out * kEmbedDim * sizeof(float));

    // Average pool: [vlen_out, 1280] → [vlen_out/2, 1280]
    std::vector<float> pooled = avgPool1d(valid_enc, vlen_out, kEmbedDim);
    const size_t pooled_len = pooled.size() / kEmbedDim;

    // Pad pooled_len to multiple of kNWindow for helper1
    const size_t padded_len = ((pooled_len + kNWindow - 1) / kNWindow) * kNWindow;
    if (pooled.size() < padded_len * kEmbedDim)
        pooled.resize(padded_len * kEmbedDim, 0.0f);

    // ── Stage 3: helper1 ─────────────────────────────────────────────────────
    // Each chunk: [100, 1280] → [100, 2048]
    Graph& g_h1 = graph(2);
    const std::string& h1_in  = g_h1.inputSpecs()[0].name;
    const std::string& h1_out = g_h1.outputSpecs()[0].name;

    std::vector<float> all_audio_out;
    all_audio_out.reserve(padded_len * kOutputDim);

    for (size_t i = 0; i < padded_len; i += kNWindow) {
        g_h1.write(h1_in, pooled.data() + i * kEmbedDim, kNWindow * kEmbedDim);
        if (!g_h1.execute(tl))
            throw std::runtime_error("Qwen2OmniAudioEncoder: helper1 execute failed");

        std::vector<float> h1_result(kNWindow * kOutputDim);
        g_h1.read(h1_out, h1_result.data(), h1_result.size());
        all_audio_out.insert(all_audio_out.end(), h1_result.begin(), h1_result.end());
    }

    // Trim to valid length
    all_audio_out.resize(pooled_len * kOutputDim);
    return all_audio_out;
}

// ── Qwen2OmniModel ────────────────────────────────────────────────────────────

Qwen2OmniModel::Qwen2OmniModel()
    : VLMModel(makeSpec()) {
    image_token_id_ = kImageTokenId;
    audio_token_id_ = kAudioTokenId;
    setEmbeddingProvider(std::make_unique<PrecomputedEmbeddingProvider>());
}

void Qwen2OmniModel::setEncoders(std::unique_ptr<Qwen2OmniVisionEncoder> vis,
                                  std::unique_ptr<Qwen2OmniAudioEncoder>  aud) {
    vision_encoder_ = std::move(vis);
    audio_encoder_  = std::move(aud);
}

void Qwen2OmniModel::setMRoPEProvider(std::unique_ptr<MRoPEInputProvider> provider) {
    mrope_provider_ = provider.get();
    addInputProvider(std::move(provider));
}

std::vector<float> Qwen2OmniModel::encodeVision(const PixelData& pixel_data) {
    return vision_encoder_->encode(pixel_data);
}

std::vector<float> Qwen2OmniModel::encodeAudio(const AudioData& audio_data) {
    return audio_encoder_->encode(audio_data);
}

void Qwen2OmniModel::preparePositions(const std::vector<int32_t>& input_ids,
                                       const VLMInput&             vlm_input,
                                       size_t                      n_past) {
    if (!mrope_provider_) return;

    // Build audio segment info from AudioVLMInput if present.
    std::vector<AudioSegmentInfo> audio_segments;
    if (const auto* a = dynamic_cast<const AudioVLMInput*>(&vlm_input)) {
        if (a->audio_data.num_frames > 0)
            audio_segments.push_back({audioFramesToLLMTokens(a->audio_data.num_frames)});
    }

    // Convert image_grid_thw to ImageGrid vector.
    std::vector<ImageGrid> image_grids(vlm_input.pixel_data.image_grid_thw.begin(),
                                       vlm_input.pixel_data.image_grid_thw.end());

    auto mrope = computeMRoPEPositions(
        input_ids, image_grids, audio_segments,
        kSpatialMergeSize,
        kVisionStartTokenId, kImageTokenId,
        kAudioStartTokenId, kAudioTokenId);

    const size_t seq_len = input_ids.size();

    // Apply n_past offset and accumulated mrope_deltas.
    for (size_t dim = 0; dim < 3; ++dim)
        for (size_t t = 0; t < seq_len; ++t)
            mrope.position_ids[dim * seq_len + t] += static_cast<int32_t>(n_past) + mrope_deltas_[dim];

    mrope_provider_->setPositionIds(mrope.position_ids, seq_len, n_past);

    // Accumulate mrope_deltas for the next turn.
    for (int d = 0; d < 3; ++d)
        mrope_deltas_[d] += mrope.deltas[d];

    // Sync accumulated deltas to provider so the decode fallback uses the correct offset.
    mrope_provider_->setMropeDeltas(mrope_deltas_);
}

void Qwen2OmniModel::clearPositions() {
    if (mrope_provider_) mrope_provider_->clearPositionIds();
}

// ── audioFramesToLLMTokens ─────────────────────────────────────────────────────

/*static*/ int32_t Qwen2OmniModel::audioFramesToLLMTokens(int32_t num_frames) {
    // Qwen2-Omni audio encoder: two CNN layers + avg-pool by 2.
    const int32_t cnn1_len = (num_frames - 1) / 2 + 1;
    const int32_t cnn2_len = (cnn1_len - 2) / 2 + 1;
    return cnn2_len / 2;
}

// ── Factory ───────────────────────────────────────────────────────────────────

std::unique_ptr<Qwen2OmniModel> makeModel(const QnnRuntimeConfig& runtime_cfg,
                                           const Qwen2OmniConfig&  config) {
    // Vision encoder
    auto vis_enc = std::make_unique<Qwen2OmniVisionEncoder>();
    if (!vis_enc->initialize(runtime_cfg, config.vision_config)) return nullptr;

    // Audio encoder
    // auto aud_enc = std::make_unique<Qwen2OmniAudioEncoder>();
    // if (!aud_enc->initialize(runtime_cfg, config.audio_config)) return nullptr;

    // LLM
    auto model = std::make_unique<Qwen2OmniModel>();
    model->setEncoders(std::move(vis_enc), /*aud_enc=*/nullptr);

    // MRoPE provider: section {16, 24, 24}, BLOCK interleaving
    model->setMRoPEProvider(std::make_unique<MRoPEInputProvider>(
        std::vector<int>{16, 24, 24}, 1000000.0f, MRoPEInterleaving::BLOCK));

    if (!model->initialize(runtime_cfg, config.llm_config)) return nullptr;

    return model;
}

} // namespace qwen2_omni
} // namespace geniex
