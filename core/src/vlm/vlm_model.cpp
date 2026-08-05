// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#include "vlm/vlm_model.h"

#include <algorithm>
#include <stdexcept>

#include "vlm/vlm_types.h"

namespace geniex {

VLMModel::VLMModel(LLMSpec spec) : LLMModel(std::move(spec)) {}

VLMModel::VLMModel(LLMSpec spec, ParsedGenieConfig gc) : LLMModel(std::move(spec), std::move(gc)) {}

bool VLMModel::onInitialized() { return LLMModel::onInitialized(); }

void VLMModel::setEmbeddingProvider(std::unique_ptr<PrecomputedEmbeddingProvider> provider) {
    emb_provider_ = provider.get();
    addInputProvider(std::move(provider));
}

void VLMModel::preparePositions(const std::vector<int32_t>&, const VLMInput&, size_t) {}
void VLMModel::clearPositions() {}

/*static*/ void VLMModel::maskedScatter(std::vector<float>& input_embeds, const std::vector<float>& multimodal_embeds,
    const std::vector<int32_t>& input_ids, int32_t target_token_id, size_t hidden_size) {
    size_t src_row = 0;
    for (size_t i = 0; i < input_ids.size(); ++i) {
        if (input_ids[i] == target_token_id) {
            std::copy(multimodal_embeds.data() + src_row * hidden_size,
                multimodal_embeds.data() + src_row * hidden_size + hidden_size,
                input_embeds.data() + i * hidden_size);
            ++src_row;
        }
    }
}

void VLMModel::prepareEmbeddings(const std::vector<int32_t>& prompt_tokens, const VLMInput& vlm_input) {
    if (!emb_provider_) {
        throw std::runtime_error("VLMModel: no embedding provider registered");
    }

    auto         text_embeds = emb_provider_->lookupBatch(prompt_tokens);
    const size_t hidden_size = text_embeds.size() / prompt_tokens.size();

    if (!vlm_input.pixel_data.pixel_values.empty()) {
        auto vision_embeds = encodeVision(vlm_input.pixel_data);
        maskedScatter(text_embeds, vision_embeds, prompt_tokens, image_token_id_, hidden_size);
    }

    emb_provider_->setBuffer(std::move(text_embeds), nPast());
}

void VLMModel::releaseEmbeddings() {
    if (emb_provider_) emb_provider_->clearBuffer();
}

std::vector<int32_t> VLMModel::generate(const std::vector<int32_t>& prompt_tokens, const VLMInput& vlm_input,
    const GenerationConfig& gen_cfg, std::function<bool(int32_t)> token_callback) {
    prepareEmbeddings(prompt_tokens, vlm_input);
    preparePositions(prompt_tokens, vlm_input, nPast());

    auto result = LLMModel::generate(prompt_tokens, gen_cfg, token_callback);

    releaseEmbeddings();
    clearPositions();

    return result;
}

}  // namespace geniex
