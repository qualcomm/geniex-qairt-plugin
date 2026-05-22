// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "llm/input_provider.h"
#include "llm/llm_model.h"
#include "llm/llm_spec_loader.h"
#include "llm/llm_types.h"
#include "logging.h"
#include "pipeline/chat_template.h"
#include "pipeline/llm_pipeline.h"

namespace geniex {

// Falcon3 instruct format: <|system|>\n...\n<|user|>\n...\n<|assistant|>\n
// Tool calls are supported: when tools are non-empty a <tools>...</tools> block
// is injected into the system message, mirroring the official Falcon3 Jinja
// chat template.
inline std::string falcon3ChatTemplate(const std::string& user_message, const std::string& system_prompt,
    const ChatTools& tools, bool /*enable_thinking*/) {
    std::string result;
    if (!tools.empty()) {
        result += "<|system|>\n";
        if (!system_prompt.empty()) {
            result += system_prompt;
            result += "\n\n";
        }
        result +=
            "# Tools\n\n"
            "You have access to the following functions. You MUST use them to answer "
            "questions when needed. For each function call, you MUST return a JSON "
            "object inside <tool_call></tool_call> tags.\n\n"
            "<tools>";
        result += "[\n";
        bool first = true;
        for (const auto& t : tools) {
            if (!first) result += ",\n";
            first = false;
            result += "  {\"type\": \"function\", \"function\": {\"name\": \"";
            for (unsigned char c : t.name) {
                if (c == '"')
                    result += "\\\"";
                else if (c == '\\')
                    result += "\\\\";
                else
                    result += static_cast<char>(c);
            }
            result += "\", \"description\": \"";
            for (unsigned char c : t.description) {
                if (c == '"')
                    result += "\\\"";
                else if (c == '\\')
                    result += "\\\\";
                else if (c == '\n')
                    result += "\\n";
                else
                    result += static_cast<char>(c);
            }
            result += "\", \"parameters\": ";
            result += t.parameters_json.empty() ? "{}" : t.parameters_json;
            result += "}}";
        }
        result += "\n]";
        result +=
            "</tools>\n\n"
            "# Output Format\n\n"
            "Your response MUST follow this format when making function calls:\n"
            "<tool_call>\n"
            "[\n"
            "  {\"name\": \"function_name\", \"arguments\": {\"arg1\": \"value1\", \"arg2\": \"value2\"}},\n"
            "  {\"name\": \"another_function\", \"arguments\": {\"arg\": \"value\"}}\n"
            "]\n"
            "</tool_call>\n"
            "If no function calls are needed, respond normally without the tool_call tags.\n";
    } else {
        if (!system_prompt.empty()) {
            result += "<|system|>\n" + system_prompt + "\n";
        }
    }
    result += "<|user|>\n" + user_message + "\n";
    result += "<|assistant|>\n";
    return result;
}

namespace falcon3 {

inline LLMModel makeModel(const ModelConfig& model_cfg) {
    const auto bundle = bundleDirOf(model_cfg);
    auto       meta   = parseQAIRTMetadata(bundle);
    auto       gc     = parseGenieConfig(bundle);

    LLMModel m(buildSpec(meta, gc));
    m.addInputProvider(makeEmbeddingProvider(meta, gc));
    m.addInputProvider(makeRoPEProvider(meta, gc));
    return m;
}

inline std::optional<LLMPipeline> makePipeline(const QnnRuntimeConfig& runtime_cfg, const ModelConfig& model_cfg) {
    try {
        LLMPipeline pipe;
        if (!pipe.create(falcon3ChatTemplate, makeModel(model_cfg), runtime_cfg, model_cfg)) return std::nullopt;
        return pipe;
    } catch (const std::exception& e) {
        GENIEX_LOG_ERROR("falcon3::makePipeline failed: {}", e.what());
        return std::nullopt;
    }
}

}  // namespace falcon3
}  // namespace geniex
