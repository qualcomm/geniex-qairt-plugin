// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <string>
#include <vector>

#include "geniex_export.h"

namespace geniex {

// `parameters_json` is the JSON Schema object describing the function's
// parameters, kept as a serialized JSON string so templates can splice it in
// verbatim without growing a JSON model in core. It is the producer's
// responsibility to ensure it is valid JSON.
struct ChatTool {
    std::string name;
    std::string description;
    std::string parameters_json;
};

using ChatTools = std::vector<ChatTool>;

// Signature for all chat-template formatters.
// Parameters: user_message, system_prompt, tools, enable_thinking.
//
// system_prompt contract:
//   - Empty system_prompt means "emit no system block at all" — formatters
//     MUST NOT inject a hard-coded default. Defaulting was a footgun: the
//     pipeline clears system_prompt after applyChatTemplate(), so a fallback
//     fires on every subsequent turn, silently re-injecting a system block
//     into the KV-cache context. It would also override model-card-specific
//     defaults the caller might rely on. Plugins that want a default (e.g.
//     "You are a helpful AI assistant.") must inject it themselves before
//     calling setSystemPrompt(), and only on the first turn.
//   - On the first turn, the caller is expected to pass the conversation's
//     system_prompt (if any). Passing it again on subsequent turns will
//     inject a second system block — that is the caller's responsibility,
//     not the formatter's.
//
// `tools` may be empty. When non-empty, formatters that support tool calls
// inject a tools block into the system message; formatters without tool-call
// support ignore the parameter.
using ChatTemplateFunc = std::string(*)(
    const std::string& user_message,
    const std::string& system_prompt,
    const ChatTools&   tools,
    bool               enable_thinking);

// ChatML format — used by Qwen3, Qwen2.5, and other ChatML-based models.
GENIEX_API std::string chatMLTemplate(
    const std::string& user_message,
    const std::string& system_prompt,
    const ChatTools&   tools,
    bool               enable_thinking);

// Phi format — used by Phi3.5 and Phi4.
GENIEX_API std::string phiChatTemplate(
    const std::string& user_message,
    const std::string& system_prompt,
    const ChatTools&   tools,
    bool               enable_thinking);

// Llama3 header-id format — used by Llama 3, Llama 3.1, and Llama 3.2 model families.
GENIEX_API std::string llama3ChatTemplate(
    const std::string& user_message,
    const std::string& system_prompt,
    const ChatTools&   tools,
    bool               enable_thinking);

} // namespace geniex
