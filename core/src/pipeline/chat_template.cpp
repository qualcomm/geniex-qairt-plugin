// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#include "pipeline/chat_template.h"

#include <cstdio>

namespace geniex {

namespace {

// Minimal JSON string-content escaper for splicing user-provided strings
// (tool name / description) into a JSON object literal. Handles the escapes
// required by RFC 8259 §7. We deliberately do not pull in a JSON library here
// — the qairt core stays JSON-free.
void appendJsonEscaped(std::string& out, const std::string& s) {
    for (unsigned char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\b': out += "\\b";  break;
            case '\f': out += "\\f";  break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:
                if (c < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", c);
                    out += buf;
                } else {
                    out += static_cast<char>(c);
                }
        }
    }
}

// Serialises ChatTools into an OpenAI-format JSON array and appends it to
// `out`. Each element is:
//   {"type":"function","function":{"name":"...","description":"...","parameters":{...}}}
// When `indent` is true the array is pretty-printed with 2-space indentation
// (required by the Falcon3 Jinja template's tojson(indent=2) call).
void appendOpenAIToolsJson(std::string& out, const ChatTools& tools, bool indent = false) {
    const char* sep      = indent ? ",\n  " : ", ";
    const char* open     = indent ? "[\n  " : "[";
    const char* close    = indent ? "\n]"   : "]";
    out += open;
    bool first = true;
    for (const auto& t : tools) {
        if (!first) out += sep;
        first = false;
        out += "{\"type\": \"function\", \"function\": {\"name\": \"";
        appendJsonEscaped(out, t.name);
        out += "\", \"description\": \"";
        appendJsonEscaped(out, t.description);
        out += "\", \"parameters\": ";
        out += t.parameters_json.empty() ? "{}" : t.parameters_json;
        out += "}}";
    }
    out += close;
}

// Emits the Qwen3-style tools block as a single merged system message. Mirrors
// the official Qwen3 Jinja chat template: a `# Tools` heading, the function
// signatures wrapped in <tools>...</tools>, and the <tool_call> instruction
// footer.
void appendChatMLToolsSystem(std::string&        out,
                             const std::string&  system_prompt,
                             const ChatTools&    tools) {
    out += "<|im_start|>system\n";
    if (!system_prompt.empty()) {
        out += system_prompt;
        out += "\n\n";
    }
    out += "# Tools\n\n"
           "You may call one or more functions to assist with the user query.\n\n"
           "You are provided with function signatures within "
           "<tools></tools> XML tags:\n<tools>";
    for (const auto& t : tools) {
        out += "\n{\"type\": \"function\", \"function\": {\"name\": \"";
        appendJsonEscaped(out, t.name);
        out += "\", \"description\": \"";
        appendJsonEscaped(out, t.description);
        out += "\", \"parameters\": ";
        out += t.parameters_json.empty() ? "{}" : t.parameters_json;
        out += "}}";
    }
    out += "\n</tools>\n\n"
           "For each function call, return a json object with function name "
           "and arguments within <tool_call></tool_call> XML tags:\n"
           "<tool_call>\n"
           "{\"name\": <function-name>, \"arguments\": <args-json-object>}\n"
           "</tool_call><|im_end|>\n";
}

} // namespace

std::string chatMLTemplate(const std::string& user_message,
                           const std::string& system_prompt,
                           const ChatTools&   tools,
                           bool               enable_thinking) {
    std::string result;
    if (!tools.empty()) {
        appendChatMLToolsSystem(result, system_prompt, tools);
    } else if (!system_prompt.empty()) {
        result += "<|im_start|>system\n" + system_prompt + "<|im_end|>\n";
    }
    result += "<|im_start|>user\n" + user_message + "<|im_end|>\n";
    result += "<|im_start|>assistant\n";
    if (!enable_thinking)
        result += "<think>\n\n</think>\n\n";
    return result;
}

std::string phiChatTemplate(const std::string& user_message,
                            const std::string& system_prompt,
                            const ChatTools&   tools,
                            bool /*enable_thinking*/) {
    std::string result;
    if (!system_prompt.empty() || !tools.empty()) {
        result += "<|system|>";
        result += system_prompt;
        if (!tools.empty()) {
            // Phi4-mini tool format: <|tool|>[...OpenAI tools array...]<|/tool|>
            // Mirrors the official Phi4-mini Jinja chat template.
            result += "<|tool|>";
            appendOpenAIToolsJson(result, tools);
            result += "<|/tool|>";
        }
        result += "<|end|>\n";
    }
    result += "<|user|>\n" + user_message + "<|end|>\n";
    result += "<|assistant|>";
    return result;
}

// Llama3 header-id format: <|start_header_id|>role<|end_header_id|>...<|eot_id|>
// Used by Llama 3, Llama 3.1, and Llama 3.2 model families.
//
// Tool-call format mirrors the Llama3 template:
//   - System block gets a fixed tool-capability preamble appended.
//   - Tool signatures are injected into the user turn (not the system block),
//     preceded by a "Given the following functions..." instruction.
//   - Response format: {"name": function_name, "parameters": {args}}
//     (note: "parameters", not "arguments" — this is Llama3's convention).
std::string llama3ChatTemplate(const std::string& user_message,
                               const std::string& system_prompt,
                               const ChatTools&   tools,
                               bool /*enable_thinking*/) {
    std::string out = "<|begin_of_text|>";

    // System block — emitted when system_prompt is set, or when tools are
    // present (tools require the capability preamble).
    if (!system_prompt.empty() || !tools.empty()) {
        out += "<|start_header_id|>system<|end_header_id|>\n\n"
               "Cutting Knowledge Date: December 2023\n";
        if (!system_prompt.empty()) {
            out += system_prompt;
            out += "\n";
        }
        if (!tools.empty()) {
            out += "When you receive a tool call response, use the output to format "
                   "an answer to the original user question.\n\n"
                   "You are a helpful assistant with tool calling capabilities.";
        }
        out += "<|eot_id|>";
    }

    out += "<|start_header_id|>user<|end_header_id|>\n\n";

    if (!tools.empty()) {
        // Inject tool list into the user turn, one JSON object per line.
        out += "Given the following functions, please respond with a JSON for a "
               "function call with its proper arguments that best answers the given prompt.\n\n"
               "Respond in the format {\"name\": function name, \"parameters\": dictionary "
               "of argument name and its value}. Do not use variables.\n\n";
        for (const auto& t : tools) {
            out += "{\"type\": \"function\", \"function\": {\"name\": \"";
            appendJsonEscaped(out, t.name);
            out += "\", \"description\": \"";
            appendJsonEscaped(out, t.description);
            out += "\", \"parameters\": ";
            out += t.parameters_json.empty() ? "{}" : t.parameters_json;
            out += "}}\n";
        }
        out += "\n";
    }

    out += user_message + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n";
    return out;
}

} // namespace geniex
