// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <algorithm>
#include <cctype>
#include <filesystem>
#include <stdexcept>
#include <string>

#include "utils/detail/json.hpp"

namespace geniex::phonon_detail {
inline std::filesystem::path bundleFile(const std::filesystem::path& root, const std::string& name) {
    const auto path = std::filesystem::u8path(name);
    if (name.empty() || path.has_root_path()) throw std::invalid_argument("Phonon bundle path must be relative");
    for (const auto& part : path)
        if (part == "..") throw std::invalid_argument("Phonon bundle path escapes root");
    const auto base     = std::filesystem::weakly_canonical(root);
    const auto full     = std::filesystem::weakly_canonical(root / path);
    const auto relative = full.lexically_relative(base);
    if (relative.empty() || *relative.begin() == "..")
        throw std::invalid_argument("Phonon bundle symlink escapes root");
    return full;
}
inline std::string contextForSoc(const qualla::json& entries, std::string soc) {
    std::transform(soc.begin(), soc.end(), soc.begin(), [](unsigned char c) { return std::toupper(c); });
    if (soc.empty()) throw std::invalid_argument("Phonon requires an explicit SoC");
    std::string found;
    for (const auto& entry : entries.items()) {
        if (!entry.value().contains("soc_models")) continue;
        for (const auto& target : entry.value().at("soc_models")) {
            if (target != soc) continue;
            if (!found.empty()) throw std::invalid_argument("Multiple Phonon contexts match " + soc);
            found = entry.value().at("file").get<std::string>();
        }
    }
    if (found.empty()) throw std::invalid_argument("No compiled Phonon context for " + soc);
    return found;
}
}  // namespace geniex::phonon_detail
