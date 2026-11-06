// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause
#include <gtest/gtest.h>

#include "phonon/bundle.h"
namespace {
TEST(PhononBundle, RequiresExplicitUnambiguousDevice) {
    const qualla::json entries = {{"xelite", {{"file", "model.bin"}, {"soc_models", {"SC8380XP"}}}}};
    EXPECT_EQ(geniex::phonon_detail::contextForSoc(entries, "sc8380xp"), "model.bin");
    EXPECT_THROW(geniex::phonon_detail::contextForSoc(entries, ""), std::invalid_argument);
    EXPECT_THROW(geniex::phonon_detail::contextForSoc(entries, "SM8750"), std::invalid_argument);
    auto duplicate     = entries;
    duplicate["other"] = entries["xelite"];
    EXPECT_THROW(geniex::phonon_detail::contextForSoc(duplicate, "SC8380XP"), std::invalid_argument);
}
TEST(PhononBundle, RejectsTraversalAndRootedFiles) {
    const auto root = std::filesystem::temp_directory_path();
    EXPECT_EQ(
        geniex::phonon_detail::bundleFile(root, "model.bin"), std::filesystem::weakly_canonical(root / "model.bin"));
    for (const char* file : {"", "../outside.bin", "/outside.bin"})
        EXPECT_THROW(geniex::phonon_detail::bundleFile(root, file), std::invalid_argument);
#ifdef _WIN32
    for (const char* file : {"C:outside.bin", "C:\\outside.bin", "\\outside.bin"})
        EXPECT_THROW(geniex::phonon_detail::bundleFile(root, file), std::invalid_argument);
#endif
}
}  // namespace
