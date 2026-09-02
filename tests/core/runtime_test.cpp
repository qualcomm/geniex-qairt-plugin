// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause
//
// Unit tests for selectHtpDir in core/include/runtime.h — the precedence rules
// deciding which folder the QNN/HTP runtime libraries load from.
//
// selectHtpDir is deliberately pure (no filesystem, no getenv, no device query),
// so these tests cover the real decision logic rather than a mock of it. The
// caller supplies `bundled_dir_exists` and does the validating; that part is
// covered by the end-to-end runs, not here.

#include "runtime.h"

#include <gtest/gtest.h>

#include <filesystem>
#include <optional>
#include <string>

namespace {

namespace fs = std::filesystem;

const fs::path kCoreDir  = fs::path("C:/geniex/lib/qairt");
const fs::path kBundled  = kCoreDir / "htp-files";
const char*    kExternal = "D:/qairt/2.48/host-libs";
const char*    kFromCfg  = "E:/custom/runtime";

// ── Rung 4: bundled runtime, the default install ─────────────────────────────

TEST(SelectHtpDir, BundledWhenNothingOverrides) {
    const auto choice = geniex::selectHtpDir(std::nullopt, nullptr, kCoreDir, /*bundled_dir_exists=*/true);

    EXPECT_EQ(choice.dir, kBundled);
    EXPECT_EQ(choice.source, geniex::HtpDirSource::Bundled);
}

// ── Rung 3: GENIEX_QNN_LIB ───────────────────────────────────────────────────

TEST(SelectHtpDir, EnvironmentOverridesBundled) {
    const auto choice = geniex::selectHtpDir(std::nullopt, kExternal, kCoreDir, /*bundled_dir_exists=*/true);

    EXPECT_EQ(choice.dir, fs::path(kExternal));
    EXPECT_EQ(choice.source, geniex::HtpDirSource::Environment);
}

// An exported-but-empty variable is the shell's way of saying "unset"; treating
// it as a path would resolve every library to a bare relative filename.
TEST(SelectHtpDir, EmptyEnvironmentIsIgnored) {
    const auto choice = geniex::selectHtpDir(std::nullopt, "", kCoreDir, /*bundled_dir_exists=*/true);

    EXPECT_EQ(choice.dir, kBundled);
    EXPECT_EQ(choice.source, geniex::HtpDirSource::Bundled);
}

// ── Rung 2: QnnRuntimeConfig::htp_dir ────────────────────────────────────────

TEST(SelectHtpDir, ConfigFieldOverridesBundled) {
    const auto choice = geniex::selectHtpDir(std::string(kFromCfg), nullptr, kCoreDir, /*bundled_dir_exists=*/true);

    EXPECT_EQ(choice.dir, fs::path(kFromCfg));
    EXPECT_EQ(choice.source, geniex::HtpDirSource::ConfigField);
}

// An explicit API argument is more specific than an ambient variable, so it wins.
TEST(SelectHtpDir, ConfigFieldBeatsEnvironment) {
    const auto choice = geniex::selectHtpDir(std::string(kFromCfg), kExternal, kCoreDir, /*bundled_dir_exists=*/true);

    EXPECT_EQ(choice.dir, fs::path(kFromCfg));
    EXPECT_EQ(choice.source, geniex::HtpDirSource::ConfigField);
}

TEST(SelectHtpDir, EmptyConfigFieldFallsThroughToEnvironment) {
    const auto choice = geniex::selectHtpDir(std::string(""), kExternal, kCoreDir, /*bundled_dir_exists=*/true);

    EXPECT_EQ(choice.dir, fs::path(kExternal));
    EXPECT_EQ(choice.source, geniex::HtpDirSource::Environment);
}

// ── Rung 5: flattened deployments ────────────────────────────────────────────

// Some packaging drops the runtime libraries directly beside geniex_core rather
// than in an htp-files/ subfolder (Android). Without this rung those layouts
// resolve to a nonexistent htp-files/ and fail to load.
TEST(SelectHtpDir, FlatLayoutWhenBundledDirAbsent) {
    const auto choice = geniex::selectHtpDir(std::nullopt, nullptr, kCoreDir, /*bundled_dir_exists=*/false);

    EXPECT_EQ(choice.dir, kCoreDir);
    EXPECT_EQ(choice.source, geniex::HtpDirSource::CoreDirFlat);
}

// An override must not be downgraded to the flat fallback just because the
// bundled folder happens to be missing.
TEST(SelectHtpDir, OverridesStillWinWhenBundledDirAbsent) {
    const auto from_env = geniex::selectHtpDir(std::nullopt, kExternal, kCoreDir, /*bundled_dir_exists=*/false);
    EXPECT_EQ(from_env.dir, fs::path(kExternal));
    EXPECT_EQ(from_env.source, geniex::HtpDirSource::Environment);

    const auto from_cfg = geniex::selectHtpDir(std::string(kFromCfg), nullptr, kCoreDir, /*bundled_dir_exists=*/false);
    EXPECT_EQ(from_cfg.dir, fs::path(kFromCfg));
    EXPECT_EQ(from_cfg.source, geniex::HtpDirSource::ConfigField);
}

}  // namespace
