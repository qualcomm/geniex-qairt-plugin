// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause
//
// Unit tests for core/include/runtime.h — which folder the QNN/HTP runtime
// libraries load from, and what layout that folder is in.
//
// selectHtpDir is deliberately pure (no filesystem, no getenv, no device query),
// so those tests cover the real decision logic rather than a mock of it. The
// caller supplies `bundled_dir_exists`; validating the choice is what
// locateHtpHostLibDir / collectHexagonSkelPath do, and those are exercised
// against skeleton directory trees below. Nothing here needs a device.

#include "runtime.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <filesystem>
#include <fstream>
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

// ── Layout translation: flat folder vs QAIRT SDK root ────────────────────────
//
// These touch the filesystem (the functions under test are the ones that decide
// what a folder *is*), so each builds a skeleton tree under a unique temp dir.
// Files are empty: only their names and locations are read.

class HtpLayout : public ::testing::Test {
   protected:
    void SetUp() override {
        root_ = fs::temp_directory_path() /
                ("geniex_htp_layout_" + std::string(::testing::UnitTest::GetInstance()->current_test_info()->name()));
        fs::remove_all(root_);
        fs::create_directories(root_);
    }

    void TearDown() override { fs::remove_all(root_); }

    void touch(const fs::path& relative) const {
        const auto full = root_ / relative;
        fs::create_directories(full.parent_path());
        std::ofstream(full).put('\0');
    }

    // The host library name the running platform looks for.
    static std::string backendLib() { return geniex::kHtpBackendLib; }

    fs::path root_;
};

TEST_F(HtpLayout, FlatFolderResolvesToItself) {
    touch(backendLib());

    EXPECT_EQ(geniex::locateHtpHostLibDir(root_), root_);
}

TEST_F(HtpLayout, SdkRootResolvesToTheTripleSubfolder) {
    const fs::path triple = fs::path("lib") / geniex::kHtpHostLibTriple;
    touch(triple / backendLib());

    EXPECT_EQ(geniex::locateHtpHostLibDir(root_), root_ / triple);
}

// SDK releases rename the triple (the Linux gcc suffix moves), so an unrecognised
// one must still resolve rather than fail on a name mismatch.
TEST_F(HtpLayout, UnknownTripleIsFoundByScanningLib) {
    touch(fs::path("lib") / "aarch64-oe-linux-gcc99.9" / backendLib());

    EXPECT_EQ(geniex::locateHtpHostLibDir(root_), root_ / "lib" / "aarch64-oe-linux-gcc99.9");
}

// A flat folder wins over a lib/ subtree in the same root: that is the bundled
// htp-files/ shape, and re-deriving it from lib/ would be a slower path to the
// same answer -- or a different one, if both carry a backend.
TEST_F(HtpLayout, FlatLayoutTakesPrecedenceOverLibSubtree) {
    touch(backendLib());
    touch(fs::path("lib") / geniex::kHtpHostLibTriple / backendLib());

    EXPECT_EQ(geniex::locateHtpHostLibDir(root_), root_);
}

TEST_F(HtpLayout, NeitherLayoutYieldsEmpty) {
    touch("README.txt");
    fs::create_directories(root_ / "lib" / "x86_64-linux-clang");  // no backend inside

    EXPECT_TRUE(geniex::locateHtpHostLibDir(root_).empty());
}

TEST_F(HtpLayout, MissingRootYieldsEmpty) {
    EXPECT_TRUE(geniex::locateHtpHostLibDir(root_ / "does-not-exist").empty());
}

TEST_F(HtpLayout, SkelPathPrefersUnsignedAndCoversEveryArch) {
    touch(fs::path("lib") / "hexagon-v73" / "unsigned" / "libQnnHtpV73Skel.so");
    touch(fs::path("lib") / "hexagon-v81" / "unsigned" / "libQnnHtpV81Skel.so");

    const std::string joined = geniex::collectHexagonSkelPath(root_);

    EXPECT_NE(joined.find((root_ / "lib" / "hexagon-v73" / "unsigned").string()), std::string::npos);
    EXPECT_NE(joined.find((root_ / "lib" / "hexagon-v81" / "unsigned").string()), std::string::npos);
    // Two entries, so exactly one separator.
    EXPECT_EQ(std::count(joined.begin(), joined.end(), geniex::kHtpPathSep), 1);
}

// Some SDKs ship an arch folder without an unsigned/ subfolder; fall back to the
// arch folder itself rather than dropping that arch off the path entirely.
TEST_F(HtpLayout, SkelPathFallsBackToTheArchFolder) {
    touch(fs::path("lib") / "hexagon-v75" / "libQnnHtpV75Skel.so");

    EXPECT_EQ(geniex::collectHexagonSkelPath(root_), (root_ / "lib" / "hexagon-v75").string());
}

// A flat runtime folder keeps skels beside the host libs, so there is nothing for
// this to collect -- resolveHtpPaths uses hasHexagonSkels for that shape instead.
TEST_F(HtpLayout, SkelPathEmptyForAFlatFolder) {
    touch(backendLib());
    touch("libQnnHtpV73Skel.so");

    EXPECT_TRUE(geniex::collectHexagonSkelPath(root_).empty());
}

TEST_F(HtpLayout, SkelPathIgnoresNonHexagonLibFolders) {
    touch(fs::path("lib") / geniex::kHtpHostLibTriple / backendLib());

    EXPECT_TRUE(geniex::collectHexagonSkelPath(root_).empty());
}

// ── Rung 5, continued ────────────────────────────────────────────────────────

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
