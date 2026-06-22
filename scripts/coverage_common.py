# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
"""Shared coverage configuration: the single source of truth for which files
count toward coverage, plus the instrumented test targets.

Both the local full-tree report (scripts/coverage.ps1) and the CI diff gate
(scripts/diff_coverage.py) import these constants so the measured surface
never drifts between the two."""

# Test targets that do NOT link geniex-proc. The geniex-proc -> tokenizers-cpp
# Rust/cc-rs chain cannot be built under clang-cl (mixes clang headers with
# cl.exe, fatal C1012), so llm_model_test is excluded from the coverage build.
# Its orchestration logic is still exercised by the MSVC build-and-test job.
COVERAGE_TEST_TARGETS = [
    "utils_test",
    "graph_test",
    "input_provider_test",
]

# Only first-party implementation files under core/ count toward coverage.
# llvm-cov reports coverage per source file seen in the binary; everything
# else (third-party headers, test code, the QNN fmt/type vendored headers,
# GoogleTest) is noise and is filtered out via -ignore-filename-regex.
#
# Paths come from two sources with different shapes: git diff yields
# repo-relative ("core/src/x.cpp") while llvm-cov yields absolute
# ("C:/gx/core/src/x.cpp"). The leading boundary therefore matches either a
# separator or start-of-string. Separators match both "/" and "\".
COVERAGE_INCLUDE_REGEX = r"(?:^|[\\/])core[\\/](src|include)[\\/]"

# Files under the include regex that still should not be measured.
# Each fragment allows a leading separator OR start-of-string so it matches
# both absolute (llvm-cov) and repo-relative (git diff) paths.
COVERAGE_EXCLUDE_REGEX_PARTS = [
    r"(?:^|[\\/])tests[\\/]",           # the test sources themselves
    r"(?:^|[\\/])third-party[\\/]",     # vendored deps
    r"(?:^|[\\/])qnn-api[\\/]",         # QNN glue + vendored fmt/json headers
    r"[\\/]_deps[\\/]",                 # FetchContent (googletest) build tree
    r"[\\/]googletest",                 # GoogleTest headers
    r"[\\/]logging\.(h|cpp)$",          # logging shim: trivial, not unit-tested
]
COVERAGE_EXCLUDE_REGEX = "|".join(COVERAGE_EXCLUDE_REGEX_PARTS)


def is_covered_source(path: str) -> bool:
    """True if `path` is a first-party file that counts toward coverage."""
    import re

    if not re.search(COVERAGE_INCLUDE_REGEX, path):
        return False
    return not re.search(COVERAGE_EXCLUDE_REGEX, path)
