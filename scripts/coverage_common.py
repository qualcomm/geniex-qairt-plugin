# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
"""Single source of truth for the coverage surface, shared by coverage.ps1
(local report) and diff_coverage.py (CI gate) so the two never drift."""

# All unit-test exes are instrumented. The whole tree (incl. the tokenizers
# Rust chain via geniex-proc) builds under clang-cl once CC/CXX point at it.
COVERAGE_TEST_TARGETS = [
    "utils_test",
    "graph_test",
    "input_provider_test",
    "llm_utils_test",
    "llm_model_test",
    "llm_spec_loader_test",
    "model_test",
    "vit_utils_test",
    "vlm_utils_test",
    "vlm_input_provider_test",
    "vlm_model_test",
]

# Boundary is (?:^|[\\/]) so the same regex matches both git-diff repo-relative
# paths (core/src/x.cpp) and llvm-cov absolute paths (C:/gx/core/src/x.cpp).
COVERAGE_INCLUDE_REGEX = r"(?:^|[\\/])core[\\/](src|include)[\\/]"

COVERAGE_EXCLUDE_REGEX_PARTS = [
    r"(?:^|[\\/])tests[\\/]",
    r"(?:^|[\\/])third-party[\\/]",
    r"(?:^|[\\/])qnn-api[\\/]",  # QNN glue + vendored fmt/json
    r"[\\/]_deps[\\/]",  # FetchContent build tree
    r"[\\/]googletest",
    r"[\\/]logging\.(h|cpp)$",  # logging shim, not unit-tested
    # Device-only: QNN backend bring-up, FastRPC/HTP detection, thread pools,
    # CPU affinity -- exercised only by on-device integration tests, not on CPU.
    r"[\\/]runtime\.(h|cpp)$",
    r"[\\/]threadpool\.(h|cpp)$",
    r"[\\/]vlm[\\/]vision_encoder\.(h|cpp)$",
    # model_init.cpp holds Model::initialize() (QNN backend load) and
    # qnnLogCallback -- device-only, exercised only by on-device integration tests.
    r"[\\/]model_init\.cpp$",
]
COVERAGE_EXCLUDE_REGEX = "|".join(COVERAGE_EXCLUDE_REGEX_PARTS)


def is_covered_source(path: str) -> bool:
    """True if `path` is a first-party file that counts toward coverage."""
    import re

    if not re.search(COVERAGE_INCLUDE_REGEX, path):
        return False
    return not re.search(COVERAGE_EXCLUDE_REGEX, path)
