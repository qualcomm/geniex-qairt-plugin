#!/usr/bin/env python3
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
"""Gate patch coverage: fail if the lines a PR adds/modifies under core/ fall
below a threshold. Full-file coverage is the wrong PR signal -- it punishes
small edits to legacy files and hides untested new code in well-covered ones.

Intersects `git diff` new-side line ranges with `llvm-cov export` per-line
data. stdlib only.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from coverage_common import is_covered_source  # noqa: E402


def changed_lines(base_ref: str) -> dict[str, set[int]]:
    """Map each changed first-party file -> new-file line numbers it adds."""
    diff = subprocess.run(
        ["git", "diff", "--unified=0", "--no-color", f"{base_ref}...HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout

    result: dict[str, set[int]] = defaultdict(set)
    current: str | None = None
    new_lineno = 0
    hunk_re = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@")

    for line in diff.splitlines():
        if line.startswith("+++ "):
            path = line[4:].strip()
            path = path[2:] if path.startswith("b/") else path
            current = path if (path != "/dev/null" and is_covered_source(path)) else None
        elif line.startswith("@@"):
            m = hunk_re.match(line)
            new_lineno = int(m.group(1)) if m else 0
        elif current is not None:
            if line.startswith("+"):
                result[current].add(new_lineno)
                new_lineno += 1
            elif not line.startswith("-"):  # deletions don't advance new-file count
                new_lineno += 1
    return result


def covered_lines(export_json: dict) -> dict[str, dict[int, bool]]:
    """Map each first-party file -> {line: was_executed} for executable lines.

    llvm-cov segment = [line, col, count, hasCount, isRegionEntry, isGapRegion].
    A line is executable if a counted, non-gap region starts on it; executed if
    any such region has count > 0."""
    files: dict[str, dict[int, bool]] = {}
    for data in export_json.get("data", []):
        for f in data.get("files", []):
            norm = f["filename"].replace("\\", "/")
            if not is_covered_source(norm):
                continue
            line_hit: dict[int, bool] = {}
            for seg in f.get("segments", []):
                line, _col, count, has_count, _entry, is_gap = seg[:6]
                if not has_count or is_gap:
                    continue
                line_hit[line] = line_hit.get(line, False) or count > 0
            files[norm] = line_hit
    return files


def match_file(diff_path: str, cov_files: dict[str, dict[int, bool]]) -> dict[int, bool] | None:
    """Match a repo-relative diff path to an absolute llvm-cov path by suffix."""
    diff_norm = diff_path.replace("\\", "/")
    for cov_path, lines in cov_files.items():
        if cov_path.endswith(diff_norm):
            return lines
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--export-json", required=True, help="llvm-cov export JSON file")
    ap.add_argument("--base-ref", required=True, help="merge-base ref (e.g. origin/main)")
    ap.add_argument("--threshold", type=float, default=85.0, help="min percent (default 85)")
    args = ap.parse_args()

    with open(args.export_json, encoding="utf-8") as fh:
        export = json.load(fh)

    diff = changed_lines(args.base_ref)
    cov = covered_lines(export)

    if not diff:
        print("No changed first-party source lines under coverage scope. Passing.")
        return 0

    total_exec = 0
    total_covered = 0
    rows: list[tuple[str, int, int, list[int]]] = []

    for path, lines in sorted(diff.items()):
        cov_lines = match_file(path, cov)
        if cov_lines is None:
            continue  # not compiled into the instrumented tests; don't punish
        exec_changed = sorted(ln for ln in lines if ln in cov_lines)
        if not exec_changed:
            continue
        missed = [ln for ln in exec_changed if not cov_lines[ln]]
        covered = len(exec_changed) - len(missed)
        total_exec += len(exec_changed)
        total_covered += covered
        rows.append((path, covered, len(exec_changed), missed))

    print(f"{'File':<48}{'Covered':>10}{'Changed':>10}{'Cover':>9}")
    print("-" * 77)
    for path, covered, n, missed in rows:
        pct = 100.0 * covered / n if n else 100.0
        print(f"{path:<48}{covered:>10}{n:>10}{pct:>8.1f}%")
        if missed:
            print(f"    uncovered changed lines: {missed}")
    print("-" * 77)

    if total_exec == 0:
        print("No executable changed lines measured. Passing.")
        return 0

    pct = 100.0 * total_covered / total_exec
    print(f"{'TOTAL (patch coverage)':<48}{total_covered:>10}{total_exec:>10}{pct:>8.1f}%")
    print(f"\nThreshold: {args.threshold:.1f}%")

    if pct + 1e-9 < args.threshold:
        print(f"FAIL: patch coverage {pct:.1f}% < {args.threshold:.1f}%")
        return 1
    print(f"PASS: patch coverage {pct:.1f}% >= {args.threshold:.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
