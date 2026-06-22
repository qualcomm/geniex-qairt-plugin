#!/usr/bin/env python3
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
"""Enforce a coverage threshold on the lines a PR adds or modifies.

Full-file coverage is the wrong gate for a PR: a small change to a poorly
covered legacy file would fail, and a large change to a well-covered file could
hide untested new code. Instead we measure only the *added/modified* executable
lines in the diff -- "patch coverage" -- which is the line-level analogue of the
Go side's coverage skill.

Pipeline:
  1. Parse `git diff <base>...HEAD` to collect added/modified line numbers per
     first-party source file.
  2. Parse `llvm-cov export` JSON to learn which of those lines are executable
     and whether each was hit.
  3. Fail if (covered changed lines) / (executable changed lines) < threshold.

stdlib only; no third-party packages on the runner.
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
    """Map each changed first-party file -> set of added/modified line numbers
    (line numbers in the NEW file), parsed from unified diff hunk headers."""
    diff = subprocess.run(
        ["git", "diff", "--unified=0", "--no-color", f"{base_ref}...HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout

    result: dict[str, set[int]] = defaultdict(set)
    current: str | None = None
    new_lineno = 0
    # +++ b/path  marks the new file; @@ -a,b +c,d @@ gives the new-side range.
    hunk_re = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@")

    for line in diff.splitlines():
        if line.startswith("+++ "):
            path = line[4:].strip()
            path = path[2:] if path.startswith("b/") else path  # strip b/
            current = path if (path != "/dev/null" and is_covered_source(path)) else None
        elif line.startswith("@@"):
            m = hunk_re.match(line)
            new_lineno = int(m.group(1)) if m else 0
        elif current is not None:
            if line.startswith("+"):
                result[current].add(new_lineno)
                new_lineno += 1
            elif line.startswith("-"):
                pass  # deletions don't advance the new-file counter
            else:
                new_lineno += 1
    return result


def covered_lines(export_json: dict) -> dict[str, dict[int, bool]]:
    """Map each first-party file -> {line: was_executed} for executable lines.

    Walks llvm-cov segments: each segment [line, col, count, hasCount, ...]
    opens a region with an execution count; a line is executable if any region
    covers it, and executed if any covering region's count > 0."""
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
                # A region starting on this line makes it executable; OR the
                # hit state so any covering region with count>0 marks it run.
                line_hit[line] = line_hit.get(line, False) or count > 0
            # Keep only the basename-relative key for matching against the diff.
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
    ap.add_argument("--threshold", type=float, default=80.0, help="min percent (default 80)")
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
            # Changed source not present in coverage data: either it isn't
            # compiled into the instrumented tests, or it has no executable
            # lines in the diff. Skip rather than punish.
            continue
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
