# Coverage tooling

Source-based code coverage for the geniex-qairt CPU unit tests, using the
LLVM toolchain bundled with Visual Studio (clang-cl + llvm-cov). MSVC `cl.exe`
has no source-coverage support and OpenCppCoverage does not run on ARM64, so the
coverage build switches the compiler to clang-cl via the CMake option below.

## How it works

The unit-test targets compile the `core/src` translation units directly into
each test executable (rather than linking the `geniex_core` DLL). With
`-DGENIEX_COVERAGE=ON` those targets are built with
`-fprofile-instr-generate -fcoverage-mapping`, so the instrumented region is
exactly the first-party `core/` code under test.

Coverage is scoped to first-party `core/src` and `core/include` sources. The
include/exclude rules live in one place -- [`coverage_common.py`](coverage_common.py)
-- and are shared by the local report and the CI gate so the measured surface
never drifts. `llm_model_test` is excluded from the coverage build: it links
`geniex-proc`, whose tokenizers-cpp Rust chain cannot be built under clang-cl.
Its orchestration logic is still exercised by the MSVC `build-and-test` job.

## Local: full-codebase HTML report

```powershell
pwsh scripts/coverage.ps1 -Open
```

Configures a clang-cl coverage build in `build-coverage/`, builds and runs the
instrumented tests, and renders an HTML report at
`build-coverage/coverage/html/index.html` plus a console summary. Requires the
"C++ Clang tools for Windows" Visual Studio component.

## CI: patch-coverage gate

`.github/workflows/coverage.yml` runs the same build on every PR, then enforces
that **the lines the PR adds or modifies** under `core/` are at least 80%
covered ([`diff_coverage.py`](diff_coverage.py)). Full-file coverage is reported
but not gated -- a small change to a legacy file is not punished, and untested
new code in a well-covered file is not hidden.

```powershell
# What CI runs, minus the build step (reuses build-coverage/coverage/export.json):
python scripts/diff_coverage.py `
    --export-json build-coverage/coverage/export.json `
    --base-ref origin/main --threshold 80
```
