# Coverage tooling

Source-based code coverage for the geniex-qairt CPU unit tests, using the
LLVM toolchain bundled with Visual Studio (clang-cl + llvm-cov). MSVC `cl.exe`
has no source-coverage support and OpenCppCoverage does not run on ARM64, so the
coverage build switches the compiler to clang-cl via the CMake option below.

## Requirements

The **"C++ Clang tools for Windows"** Visual Studio component (provides
`clang-cl`, `llvm-cov`, `llvm-profdata` under `VC\Tools\Llvm\ARM64\bin`). It is
a checkbox in the VS Installer -- no separate download or PATH setup. Your
normal MSVC build of geniex-qairt is unaffected: coverage uses a separate
`build-coverage/` dir and only switches the compiler there.

## How it works

The unit-test targets compile the `core/src` translation units directly into
each test executable (rather than linking the `geniex_core` DLL). With
`-DGENIEX_COVERAGE=ON` those targets are built with
`-fprofile-instr-generate -fcoverage-mapping`, so the instrumented region is
exactly the first-party `core/` code under test.

Coverage is scoped to first-party `core/src` and `core/include` sources. The
include/exclude rules live in one place -- [`coverage_common.py`](coverage_common.py)
-- and are shared by the local report and the CI gate so the measured surface
never drifts. All unit-test exes are instrumented; the whole tree builds under
clang-cl once `CC`/`CXX` point at it (which `coverage.ps1` and the CI jobs set),
so the `build-and-test` gate and coverage use the same compiler.

Device-only files are excluded from the measured surface because they cannot run
without an NPU and are covered by on-device integration tests instead:
`runtime.cpp` (QNN/FastRPC HTP bring-up), `threadpool.cpp` (worker threads + CPU
affinity), `vlm/vision_encoder.cpp` (QNN graph encode), and the `logging` shim.
`model.cpp` stays measured -- its accessors and `applyConnections` are unit-
tested, while `Model::initialize()` (QNN backend load) is unreachable on CPU and
shows as uncovered.

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
that **the lines the PR adds or modifies** under `core/` are at least 85%
covered ([`diff_coverage.py`](diff_coverage.py)). Full-file coverage is reported
but not gated -- a small change to a legacy file is not punished, and untested
new code in a well-covered file is not hidden.

```powershell
# What CI runs, minus the build step (reuses build-coverage/coverage/export.json):
python scripts/diff_coverage.py `
    --export-json build-coverage/coverage/export.json `
    --base-ref origin/main --threshold 85
```
