# Contributing to geniex-qairt-plugin

Thanks for your interest in contributing! Whether you're fixing a bug, adding a model, or improving the runtime, this guide takes you from `git clone` to a merged PR. The same rules apply to external and internal contributors — this is the single source of truth.

Please also read our [Code of Conduct](CODE-OF-CONDUCT.md) and [license](LICENSE.txt).

## Getting started

This is a C++20 NPU inference runtime built on Qualcomm AI Runtime (QAIRT). The bundled HTP runtime targets **QAIRT v2.45** and is backward-compatible with models compiled at v2.45 or earlier. No `QNN_SDK_ROOT` is required — QNN headers are vendored under `qnn-api/` and runtime libs under `third-party/`.

Clone with submodules — the build vendors `third-party/geniex-proc` (tokenizer / preprocessing):

```bash
git clone --recursive https://github.com/qualcomm/geniex-qairt-plugin.git
```

Prerequisites and build per platform (CMake ≥ 3.17, Rust for the tokenizer):

- **Windows (native ARM64)** — VS 2022 with the MSVC ARM64 workload, Rust `aarch64-pc-windows-msvc`:

  ```shell
  cmake -B build -A ARM64
  cmake --build build --config Release -j32
  ```

- **Android (cross-compile from Linux/macOS)** — Android NDK r25+, Rust `aarch64-linux-android`:

  ```shell
  export ANDROID_NDK_ROOT=/path/to/android-ndk
  ./build_android.sh            # run with --help for flags (--target, --vlm, --debug)
  ```

- **Linux (native aarch64)** — gcc ≥ 11.2 (matching `third-party/linux-gcc11.2/`), Rust:

  ```shell
  cmake -B build -DCMAKE_BUILD_TYPE=Release
  cmake --build build -j$(nproc)
  ```

Common options: `-DGENIEX_BUILD_VLM=ON` (Vision-Language models), `-DGENIEX_BUILD_EXAMPLES=ON`, `-DGENIEX_BUILD_TESTS=ON`. See [README.md](README.md) for the full build matrix and supported models.

## Running tests

CPU-only unit tests (the QNN boundary is replaced by a link-time stub), GoogleTest via CMake FetchContent, run under CTest. This is what CI runs:

```pwsh
cmake -B build -A ARM64 -DGENIEX_BUILD_TESTS=ON
cmake --build build --config Release -j
ctest --test-dir build -C Release -L unit --output-on-failure
```

Run a single case directly: `.\build\bin\Release\graph_test.exe --gtest_filter=GraphIO.*`. See [tests/README.md](tests/README.md) for adding tests.

**Coverage gate**: PRs must keep patch coverage of modified `core/` lines at **≥ 85%**. Generate a local HTML report with `pwsh scripts/coverage.ps1 -Open` (see [scripts/README.md](scripts/README.md)).

## Project structure

| Directory        | Contents                                                                     |
|------------------|------------------------------------------------------------------------------|
| [core/](core/)   | `geniex_core` framework library (`llm/`, `vlm/`, `pipeline/`, graph/runtime). Never model-specific — see [.claude/rules/engineering-principles.md](.claude/rules/engineering-principles.md). |
| [models/](models/) | Per-model spec headers + example executables (falcon3, llama3, phi3_5, qwen2_5, qwen3, …). |
| [qnn-api/](qnn-api/) | QNN SDK integration layer (vendored QNN headers + wrappers).             |
| [tests/](tests/) | CPU-only GoogleTest unit tests, mirroring `core/src/`.                        |
| [examples/](examples/) | Higher-level examples (`auto_llm/`, `continuous_batching/`).           |
| [scripts/](scripts/) | Coverage tooling.                                                        |
| [docs/](docs/)   | Engineering-principle and xtensor→QNN guides.                                |

## Making a change

Develop on a branch off `main`; PRs target `main`.

1. [Fork](https://github.com/qualcomm/geniex-qairt-plugin/fork) and clone your fork.
2. Create a branch: `git checkout -b <type>/<short-topic> main` (e.g. `feat/sliding-window-eviction`).
3. Add an upstream remote to keep your branch current:

   ```bash
   git remote add upstream https://github.com/qualcomm/geniex-qairt-plugin.git
   git pull --rebase upstream main
   ```

4. Make your change, add tests, keep the change focused — split independent changes into separate PRs.

### Commits — Conventional Commits

Commit messages follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <subject>
```

| Type       | Meaning                                      |
|------------|----------------------------------------------|
| `feat`     | New user-visible feature.                    |
| `fix`      | Bug fix.                                      |
| `perf`     | Performance improvement, no behavior change. |
| `refactor` | Internal restructure, no behavior change.    |
| `docs`     | Documentation only.                          |
| `chore`    | Build, deps, tooling, misc.                  |
| `test`     | Test-only change.                            |
| `ci`       | CI config only.                              |

Scopes seen in this repo include `llm`, `vlm`, `core`. Subject: imperative mood (`add`, not `added`), ≤ 72 characters, no trailing period. Add a body only when the "why" is non-obvious.

### Sign your commits (DCO)

This project uses the [Developer Certificate of Origin](https://developercertificate.org/). Every commit must carry a `Signed-off-by` line — add it with `-s`:

```bash
git commit -s -m "feat(llm): opt-in sliding window context eviction"
```

The DCO check in CI rejects PRs whose commits are not signed off.

## Code style & linting

This project uses **clang-format** (Google base style, 4-space indent, 120-column limit, aligned consecutive assignments/declarations — see [`.clang-format`](.clang-format)). CI's [Lint](.github/workflows/lint.yml) check runs `clang-format-20` on changed C/C++ files, so run it before committing:

```bash
clang-format -i <files>
```

Design rules for `core/` vs `models/` and NPU hardware constraints are documented under [.claude/rules/](.claude/rules/).

## Opening a PR

1. Push your branch to your fork (`git push -u origin <type>/<short-topic>`).
2. [Open a PR](https://github.com/qualcomm/geniex-qairt-plugin/pulls) against `main`.

- **Title**: follow the Conventional Commits format above.
- **Description**: what changed and why.
- **Checks**: [Build and Test](.github/workflows/build-and-test.yml) (compile + `ctest -L unit`), [Lint](.github/workflows/lint.yml) (clang-format), [Coverage](.github/workflows/coverage.yml) (≥ 85% patch coverage), and [QC Preflight Checks](.github/workflows/qcom-preflight-checks.yml) — **Semgrep** static analysis plus DCO, copyright/license, and repolinter. Resolve anything they flag before merge.
- Reviewers are assigned automatically via [CODEOWNERS](CODEOWNERS). It's a good idea to discuss large features or architecture changes in an issue first — reviews go faster with no surprises.

## Reporting issues

Open an issue at [github.com/qualcomm/geniex-qairt-plugin/issues](https://github.com/qualcomm/geniex-qairt-plugin/issues). A good bug report includes a minimal repro, your environment (host OS, SoC / HTP arch, model, QAIRT version), and relevant logs. Feature requests are welcome too — describe the use case.

For **security vulnerabilities**, do **not** open a public issue — follow [SECURITY.md](SECURITY.md).

## Community & Code of Conduct

Participation is governed by our [Code of Conduct](CODE-OF-CONDUCT.md). Be respectful and constructive.
