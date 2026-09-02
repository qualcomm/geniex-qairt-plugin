<h1 align="center">GenieX-QAIRT-plugin</h1>

<p align="center">A framework for running generative AI models on Snapdragon NPUs.</p>

## What is QAIRT?

**Qualcomm AI Runtime (QAIRT)** is a suite of tools for developing, running, and optimizing AI models on Qualcomm hardware. It has the best hardware-aware design to get the metal performance.

This plugin, as one of the backends of `geniex`, uses QAIRT to support various generative AI models.

## Installation

Executables and `geniex_core` (shared library) are placed under the build tree; see each platform below. The HTP runtime libs are copied to `<build>/bin/htp-files/` automatically.

### Common CMake options

| Option | Default | Description |
|---|---|---|
| `GENIEX_BUILD_VLM` | `OFF` | Build Vision-Language models (e.g. Qwen2.5-VL). |
| `GENIEX_BUILD_EXAMPLES` | `OFF` | Build per-model example executables. |
| `GENIEX_BUILD_TESTS` | `OFF` | Register CTest entries for LLM/VLM pipeline tests. Requires a Snapdragon NPU host. See [`tests/README.md`](tests/README.md). |
| `GENIEX_DEBUG` | `OFF` | Verbose logging with file/line/func info. |

### Windows (native ARM64)

Prerequisites: Visual Studio 2022 with the MSVC ARM64 workload, CMake ≥ 3.17, Rust (with `aarch64-pc-windows-msvc` target — needed for the tokenizer).

```shell
# Configure
cmake -B build -A ARM64

# Build everything
cmake --build build --config Release -j32

# Build a specific model target
cmake --build build --config Release --target qwen3_4b -j32

# VLM build (Qwen2.5-VL)
cmake -B build -A ARM64 -DGENIEX_BUILD_VLM=ON
cmake --build build --config Release -j32
```

Output: `build/bin/Release/*.exe` and `geniex_core.dll`.

### Android (cross-compile from Linux/macOS)

Prerequisites: [Android NDK](https://developer.android.com/ndk/downloads) (r25+ recommended), CMake, Rust with `aarch64-linux-android` target (`rustup target add aarch64-linux-android`).

```shell
export ANDROID_NDK_ROOT=/path/to/android-ndk
./build_android.sh                                 # arm64-v8a Release, all examples
./build_android.sh --target qwen3_4b               # build a single target
./build_android.sh --vlm --target qwen2_5_vl_7b    # enable GENIEX_BUILD_VLM
./build_android.sh --debug --debug-log             # Debug + verbose logging
./build_android.sh --help                          # full flag list
```

Output: `build-android/bin/*` (no extension) and `libgeniex_core.so`.

> The script auto-detects the NDK host tag (`linux-x86_64` vs `darwin-x86_64`). It does not support building on Windows hosts.

### Linux (native aarch64)

Prerequisites: gcc ≥ 11.2 (matching the bundled runtime in `third-party/linux-gcc11.2/`), CMake, Rust.

```shell
# Configure
cmake -B build -DCMAKE_BUILD_TYPE=Release

# Build
cmake --build build -j$(nproc)

# Build a specific model target
cmake --build build --target qwen3_4b -j$(nproc)
```

Output: `build/bin/*` and `libgeniex_core.so`.

## Supported Hardware

| Hardware | SoC | HTP Arch | SoC Model |
|----------|-----|----------|-----------|
| Snapdragon X Elite / Plus | SC8380 | v73 | 60 |
| IQ-9075 | QCS9075 | v73 | 60 |
| Snapdragon 8 Elite | SM8750 | v79 | 69 |
| Snapdragon 8 Elite Gen5 | SM8850 | v81 | 88 |

> The bundled HTP runtime libs in `third-party/` (`windows`, `android`, `linux-gcc11.2`) are QAIRT **v2.45.0.260326** (single source of truth: `GENIEX_QAIRT_VERSION` in [`core/include/version.h`](core/include/version.h); consumers read it at runtime via `geniex_qairt_version()`). Runtime version is backward compatible with compile version, so all models compiled with v2.45 or earlier will run correctly.
>
> That is the version of the *libs*. What decides whether a runtime loads is the C API in `qnn-api/include/` — see [Using a different QAIRT runtime](#using-a-different-qairt-runtime).

## Using a different QAIRT runtime

The build bundles the QAIRT runtime above and copies it to `htp-files/` next to `geniex_core`, so **the default path needs no configuration** — no SDK download, no paths to set.

To run against a different QAIRT version instead, set `GENIEX_QNN_LIB` to a directory holding the runtime libraries:

```shell
# Windows
set GENIEX_QNN_LIB=C:\path\to\qairt-libs
# Linux / Android
export GENIEX_QNN_LIB=/path/to/qairt-libs
```

One build drives many runtimes: the plugin reaches QNN only through the versioned C interface, which negotiates at load time. There is a single header set in `qnn-api/include/`, deliberately the lowest we support — newer headers would narrow the accepted range, not widen it.

What sets the floor is the **C API version** (`GENIEX_QNN_API_VERSION`, 2.27), not the bundled-lib release (`GENIEX_QAIRT_VERSION`, 2.45):

| QAIRT SDK | QNN C API | Loads? |
|-----------|-----------|--------|
| 2.36 (what we compile against) | 2.27 | ✅ floor |
| 2.45 (bundled) | 2.34 | ✅ verified |
| 2.48 | 2.37 | ✅ verified |
| 2.49 | 2.38 | ✅ verified |
| older than 2.36 | < 2.27 | ❌ rejected at load |

Entry points added after C API 2.27 are not callable from this build.

**Expected directory shape.** Either layout works. A *flat* folder holding the host libraries and their arch stubs together — the same shape as the bundled `htp-files/`:

```
qairt-libs/
├── QnnHtp.dll                     (libQnnHtp.so)
├── QnnSystem.dll                  (libQnnSystem.so)
├── QnnHtpNetRunExtensions.dll     (libQnnHtpNetRunExtensions.so)
└── QnnHtpV73Stub.dll, ...         arch stubs and skels
```

…or a **stock QAIRT SDK root**, as unpacked from the Qualcomm Software Center:

```
qairt/2.XX.0/
└── lib/
    ├── aarch64-windows-msvc/      host libraries (or aarch64-android,
    │                              aarch64-oe-linux-gcc11.2)
    └── hexagon-v73/unsigned/      skels, one folder per arch
        hexagon-v81/unsigned/
```

Host libraries are taken from `lib/<target-triple>/`, and every `lib/hexagon-v*/` folder goes on `ADSP_LIBRARY_PATH` so FastRPC matches the device's arch. An unrecognised triple is found by scanning `lib/`, so a renamed one (the Linux gcc suffix moves between releases) still resolves. The INFO log names the folder the host libraries actually came from, which for an SDK root is not the path you passed.

Resolution order, highest precedence first:

| Rung | Source |
|------|--------|
| 1 | `QnnRuntimeConfig::backend_path` / `system_lib_path` / `extensions_path`, when all three are set |
| 2 | `QnnRuntimeConfig::htp_dir` |
| 3 | `GENIEX_QNN_LIB` |
| 4 | bundled `htp-files/` next to `geniex_core` |

The chosen directory and the rung it came from are logged at INFO. Check that line before trusting a run against a non-bundled runtime: a mismatched runtime can load and generate at full speed while producing wrong output, so confirming *which* libraries loaded is the only reliable check.

## Project Structure

```
├── models/              # Model specs (.h) and example executables (.cpp)
│   ├── falcon3/
│   ├── llama3/
│   ├── llama3_1/
│   ├── llama3_2/
│   ├── llama3_2_ssd/
│   ├── phi3_5/
│   ├── qwen2_5/
│   ├── qwen2_5_vl/        # VLM (requires GENIEX_BUILD_VLM=ON)
│   └── qwen3/
├── core/                # geniex_core framework (LLM model, graph, KV cache, RoPE)
├── modelfiles/          # Tokenizer and config files per model
├── qnn-api/             # QNN SDK integration layer (headers + API wrappers)
├── third-party/         # HTP runtime libs + geniex-proc submodule (tokenizer, preprocessing)
└── docs/                # Documentation
```

## Getting in Contact

- [Report an Issue on GitHub](../../issues)
- [Open a Discussion on GitHub](../../discussions)

For security-sensitive reports, see [SECURITY.md](SECURITY.md).

## Contributing

Contributions are welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md)
for the branching model, pull-request workflow, and DCO sign-off
requirement, and [CODE-OF-CONDUCT.md](CODE-OF-CONDUCT.md) for community
expectations.

## License

GenieX-QAIRT-plugin is licensed under the
[BSD 3-Clause License](https://spdx.org/licenses/BSD-3-Clause.html). See
[LICENSE.txt](LICENSE.txt) for the full license text.

This project also ships vendored third-party components (the QAIRT SDK
files under `qnn-api/` and the prebuilt runtime libraries under
`third-party/`) that are governed by separate licenses. See
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for details.