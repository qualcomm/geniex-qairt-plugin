<h1 align="center">Geniex-QAIRT-plugin</h1>

<p align="center">A framework for running generative AI models on Snapdragon NPUs.</p>

## What is QAIRT?

**Qualcomm AI Runtime (QAIRT)** is a suite of tools for developing, running, and optimizing AI models on Qualcomm hardware. It has the best hardware-aware design to get the metal performance. 

This plugin, as one of the backend of the `geniex`, uses the QAIRT as the toolkits to support various generative AI models. 

## Installation

Executables and `geniex_core` (shared library) are placed under the build tree; see each platform below. The HTP runtime libs are copied to `<build>/bin/htp-files/` automatically.

### Common CMake options

| Option | Default | Description |
|---|---|---|
| `GENIEX_BUILD_VLM` | `OFF` | Build Vision-Language models (e.g. Qwen2.5-VL). |
| `GENIEX_DEBUG` | `OFF` | Verbose logging with file/line/func info. |
| `BUILD_EXAMPLES` | `ON` | Build per-model example executables. |

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

## Supported Models

The following models from [Qualcomm AI Hub](https://aihub.qualcomm.com/compute/models?domain=Generative+AI&useCase=Text+Generation&runtime=genie) are supported. All models below run on **Snapdragon X Elite / Plus**.

| Model | Target | Compile version | Source |
|-------|--------|-------|--------|
| Phi 3.5 Mini | `phi3_5` | v2.43 | [HuggingFace](https://huggingface.co/yichqian/geniex-qairt-models/tree/main/phi3_5_aihub) |
| Qwen3 4B | `qwen3_4b` | v2.42 | [HuggingFace](https://huggingface.co/yichqian/geniex-qairt-models/tree/main/qwen3_4b_aihub) |
| Qwen3 4B Instruct 2507 | `qwen3_4b_instruct_2507` | v2.42 | [HuggingFace](https://huggingface.co/yichqian/geniex-qairt-models/tree/main/qwen3_4b_instruct_2507_aihub) |
| Qwen2.5 7B Instruct | `qwen2_5_7b_instruct` | v2.42 | - |
| Llama-v3-8B-Instruct | `llama_v3_8b_instruct` | v2.42 | [HuggingFace](https://huggingface.co/yichqian/geniex-qairt-models/tree/main/llama_v3_8b_instruct_aihub) |
| Llama-v3-ELYZA-JP-8B | `llama_v3_elyza_jp_8b` | v2.42 | [HuggingFace](https://huggingface.co/yichqian/geniex-qairt-models/tree/main/llama_v3_elyza_jp_8b_aihub) |
| Llama3-TAIDE-LX-8B-Chat | `llama_v3_taide_8b_chat` | v2.42 | [HuggingFace](https://huggingface.co/yichqian/geniex-qairt-models/tree/main/llama_v3_taide_8b_chat_aihub) |
| Llama-v3.2-1B-Instruct | `llama3_2_1b` | v2.42 | - |
| Llama-v3.2-3B-Instruct | `llama3_2_3b` | v2.42 | - |
| Falcon3-7B-Instruct | `falcon3_7b` | v2.42 | - |
| Llama-3.1-8B-Instruct | `llama3_1_8b` | v2.42 | - |
| Llama-SEA-LION-v3.5-8B-R | `sea_lion_8b` | v2.42 | - |
| Llama-v3.2-3B-Instruct-SSD | `llama3_2_3b_ssd` | v2.42 | [HuggingFace](https://huggingface.co/yichqian/geniex-qairt-models/tree/main/llama_v3_2_3b_instruct_ssd) |
| Qwen2.5-VL-7B-Instruct | `qwen2_5_vl_7b` | v2.45 | [HuggingFace](https://huggingface.co/yichqian/geniex-qairt-models/tree/main/qwen2_5_vl_7b) |

> The bundled HTP runtime libs in `third-party/` (`windows`, `android`, `linux-gcc11.2`) are QAIRT **v2.45.0.260326**. Runtime version is backward compatible with compile version, so all models compiled with v2.45 or earlier will run correctly.

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