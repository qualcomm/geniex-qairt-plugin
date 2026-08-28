# CLAUDE.md

## Project Overview

GenieX-QAIRT-plugin: a C++20 inference runtime for generative AI models on Snapdragon NPUs via Qualcomm AI Runtime (QAIRT). Backend plugin for the `geniex` framework, analogous to what vLLM provides for CUDA but targeting NPU hardware with static graph execution constraints.

## Architecture

### Library tiers

- **geniex_core** (always built): QNN API wrappers + core framework + LLM inference engine + tokenizer (via geniex-proc submodule).
- **geniex_vlm** (optional): Vision/audio encoder integration, multimodal embedding injection.

### Class hierarchy

```
Model (base)                    — QNN backend init, graph loading, inter-graph connections
  LLMModel                      — prefill/decode loop, KV cache, multi-shard execution
    VLMModel                    — multimodal embedding injection

InputProvider (interface)       — CPU-side tensor preparation before NPU execution
  EmbeddingInputProvider        — token ID -> embedding lookup (CPU-side table)
  TokenIdInputProvider          — pass-through for on-device embedding (AI Hub/Genie models)
  RoPEInputProvider             — standard rotary position encoding
  LongRoPEInputProvider         — long-rope with dynamic scaling and per-dimension ext_factors
  PartialRoPEInputProvider      — partial-dimension RoPE with post-scale factor
  Llama3RoPEInputProvider       — llama3 frequency-dependent RoPE scaling
  PrecomputedEmbeddingProvider  — VLM: switches prefill input to precomputed embeddings
  MRoPEInputProvider            — VLM: multi-dimensional positional inputs

VisionEncoder / AudioEncoder    — abstract modality encoder interfaces
LLMPipeline                     — high-level API: tokenizer + chat template + streaming generation
```

### Key directories

- `core/` — framework library (Model, Graph, LLMModel, LLMPipeline, InputProviders)
- `models/` — example executables (`.cpp`); a spec header (`.h`) only for families that override runtime behaviour
- `modelfiles/` — tokenizer configs, embedding tables, HTP configs per model
- `qnn-api/` — QNN SDK headers (`include/QNN/`) and API wrappers (`src/qnn-api/`)
- `third-party/geniex-proc/` — git submodule for tokenizer and preprocessing

## Build Commands

### Windows (primary development platform, ARM64 target)

```shell
cmake -B build -A ARM64
cmake --build build --config Release -j32
cmake --build build --config Release --target qwen3_4b -j32
```

### Android cross-compilation

```shell
export ANDROID_NDK_ROOT=/path/to/android-ndk
./build_android.sh
./build_android.sh --abi x86_64 --debug
```

### CMake options

| Option | Default | Description |
|--------|---------|-------------|
| `GENIEX_BUILD_VLM` | OFF | Vision-Language Models (requires OpenCV) |
| `BUILD_EXAMPLES` | ON | Per-model example executables |

## Code Formatting

`.clang-format`: Google style, 4-space indent, 120 column limit.

## Skills (slash commands)

- `/add-model <name>` — scaffold a new model (spec header, example, CMakeLists)
- `/write-xtensor` — write tensor logic using xtensor (NumPy-like C++ API) for prototyping
- `/convert-xtensor-to-qnn` — convert xtensor prototype to production QNN direct-buffer operations
- `/dev-qairt-runtime` — reference for QNN graph execution, KV cache, tensor I/O, multi-shard wiring. Used when we write core runtime related logic.
