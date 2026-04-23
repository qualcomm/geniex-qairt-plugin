# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Geniex-QAIRT-plugin: a C++20 inference runtime for generative AI models on Snapdragon NPUs via Qualcomm AI Runtime (QAIRT). It serves as a backend plugin for the `geniex` framework, analogous to what vLLM provides for CUDA but targeting NPU hardware with static graph execution constraints.

## Architecture

> **Important:** Always refer to [`docs/engineering_principle.md`](docs/engineering_principle.md) for the full design philosophy, engineering principles, and guidelines on when to extend `core/` vs. `models/`.
> To understand the overall principals, refer to [`docs/README.md`](docs/README.md). This covers more detail of the HTP hardware properties. 

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

### Key NPU constraints that shape the code

1. **Static graph execution**: all tensor shapes are fixed at compile time. Different sequence/context lengths require separate pre-compiled graph variants.
2. **Prefill = 128-token chunks, decode = 1 token**: long prompts are chunked; short prompts are padded.
3. **Multi-CL promotion**: models with multiple context-length variants start with the smallest sufficient CL and promote upward, reshaping the KV cache.
4. **2GB file limit**: large models are sharded across multiple `.bin` files.
5. **KV cache is pre-allocated** at the max context length with zero-padding for unused positions.

### Key directories

- `core/` — framework library (Model, Graph, LLMModel, LLMPipeline, InputProviders)
- `models/` — per-model specs (`.h`) and example executables (`.cpp`)
- `modelfiles/` — tokenizer configs, embedding tables, HTP configs per model
- `qnn-api/` — QNN SDK headers (`include/QNN/`) and API wrappers (`src/qnn-api/`)
- `third-party/geniex-proc/` — git submodule for tokenizer and preprocessing
- `docs/` — detailed architecture docs, xtensor development workflow guides

## Build Commands

### Windows (primary development platform, ARM64 target)

```shell
# Configure
cmake -B build -A ARM64

# Build all targets
cmake --build build --config Release -j32

# Build a single model target
cmake --build build --config Release --target qwen3_4b -j32
```

Executables output to `build/bin/Release/`.

### Android cross-compilation

```shell
export ANDROID_NDK_ROOT=/path/to/android-ndk
./build_android.sh              # defaults: arm64-v8a, Release
./build_android.sh --abi x86_64 --debug
```

### CMake options

| Option | Default | Description |
|--------|---------|-------------|
| `GENIEX_BUILD_VLM` | OFF | Vision-Language Models (requires OpenCV) |
| `BUILD_EXAMPLES` | ON | Per-model example executables |

## Code Formatting

`.clang-format`: Google style, 4-space indent, 120 column limit.

## Adding a new model

Each model lives in `models/<name>/` with three files:

1. **`<name>.h`** — header-only: define `makeSpec()` returning an `LLMSpec` and `makeModel()` returning an `LLMModel` with appropriate `InputProvider`s. For non-standard behavior, subclass `LLMModel`. If a new model shares the exact same architecture as an existing one (e.g. a fine-tune), reuse the existing `makeModel()` — only a new example `.cpp` with different paths is needed.
2. **`<name>_example.cpp`** — example executable: parse args, configure `QnnRuntimeConfig` + `ModelConfig`, initialize, run generate loop.
3. **`CMakeLists.txt`** — `add_executable` + link to `geniex_core geniex-proc`, add as `add_subdirectory` in root CMakeLists.txt. Also add include dir to `geniex_core` target in root CMakeLists.txt.

Model files (tokenizer, embeddings, HTP config, `.bin` shards) go in `modelfiles/<name>/`.

#### LLMSpec structure

`LLMSpec` uses two key fields to describe the model's shard layout:

- **`.shards`** — vector of `ShardSpec{in_state_name, out_state_name}`, one per shard, describing inter-shard hidden state wiring.
- **`.state_blocks`** — vector of `StateBlockSpec` (typically one KV block). Use `makeKVOnlyStateBlock(...)` with a per-shard vector of `LayerRange{begin, end}` or `std::nullopt` for shards with no KV cache (e.g. embedding-only shards).

Example (3-shard model with embedding shard + 2 KV shards):
```cpp
.shards = {
    {"input_ids", "_model_model_embed_tokens_Gather_output_0"},
    {"_model_model_embed_tokens_Gather_output_0", "_model_model_layers_7_Add_1_output_0"},
    {"_model_model_layers_7_Add_1_output_0", "logits"},
},
.state_blocks = {
    makeKVOnlyStateBlock({std::nullopt, LayerRange{0, 7}, LayerRange{8, 15}}),
},
```

#### Choosing the right InputProvider

- **`TokenIdInputProvider`** — for Genie/AI Hub exports where embedding runs on-device (shard 0 takes `input_ids`).
- **`EmbeddingInputProvider`** — for custom exports with CPU-side embedding table (requires `model_cfg.embedding_path`).
- **`RoPEInputProvider`** — standard RoPE (no scaling). Used by Qwen3, Falcon3, etc.
- **`LongRoPEInputProvider`** — long-rope with dynamic scaling and per-dimension extension factors. Caller passes `ext_factors` as a `std::vector<float>`. Used by Phi3.5.
- **`PartialRoPEInputProvider`** — partial-dimension RoPE with post-scale. Caller passes `rope_fraction` and `scale`.
- **`Llama3RoPEInputProvider`** — Llama 3 frequency-dependent RoPE scaling. Used by Llama 3.1/3.2. Pay attention to the `factor` parameter: Llama 3.2 uses factor=32, Llama 3.1 uses factor=8.

### Pitfalls when adding new models

- **Tensor names**: The metadata.yaml in modelfiles uses ONNX-style names with slashes (e.g. `/model/model/embed_tokens/Gather_output_0`) but the actual QNN graph tensors may use underscores instead (e.g. `_model_model_embed_tokens_Gather_output_0`). Always verify actual tensor names by dumping `graph.inputSpecs()`/`graph.outputSpecs()` at runtime.
- **Graph name patterns**: In practice, all Genie-exported models seen so far use `prompt_`/`token_` prefixes requiring `graph_name_pattern = "{phase}_ar{ar}_cl{cl}_{shard}_of_{total}"` — even when the metadata.yaml omits the prefix. Always set this pattern for Genie exports.
- **Tensor dtypes**: Some model exports use float16 tensors (e.g. Genie w4 exports), others use float32 or quantized types. `Graph::write(float*)` and `Graph::read(float*)` handle float32, float16, ufixed8/16, and int32.
- **Linker**: Example executables must link `geniex-proc` in addition to `geniex_core` because `geniex_core` links `geniex-proc` as PRIVATE, so the tokenizer symbols don't propagate transitively.
- **HTP version**: The bundled runtime libs in `third-party/` are QAIRT v2.45.0.260326 (`windows`, `android`, `linux-gcc11.2`). Runtime version is backward compatible with compile version, so models compiled with an older QAIRT (e.g. v2.42) will run fine. When adding a new model, verify its compile version from the `buildId` field in the shard `.json` config files and ensure the `dsp_arch` in `htp_backend_ext_config.json` matches the target `third-party/` folder (e.g. `v73` → `windows`).

### Development workflow: xtensor prototyping

For complex tensor logic, prototype with xtensor (NumPy-like C++ API) then convert to direct QNN buffer operations for production. See `docs/xtensor-prompt.md` and `docs/xtensor2qnn.md`. Xtensor utilities are in `core/src/llm/xtensor_utils.cpp` (not compiled into geniex_core by default to avoid the dependency).
