---
name: add-model
description: Add a new LLM model to the geniex runtime (creates spec header, example executable, CMakeLists)
allowed-tools: Read, Edit, Write, Bash, Grep
arguments: [model_name]
---

# Add a new model

Add a new model called `$ARGUMENTS` (or ask the user for the model name if not provided).

## Steps

1. **Create model directory**: `models/<name>/`

2. **Usually no header.** A plain decoder-only model needs none: call
   `geniex::llm_family::makeModel` / `makePipeline` from
   `core/include/pipeline/llm_family.h`, and add a `model_id` prefix check to
   `makeLLMPipeline` in `models/dispatch.h`. `prepend_bos` is the only
   per-family knob — pass it when the model needs a leading BOS its chat
   template does not emit (Qwen3, Gemma; not Llama-3, Qwen2.5, Falcon3, Phi).

   Create `<name>.h` only if the family overrides runtime behaviour (a custom
   `LLMModel` subclass, extra InputProviders).

3. **Create `<name>_example.cpp`** — example executable:
   - Parse command-line arguments
   - Configure `QnnRuntimeConfig` (backend paths)
   - Configure `ModelConfig` (model binary paths, tokenizer)
   - Initialize model with `model.initialize(runtime_cfg, model_cfg)`
   - Run inference loop with `model.generate()`

4. **Create `CMakeLists.txt`**:
   ```cmake
   add_executable(<name> <name>_example.cpp)
   target_link_libraries(<name> PRIVATE geniex_core geniex-proc)
   set_target_properties(<name> PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
   ```

5. **Update root `CMakeLists.txt`**:
   - Add `add_subdirectory(${CMAKE_SOURCE_DIR}/models/<name>)`
   - Add include dir to `geniex_core` target

6. **Verify build**: `cmake --build build --config Release --target <name> -j32`

## LLMSpec structure

`LLMSpec` uses two key fields for shard layout:

- **`.shards`** — vector of `ShardSpec{in_state_name, out_state_name}`, one per shard
- **`.state_blocks`** — vector of `StateBlockSpec`. Use `makeKVOnlyStateBlock(...)` with per-shard `LayerRange{begin, end}` or `std::nullopt` for shards with no KV cache

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

## Choosing InputProvider

| Provider | When to use |
|----------|-------------|
| `TokenIdInputProvider` | Genie/AI Hub exports (on-device embedding, shard 0 takes `input_ids`) |
| `EmbeddingInputProvider` | Custom exports with CPU-side embedding table (needs `model_cfg.embedding_path`) |
| `RoPEInputProvider` | Standard RoPE, no scaling (Qwen3, Falcon3, etc.) |
| `LongRoPEInputProvider` | Long-rope with dynamic scaling + per-dimension `ext_factors` (Phi3.5) |
| `PartialRoPEInputProvider` | Partial-dimension RoPE with `rope_fraction` and `scale` |
| `Llama3RoPEInputProvider` | Llama 3 frequency-dependent scaling (factor=32 for 3.2, factor=8 for 3.1) |

## Common pitfalls

- **Tensor names**: metadata.yaml uses ONNX-style slashes (`/model/model/...`) but QNN graphs may use underscores (`_model_model_...`). Verify at runtime via `graph.inputSpecs()`/`graph.outputSpecs()`.
- **Graph name patterns**: `LLMModel::onInitialized` auto-detects both prefixed (`prompt_arN_clM_S_of_T`, `token_arN_clM_S_of_T`) and unprefixed (`arN_clM_S_of_T`) graph names via regex; nothing to set on `LLMSpec`.
- **Tensor dtypes**: Some exports use float16, others float32 or quantized. `Graph::write(float*)` / `Graph::read(float*)` handle conversion.
- **Linker**: Example executables must link `geniex-proc` explicitly (PRIVATE linkage in geniex_core doesn't propagate).
- **HTP version**: Bundled runtime is QAIRT v2.45.0.260326. Verify model compile version from shard `.json` `buildId` field.