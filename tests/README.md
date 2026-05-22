# Tests

CTest-driven tests for the `geniex` LLM and VLM pipelines. They exercise the
real `LLMPipeline` / `VLMPipeline` on-device

## Build

```pwsh
cmake -B build -A ARM64 `
      -DGENIEX_BUILD_VLM=ON `
      -DGENIEX_BUILD_TESTS=ON
cmake --build build --config Release -j
```

CMake options:

| Option | Default | Purpose |
|---|---|---|
| `GENIEX_BUILD_TESTS` | `OFF` | Enable this subtree. |
| `GENIEX_BUILD_VLM`   | `OFF` | Required for `vlm_test` to build. |

Defaults for `--prompt`, `--max-tokens`, `--min-tokens`, and the VLM image
path live in the `Args` struct at the top of `llm.cpp` / `vlm.cpp`.

## Run

```pwsh
# Everything
ctest --test-dir build -C Release --output-on-failure

# By label
ctest --test-dir build -C Release -L llm            # all LLM tests
ctest --test-dir build -C Release -L vlm            # all VLM tests

# By name
ctest --test-dir build -C Release -R qwen3 -V
```

Exit codes:

| Code | Meaning |
|---|---|
| `0` | Success |
| `1` | Init / runtime error (missing files, QNN init failure, exception) |
| `2` | Assertion failed (fewer than `--min-tokens` tokens generated) |

## Adding a new LLM

1. Drop the standard QAIRT bundle (`config.json`, `metadata.json`, `tokenizer.json`,
   `*.bin`, `htp_backend_ext_config.json`) into `modelfiles/<name>/`. The
   architecture-based dispatcher in [`models/dispatch.h`](../models/dispatch.h)
   reads `config.json`'s `architectures[0]` and routes to the matching family
   factory automatically — no source change required for a variant of an
   already-supported family.
2. Add an entry to `modelFilesTable()` in [`llm.cpp`](llm.cpp) mapping the
   model id to its `modelfiles/` subdirectory and shard filenames (test only).
3. Add the model id to `_LLM_MODELS` in [`CMakeLists.txt`](CMakeLists.txt).
4. If the bundle's architecture is new, add a case to `geniex::makeLLMPipeline`
   in [`models/dispatch.h`](../models/dispatch.h).
