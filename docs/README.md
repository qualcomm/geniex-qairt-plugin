# GenieX Runtime: QNN Backend Documentation

This document provides comprehensive documentation for the GenieX QNN runtime. If you're familiar with frameworks like vLLM or SGLang for GPU-based LLM inference, this guide will help you understand the architectural decisions and constraints unique to Qualcomm AI Runtime (QAIRT).

## Table of Contents

- [Overview](#overview)
- [Glossary](#glossary)
- [Key Concepts for PyTorch Users](#key-concepts-for-pytorch-users)
- [QAIRT Features](#qairt-features)
- [LLM Inference Under Constraints](#llm-inference-under-constraints)
- [Repository Architecture](#repository-architecture)
- [Class Hierarchy](#class-hierarchy)
- [How to Add a New Model](#how-to-add-a-new-model)
- [Performance Considerations](#performance-considerations)
- [Development Workflow: xtensor to QNN](#development-workflow-xtensor-to-qnn)

---

## Overview

This repository serves as the **inference runtime for generative AI models on Snapdragon NPUs**, analogous to what vLLM or SGLang provides for CUDA-based deployments. However, due to the unique characteristics of the QAIRT backend, we operate under a different set of constraints that require careful consideration of static graph execution, KV cache management, and memory limitations.

The core philosophy: we achieve high-performance inference while respecting the rigid constraints of the NPU execution model.

---

## Glossary

| Term | Definition |
|------|------------|
| **Context Binary** | Pre-compiled, optimized graph in QAIRT format (`.bin` file). Contains model weights and operations for specific input shapes. |
| **Graph** | A single executable unit within a context binary, representing one `(sequence_length, context_length)` configuration. |
| **Shard** | A portion of a large model split across multiple context binary files to respect the 2GB file size limit. |
| **Context Length (CL)** | Maximum number of tokens the KV cache can hold. Determines the model's "memory" for conversation history. |
| **Sequence Length / AR Length** | Number of tokens processed in a single forward pass. 128 for prefill, 1 for decode. |
| **Prefill** | Initial phase that processes the input prompt, populating the KV cache. |
| **Decode** | Autoregressive phase that generates one token at a time using the KV cache. |
| **KV Cache** | Key-Value cache storing attention states from previous tokens, enabling efficient autoregressive generation. |
| **VTCM** | Vector Tightly Coupled Memory - fast on-chip memory on Snapdragon NPU, analogous to GPU shared memory. |
| **HTP** | Hexagon Tensor Processor - the NPU compute unit on Snapdragon SoCs. |
| **QAIRT** | Qualcomm AI Runtime - the SDK for deploying AI models on Qualcomm hardware. |
| **Input Provider** | CPU-side component that prepares input tensors before NPU execution (embeddings, RoPE, etc.). |
| **RoPE** | Rotary Position Embedding - position encoding method used by modern LLMs. |
| **Multi-CL** | Multi-Context-Length support - using multiple graph variants optimized for different context lengths. |
| **Attention Mask** | Binary or float mask that controls which positions attend to which in self-attention. |
| **TensorSpec** | Metadata describing a tensor (name, dtype, shape, quantization parameters). |
| **Connection** | Specification for routing one graph's output to another graph's input. |
| **TTFT** | Time To First Token - latency from prompt submission to first generated token. |

---

## Key Concepts for PyTorch Users

### From Dynamic to Static Execution

| PyTorch/vLLM | GenieX/QAIRT |
|--------------|--------------|
| Dynamic shapes at runtime | Fixed tensor shapes at compile time |
| In-place tensor operations | All operations pre-compiled into static graphs |
| Single model file | Multiple context binaries (one per shape configuration) |
| Flexible batch size | Fixed sequence length chunks |
| Dynamic KV cache growth | Pre-allocated KV buffers with fixed context lengths |


---

## QAIRT Features

Understanding these features is essential for working with this codebase:

### 1. Static Graph Execution

Every compiled graph has **fixed tensor shapes**. You cannot:
- Modify tensor shapes at runtime
- Use dynamic control flow within a graph
- Resize intermediate activations

**Implication**: For different input configurations (e.g., prefill vs. decode, or different context lengths), you need separate pre-compiled graphs. The runtime chains these graphs together.

### 2. Weight Sharing Across Graphs

While graphs must be static, weights can be **shared across multiple graph variants**. For example:
- A model supporting both 128-token prefill and 1-token decode shares the same weight tensors
- Models with multiple context length variants (512, 1024, 2048, 4096) share weights but have different activation buffers

This is similar to how you might use the same `nn.Module` weights with different input shapes in PyTorch, but here it's explicit at the binary level.

### 3. VTCM Size Limits

The **Vector Tightly Coupled Memory (VTCM)** is the NPU's fast on-chip memory (similar to GPU shared memory). Each intermediate tensor must fit within VTCM constraints, which limits:
- Maximum sequence length per graph (typically 128 for prefill)
- Maximum batch size
- Activation tensor sizes

### 4. File Size Limit (2GB)

Each context binary file cannot exceed 2GB. Large models (7B+ parameters) must be **sharded** into multiple `.bin` files, each containing a subset of layers.

### 5. HTP Backend Config and Multicore Execution

Each bundle may ship an `htp_backend_ext_config.json` next to its context binaries. The file is parsed by QAIRT's closed-source `QnnHtpNetRunExtensions` library at backend init and configures the HTP device:

```json
{
  "devices": [
    {
      "soc_model": 77,
      "dsp_arch": "v73",
      "cores": [
        { "core_id": 0, "perf_profile": "burst", "rpc_control_latency": 100 }
      ]
    }
  ],
  "memory": { "mem_type": "shared_buffer" },
  "context": { "weight_sharing_enabled": true }
}
```

| Key | Meaning |
|-----|---------|
| `devices[].soc_model` | Numeric SoC model the config targets (e.g. 77 = Snapdragon X Elite family). |
| `devices[].dsp_arch` | Hexagon DSP architecture the binaries were compiled for (`v68`/`v73`/`v75`/`v79`...). Must match the device. |
| `devices[].cores[]` | One entry per NSP core to configure. `core_id` selects the core; `perf_profile` (e.g. `burst`) and `rpc_control_latency` (µs) tune it. |
| `memory.mem_type` | Buffer allocation strategy; `shared_buffer` enables zero-copy CPU↔NPU I/O. |
| `context.weight_sharing_enabled` | Share weights across graphs in a context (see feature 2). |

**Multicore.** `geniex_core` reads the `cores` list back (`parseHtpCoreCount`) and, when it names more than one core, requests multicore graph execution by setting `QNN_HTP_GRAPH_CONFIG_OPTION_NUM_CORES` on every loaded graph. The knob is `ModelConfig::num_cores`: `0` (default) derives the count from `htp_config_path`'s JSON at init — so both `modelConfigFromDirectory` bundles and hand-built configs (example executables) pick it up — `1` forces single core, and any explicit value overrides the JSON. At init the runtime logs the device-reported core count and the effective core count, warns and clamps when the JSON requests more cores than the device exposes, and warns (without failing) when the backend rejects the config.

**Context-length variant selection.** A bundle that ships several context-length variants (`..._cl512_...`, `..._cl4096_...`) pays for all of them at load: HTP reserves persistent per-graph I/O for every graph it deserializes, even though only one context length is active at a time. On a 4-shard 3.0 GB bundle with five variants that is ~1.7 GB of I/O on top of ~2.9 GB of weights, which overruns the ~4 GB protection domain and fails the last shard.

The knob is `ModelConfig::context_lengths`, and like `num_cores` it derives itself. Empty (default) loads every variant; only if context creation is refused does the runtime retry with fewer, dropping the middle variants first, then the smallest, and never the largest — so the model's usable context is unchanged, and only short-prompt speed is traded away as promotion steps disappear. Setting the field explicitly pins the selection and disables back-off, which is worth doing once the log has told you which set the device accepted, because each failed attempt costs a full deserialize. Requested lengths the bundle does not provide are reported and ignored; a request matching nothing fails init and lists what the bundle has.

There is no pre-flight check available for this: no QNN API reports free protection-domain space (`QNN_HTP_CONTEXT_CONFIG_OPTION_IO_MEM_ESTIMATION` performs the fit check internally and reports nothing back), so the selection is necessarily empirical. Back-off is implemented on the synchronous create-from-binary path, which is the one `geniex_core` uses; requesting async init logs that back-off is unavailable. The `LLMSpec::context_lengths` field is unrelated to this knob — the runtime overwrites it at init from the graphs actually loaded.

What is fixed at context-binary **generation** time versus settable at **load** time:

- *Generation time*: graph structure, tensor shapes, and any multicore graph partitioning baked in by the compiler. Binaries compiled without a multicore graph config may not benefit from (or accept) a higher core count at load time.
- *Load time*: the `NUM_CORES` graph config, per-core perf profiles, and RPC latency from the JSON. `geniex_core` applies `NUM_CORES` after context creation and before the first execute — the only window QNN allows for graphs retrieved from a prebuilt binary.

A QNN verbose trace of `Error code 1000 returned for property query with key = 304` followed by `Multicore support is unavailable` means the DSP skel does not answer the multicore property query — the device/driver combination cannot run this workload across multiple NSP cores regardless of the JSON contents. Most current Snapdragon mobile/compute SoCs expose a single NSP core to QNN; multicore is limited to SoCs (e.g. certain automotive/IoT parts) whose platform info reports `numCores > 1`, which the runtime now logs at init.

---

## LLM Inference Under Constraints

### Prefill and Decode Phases

Like GPU-based frameworks, we distinguish between:
- **Prefill**: Process the input prompt (sequence_length > 1)
- **Decode**: Generate tokens one at a time (sequence_length = 1)

However, due to static graphs:
- Prefill uses a fixed chunk size (typically 128 tokens)
- Decode always processes exactly 1 token

### Handling Variable-Length Inputs

**Long prompts (> 128 tokens)**: Chunked into multiple prefill iterations
```
Prompt: 300 tokens
  → Prefill chunk 1: tokens 0-127   (128 tokens)
  → Prefill chunk 2: tokens 128-255 (128 tokens)
  → Prefill chunk 3: tokens 256-299 (44 tokens, padded to 128)
```

**Short prompts (< 128 tokens)**: Padded to the fixed sequence length
```
Prompt: 50 tokens
  → Prefill: tokens 0-49 + 78 padding tokens (128 total)
```

### KV Cache Management

The KV cache is the **central state** of LLM inference. Key considerations:

1. **Pre-allocated buffers**: KV cache size is fixed at the maximum context length (e.g., 4096)
2. **Zero-padding strategy**: Unused positions are zero-padded and masked out during attention
3. **Strided memory layout**: `[num_layers, num_kv_heads, context_length, head_dim]`

The attention mask must be carefully constructed to:
- Mask out padding tokens in the current input
- Mask out unfilled positions in the KV cache
- Implement causal masking for autoregressive generation

### Why Zero-Padding Works

The attention computation:
```
O = softmax(QK^T / sqrt(d)) * V
```

With Q of shape `[S, d]` and K/V of shape `[L, d]`:
- Zero-padded positions in K/V contribute zero to the dot product
- Combined with proper attention masking, these positions don't affect the output
- This allows us to use fixed-size buffers while supporting variable-length sequences

### Multi-Context-Length (Multi-CL) Support

Some model exports provide multiple context length variants (e.g., 512, 1024, 2048, 3072, 4096). The runtime:
1. Starts with the smallest sufficient context length
2. Automatically "promotes" to larger variants as the sequence grows
3. Reshapes the KV cache when switching between variants

This optimizes memory usage and potentially improves performance for shorter sequences.

---

## Repository Architecture

```
qnn-run/
├── assets/                    # Static assets (logos, images)
├── docs/                      # Documentation (you are here)
├── core/                      # Core framework library (geniex_core)
│   ├── include/
│   │   ├── graph.h            # Graph wrapper for QNN execution
│   │   ├── model.h            # Base Model class
│   │   ├── types.h            # Shared data structures
│   │   ├── llm/               # LLM-specific components
│   │   │   ├── llm_model.h    # LLMModel: prefill/decode logic, KV cache
│   │   │   ├── llm_types.h    # LLMSpec, LLMRunContext
│   │   │   ├── input_provider.h  # CPU-side input preprocessing
│   │   │   └── llm_utils.h    # RoPE, attention mask utilities
│   │   ├── vlm/               # Vision-Language Model components
│   │   │   ├── vlm_model.h    # VLMModel: multimodal embedding injection
│   │   │   ├── vision_encoder.h
│   │   │   └── audio_encoder.h
│   │   └── pipeline/
│   │       └── llm_pipeline.h # High-level API with tokenizer + chat template
│   └── src/                   # Implementation files
├── models/                    # Model-specific configurations and examples
│   ├── phi3_5/                # Microsoft Phi-3.5 Mini
│   └── qwen3/                 # Alibaba Qwen3 (4B, 8B variants)
├── qnn-api/                    # QNN SDK integration layer
│   ├── include/QNN/           # QNN SDK headers
│   └── src/
│       ├── qnn-api/           # QNN API wrappers (QnnApi, IOTensor, etc.)
│       └── MmappedFile/       # Memory-mapped file utilities
├── third-party/               # External dependencies
│   ├── windows/           # HTP runtime libs for Windows
│   ├── android/           # HTP runtime libs for Android
│   ├── linux-gcc11.2-v73/     # HTP runtime libs for Linux
│   └── geniex-proc/              # Tokenizer and preprocessing (submodule)
├── CMakeLists.txt             # Build configuration
└── build_android.sh           # Android cross-compilation script
```

### Core Components

#### `Graph` (graph.h)
Wraps a single QNN graph execution unit. Responsibilities:
- Tensor I/O (write inputs, read outputs)
- Type conversion (float32 ↔ quantized formats)
- Graph execution via QNN API

#### `Model` (model.h)
Base class for all models. Provides:
- QNN backend initialization
- Graph loading and setup
- Inter-graph tensor connections
- Sub-model composition (for multi-component architectures)

#### `LLMModel` (llm/llm_model.h)
Extends `Model` for language models. Implements:
- Prefill/decode loop orchestration
- KV cache management (copy, reshape, reset)
- Multi-shard execution with hidden state passing
- Multi-context-length promotion

#### `InputProvider` (llm/input_provider.h)
Abstract interface for CPU-side tensor preparation. Built-in implementations:
- `EmbeddingInputProvider`: Token ID → embedding lookup
- `TokenIdInputProvider`: Pass-through for on-device embedding
- `RoPEInputProvider`: Rotary position embedding computation

#### `LLMPipeline` (pipeline/llm_pipeline.h)
High-level API combining:
- Tokenizer integration
- Chat template formatting
- Streaming generation callback
- KV cache persistence (save/load)

#### `VLMModel` (vlm/vlm_model.h)
Extends `LLMModel` for vision-language models:
- Vision/audio encoder integration
- Multimodal embedding injection via `maskedScatter`
- Position ID handling for image/audio tokens

---

## Class Hierarchy

```
Model (base)
├── LLMModel (text generation)
│   └── VLMModel (multimodal generation)
│
InputProvider (interface)
├── EmbeddingInputProvider
├── TokenIdInputProvider
└── RoPEInputProvider
```

---

## How to Add a New Model

### Step 1: Create Model Directory

```bash
mkdir models/your_model/
```

### Step 2: Define the Model

Create `your_model.h`. For most standard decoder-only LLMs, fill in an `LLMSpec` and return a plain `LLMModel`:

```cpp
#pragma once
#include "llm/llm_types.h"
#include "llm/llm_model.h"
#include "llm/input_provider.h"

namespace geniex {
namespace your_model {

inline LLMSpec makeSpec() {
    return LLMSpec{
        // Hidden state tensor names (one per shard boundary)
        .in_states_names  = {"input_embeds", "hidden_state_mid"},
        .out_states_names = {"hidden_state_mid", "logits"},

        // KV layer distribution across shards
        .start_layer_idxs = {0,  16},
        .end_layer_idxs   = {15, 31},

        // KV tensor naming patterns ({} replaced with layer index)
        .kv_key_in_pattern    = "past_key_{}_in",
        .kv_value_in_pattern  = "past_value_{}_in",
        .kv_key_out_pattern   = "past_key_{}_out",
        .kv_value_out_pattern = "past_value_{}_out",

        // Sequence length configurations
        .seq_len_prefill = 128,
        .seq_len_decode  = 1,

        // Architecture parameters
        .hidden_size   = 4096,
        .num_heads     = 32,
        .num_kv_heads  = 8,
        .head_dim      = 128,
        .vocab_size    = 32000,

        // Supported context lengths (sorted ascending)
        .context_lengths = {4096},

        .eos_token_ids = {2},  // EOS token ID(s)
    };
}

inline LLMModel makeModel() {
    LLMModel m(makeSpec());
    // Add input providers based on model requirements
    m.addInputProvider(std::make_unique<EmbeddingInputProvider>());
    m.addInputProvider(std::make_unique<RoPEInputProvider>(128, 10000.0f));
    return m;
}

} // namespace your_model
} // namespace geniex
```

If the default flow isn't sufficient, subclass `LLMModel` directly and override whichever methods need different behavior. The one dedicated extension point is `onInitialized()` (called after all graphs are loaded; always invoke the base first). Beyond that, any public or protected method — including `generate()` — can be shadowed in the subclass to provide completely different logic.

### Step 3: Create Example Executable

Create `your_model_example.cpp` following the pattern in existing examples:

1. Parse command-line arguments
2. Configure `QnnRuntimeConfig` (backend paths)
3. Configure `ModelConfig` (model binary paths, tokenizer)
4. Initialize model with `model.initialize(runtime_cfg, model_cfg)`
5. Run inference loop with `model.generate()`

### Step 4: Add CMake Configuration

Create `CMakeLists.txt` in your model directory:

```cmake
add_executable(your_model your_model_example.cpp)
target_link_libraries(your_model PRIVATE geniex_core)
set_target_properties(your_model PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)
```

Add to root `CMakeLists.txt`:
```cmake
add_subdirectory(${CMAKE_SOURCE_DIR}/models/your_model)
```

### Step 5: Prepare Model Files

Organize model files in `modelfiles/your_model/`:
- `*.bin` - Context binary shards
- `tokenizer.json` - Tokenizer configuration
- `embedding.bin` - Token embeddings (if CPU-side)
- `htp_backend_ext_config.json` - HTP configuration

---

## Performance Considerations

### Optimization Tips

1. **Match context length to workload**: Use Multi-CL exports when available to avoid allocating 4096-token buffers for short conversations.

2. **Minimize CPU-NPU transfers**: Input providers run on CPU; keep preprocessing efficient.

3. **Batch prefill when possible**: Longer prefill chunks (up to VTCM limits) are more efficient than many small ones.

4. **Profile with verbose mode**: Example executables support `--verbose` for timing breakdown.

### Expected Performance Metrics

- **TTFT**: Primarily determined by prefill speed and prompt length
- **Decode throughput**: Tokens per second during generation (typically 20-50 tok/s on Snapdragon 8 Gen 3)
- **Memory footprint**: Dominated by KV cache size (`num_layers * num_kv_heads * context_length * head_dim * 2 * dtype_size`)

---

## Development Workflow: xtensor to QNN

### The Trade-off: Efficiency vs. Intuition

Working directly with QNN tensors is **memory-efficient** (no extra copies, direct NPU buffer access) but can be cumbersome during development:
- Manual buffer pointer management
- No familiar NumPy-like syntax
- Harder to debug intermediate values

We provide an alternative development path using **xtensor**, a C++ library with NumPy-like semantics. This introduces an extra memory copy but offers intuitive tensor manipulation familiar to PyTorch/NumPy users.

### Two-Phase Development Process

```
Phase 1: Prototype with xtensor     Phase 2: Optimize to QNN tensors
┌─────────────────────────────┐     ┌─────────────────────────────┐
│  • NumPy-like syntax        │     │  • Direct buffer access     │
│  • Easy debugging           │ ──► │  • Zero-copy operations     │
│  • Familiar operations      │     │  • Production-ready         │
│  • Extra memory copies      │     │  • Minimal overhead         │
└─────────────────────────────┘     └─────────────────────────────┘
```

**Phase 1: Develop with xtensor**

Write your inference logic using xtensor's intuitive API:

```cpp
#include <xtensor/xarray.hpp>
#include <xtensor/xview.hpp>

// NumPy-like tensor manipulation
xt::xarray<float> hidden = xt::zeros<float>({batch, seq_len, hidden_size});
auto chunk = xt::view(hidden, xt::range(0, 128), xt::all());
xt::xarray<float> normalized = (hidden - xt::mean(hidden)) / xt::stddev(hidden);
```

**Phase 2: Convert to QNN tensors**

Once your logic is verified, convert to direct QNN buffer operations:

```cpp
// Direct QNN buffer access (zero-copy)
float* hidden_ptr = static_cast<float*>(graph.inputPtr("hidden_state"));
// ... write directly to buffer
```

### Automated Conversion

The xtensor → QNN conversion can be **automated by AI** since it follows predictable patterns:
- `xt::xarray<T>` → raw pointer + shape metadata
- `xt::view` → offset calculation into shared buffer
- `xt::zeros/ones` → `memset` or loop initialization

### Reference Documentation

| Document | Description |
|----------|-------------|
| [xtensor-prompt.md](xtensor-prompt.md) | PyTorch/NumPy to xtensor conversion reference |
| [xtensor2qnn.md](xtensor2qnn.md) | xtensor to QNN tensor conversion guide |

### Key xtensor Concepts

| xtensor | NumPy/PyTorch Equivalent | Notes |
|---------|--------------------------|-------|
| `xt::xarray<float>` | `np.ndarray` / `torch.Tensor` | Dynamic shape |
| `xt::xtensor<float, 3>` | Fixed-rank tensor | Compile-time rank |
| `xt::view(arr, xt::range(a, b))` | `arr[a:b]` | Returns a view, not a copy |
| `xt::adapt(ptr, shape)` | Wrap raw pointer | Zero-copy adapter |
| `xt::eval(expr)` | Force evaluation | xtensor uses lazy evaluation |

### When to Use Each Approach

| Scenario | Recommendation |
|----------|----------------|
| Initial prototyping | xtensor |
| Debugging complex logic | xtensor |
| Production deployment | QNN tensors |
| Performance-critical paths | QNN tensors |
| One-off preprocessing | xtensor (acceptable overhead) |

---

## Further Reading

- [Qualcomm AI Hub Models](https://aihub.qualcomm.com/compute/models?domain=Generative+AI)
- [QNN SDK Documentation](https://docs.qualcomm.com/bundle/publicresource/topics/80-63442-50/overview.html)
- Root [README.md](../README.md) for build instructions and supported models
