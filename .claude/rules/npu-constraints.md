# NPU / QAIRT Runtime Constraints

These hardware constraints shape all architectural decisions in this codebase.

## Static graph execution

Every compiled graph has fixed tensor shapes. You cannot modify tensor shapes at runtime, use dynamic control flow, or resize intermediate activations. Different input configurations (prefill vs. decode, different context lengths) require separate pre-compiled graphs.

## Prefill and decode phases

- **Prefill** uses a fixed chunk size (128 tokens). Long prompts are chunked; short prompts are padded.
- **Decode** always processes exactly 1 token.

```
300-token prompt → 3 prefill passes: [0-127], [128-255], [256-299 + padding]
50-token prompt  → 1 prefill pass:   [0-49 + 78 padding tokens]
```

## KV cache

- Pre-allocated at max context length with zero-padding for unused positions.
- Layout: `[num_layers, num_kv_heads, context_length, head_dim]`
- Attention mask must mask out padding tokens, unfilled KV positions, and enforce causality.

## Multi-context-length (Multi-CL) promotion

Models with multiple CL variants (e.g. 512, 1024, 2048, 4096) start with the smallest sufficient CL and promote upward, reshaping the KV cache buffer when switching.

## Weight sharing

Weights are shared across multiple graph variants (prefill/decode, different CLs). Only activation buffers differ.

## VTCM size limits

Vector Tightly Coupled Memory (fast on-chip memory, analogous to GPU shared memory) constrains maximum sequence length per graph (typically 128 for prefill) and activation tensor sizes.

## 2GB file size limit

Each context binary file cannot exceed 2GB. Large models are sharded into multiple `.bin` files, each containing a subset of layers.

## Glossary

| Term | Definition |
|------|------------|
| Context Binary | Pre-compiled graph in QAIRT format (`.bin`). Fixed input shapes. |
| Graph | Single executable unit: one `(sequence_length, context_length)` config. |
| Shard | Model portion split across multiple `.bin` files (2GB limit). |
| CL | Context Length — max tokens the KV cache holds. |
| AR Length | Tokens per forward pass. 128 (prefill) or 1 (decode). |
| VTCM | Vector Tightly Coupled Memory — fast on-chip NPU memory. |
| HTP | Hexagon Tensor Processor — the NPU compute unit. |
| PD | Protection Domain — an isolated DSP address space, roughly an NPU-side process. Weights *and* every deserialized graph's persistent I/O must map into one PD, whose capacity is ~4 GB on X Elite. No API reports its free space. |
| QAIRT | Qualcomm AI Runtime SDK. |
| Multi-CL | Multiple context-length graph variants for efficiency. |
| TTFT | Time To First Token. |