# Engineering Principles

## Design philosophy

**LLM-first runtime.** A VLM is fundamentally an autoregressive decoder runtime. Modality encoders produce decoder-space embeddings that are injected into the token stream; the `LLMModel` prefill/decode loop runs unchanged.

**Generalize the workflow, not the math.** The core runtime generalizes orchestration (graph sequencing, KV management, shard wiring, context-length switching). Model-specific numerical details stay in leaf model code.

**Pluggable CPU-side compute.** Embeddings, RoPE, attention masks, and multimodal positions are injected through `InputProvider` subclasses rather than hardcoded inside the decode loop.

## Rules

1. **`core/` must never contain model-specific code.** Every class/function in `core/` must be reusable by at least two models (or be a generic algorithm parameterized by model-specific data). Model-specific data (hardcoded weight arrays, magic constants) belongs in `models/`.

2. **Name algorithms, not models.** Core classes are named after what they *do*, not which model first needed them. Example: `SlidingWindowAttention` (not `MistralAttention`).

3. **Prefer composition over inheritance.** `LLMModel` is configured via `LLMSpec` + `InputProvider` injection. Only subclass when the model needs to override *runtime behavior* (e.g. `VLMModel` overrides prefill).

4. **No per-family header unless the family needs one.** A plain decoder-only family is fully described by its bundle, so it uses the shared `auto_model::makeModel` / `makePipeline` (`core/include/pipeline/auto_model.h`) and its folder holds only example `.cpp` files. `makeLLMPipeline` in `models/dispatch.h` serves it automatically as the fallback -- no dispatch entry needed unless the family needs a knob (e.g. BOS) keyed off config.json's `architectures[0]`. Add a header (or a `.cpp`) only when the family overrides runtime behaviour — Gemma3/4's per-layer embeddings, the SSD/EAGLE speculative variants, the VLM towers.

5. **Keep `core/` dependency-free.** `geniex_core` has no optional dependencies. VLM preprocessing lives in `geniex_vlm`.

## When to extend `core/` vs. `models/`

**Add to `core/` when:**
- The algorithm is reusable by at least two model families.
- The class is named after what it *does*, not which model uses it.
- No model-specific constants baked in.

**Keep in `models/` when:**
- Logic is only meaningful for one model family.
- Contains hardcoded model weights or constants (pass them to generic `core/` classes).

**When a model needs a `.cpp` (not just a header):**
- Runtime logic that cannot be expressed as `LLMSpec` + `InputProvider` composition (e.g. VLM subclass overriding `encodeVision()`).

**Extend existing class vs. create new:**
- **Add parameters** when the new behavior generalizes the current behavior.
- **Create a sibling class** when the algorithm is structurally different.
- **Subclass `LLMModel`/`VLMModel`** only when overriding runtime behavior. Never subclass just to set parameters.