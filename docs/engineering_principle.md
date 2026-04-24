# Engineering Principles

## Design philosophy

**LLM-first runtime.** A VLM is fundamentally an autoregressive decoder runtime. Modality encoders produce decoder-space embeddings that are injected into the token stream; the `LLMModel` prefill/decode loop runs unchanged. This keeps the highest-complexity code path in one place.

**Generalize the workflow, not the math.** The core runtime generalizes orchestration (graph sequencing, KV management, shard wiring, context-length switching). Model-specific numerical details (encoder graph order, placeholder tokens, RoPE extension factors, partial-dimension rules) stay in leaf model code.

**Pluggable CPU-side compute.** Embeddings, RoPE, attention masks, and multimodal positions are injected through `InputProvider` subclasses rather than hardcoded inside the decode loop.

## Engineering principles

1. **`core/` must never contain model-specific code.** Every class and function in `core/` must be reusable by at least two models (or be a generic algorithm parameterized by model-specific data). If a piece of logic is only meaningful for one model family, it belongs in `models/`. Model-specific *data* (e.g. hardcoded weight arrays, magic constants) must live in the model header, not in core.

2. **Name algorithms, not models.** Core classes should be named after what they *do*, not which model first needed them. For example: `SlidingWindowAttention` (not `MistralAttention`), `GroupedQueryProjection` (not `LlamaGQAProjection`). Model-specific parameters are passed in by the model header.

3. **Prefer composition over inheritance.** `LLMModel` is configured via `LLMSpec` + `InputProvider` injection, not by subclassing per model. Only subclass when the model needs to override *runtime behavior* (e.g. `VLMModel` overrides the prefill stage).

4. **Header-only model specs, `.cpp` only when needed.** Most models are fully described by a header (`makeSpec()` + `makeModel()`). A `.cpp` is only needed when the model requires runtime logic that cannot be expressed as provider composition (see guidelines below).

5. **Keep `core/` dependency-free.** `geniex_core` has no optional dependencies. VLM preprocessing libraries are isolated in `geniex_vlm`.

## Guidelines: when to extend `core/` vs. `models/`

**When to add code to `core/`:**
- The algorithm is reusable by at least two model families (or is a generic algorithm parameterized by model-specific data passed in by the caller).
- The class is named after what it *does*, not which model uses it.
- It has no model-specific constants, weight arrays, or magic numbers baked in.

**When to keep code in `models/`:**
- The logic is only meaningful for one model family (e.g. a specific encoder graph sequence, a family-specific placeholder token scheme).
- It contains hardcoded model weights or constants (e.g. RoPE extension factor arrays — these belong in the model header and are passed to generic `core/` classes).

**When a model needs a `.cpp` file (not just a header):**
- The model requires runtime logic that cannot be expressed as `LLMSpec` + `InputProvider` composition — e.g. a VLM that subclasses `VLMModel` and overrides `encodeVision()` or `preparePositions()`.
- If `makeSpec()` + `makeModel()` in a header is sufficient, no `.cpp` is needed.

**When to extend an existing `core/` class vs. subclass it:**
- **Add parameters to an existing class** when the new behavior is a generalization of the current behavior (e.g. adding `rope_fraction` to make a partial-dimension RoPE generic).
- **Create a new sibling class** when the algorithm is structurally different and would not share the parent's `forward()` logic (e.g. `LongRoPEEmbedding` recomputes `inv_freq` dynamically — inheriting `RotaryEmbedding` would leave the base `inv_freq_` unused).
- **Subclass `LLMModel`/`VLMModel`** only when the model needs to override *runtime behavior* (prefill flow, decode loop, KV management). Never subclass just to set parameters — use `LLMSpec` + providers instead.

## Code comments

Comments convey intent and context that code cannot express on its own — not a restatement of what the code does.

**Comment only where needed.** Self-documenting names are the primary documentation. Add a comment only when the name or structure alone is insufficient: non-obvious constraints or invariants, magic numbers, intentional workarounds, decisions that would otherwise look wrong, or non-trivial know-how worth preserving for the next reader.

**Header vs. implementation.** In `.h` files, comment the purpose, usage contract, and non-obvious behavior of public interfaces — this is the primary reference for callers. In `.cpp` files, comment sparingly; routine logic needs no narration.

**Be concise.** One precise sentence beats three vague ones. Drop filler like "This function is responsible for…" and state the purpose directly.

**Describe purpose, not behavior.** Write what something is *for*, not what it currently does. Behavior changes; purpose is more stable.
- ❌ `// Used by LLMModel during prefill to pad the token count`
- ✅ `// Aligns position to a chunk boundary, as required by the NPU graph's static input shape`

**Keep comments self-contained.** A comment should be understandable without consulting another file or class. Cross-references are acceptable only when the coupling is real and load-bearing (e.g., a shared buffer layout two components must agree on).

**Keep comments current.** A wrong comment is worse than none. Update or remove comments whenever the code they describe changes.

### Style
- Use `//` throughout; reserve `/* */` for file-level license headers only.
- Full sentences (capital, period) for multi-line comments; inline comments may omit the period.
- No decorative separators (e.g., `// ── Section ──────`) unless marking a genuinely distinct named section in a long file.
- No Doxygen-style tags.
