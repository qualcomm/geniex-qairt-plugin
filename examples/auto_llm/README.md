 # auto_llm

A config-driven LLM loader that constructs and runs any standard
transformer decoder model (Llama, Qwen, Phi, Falcon, etc.) without
per-family code.

## Motivation

The QAIRT runtime already provides a general-purpose wrapper around the
NPU execution engine (`Model` → `LLMModel`). Most transformer decoder
architectures share the same runtime structure: chunked prefill, single-
token decode, KV cache management, embedding lookup, and RoPE position
encoding. The differences between model families reduce to numerical
hyperparameters (head counts, hidden size, RoPE base/scaling) and chat
template formatting — not runtime logic.

This means we can build an **abstract layer** on top of `LLMModel` that
constructs any supported decoder model purely from its bundle metadata,
without hardcoding model-specific constants or linking against per-family
headers. That abstraction is `auto_llm`.

This also extends to customized architectures — for example, a Llama-based
model with fewer layers, different head counts, or a non-standard context
length. As long as the model follows standard transformer decoder execution
(prefill/decode with embedding + RoPE), `auto_llm` will construct it
correctly from whatever dimensions the bundle metadata declares. No source
changes are needed to support a new architectural variant.

## Why this works

All transformer-based decoder LLMs in the codebase share:

1. The same execution flow: prefill (128-token chunks) → decode (1 token).
2. The same two CPU-side input providers: embedding lookup + RoPE.
3. The same KV cache layout and multi-CL promotion logic.
4. Chat templates expressible as HuggingFace Jinja templates.

The only things that vary between families are configuration values
(dimensions, layer counts, RoPE parameters, EOS tokens) — all of which
are already present in the QAIRT export bundle's `metadata.json` and
`genie_config.json`.

`auto_llm` reads these files at runtime and calls the same `core/`
loaders (`buildSpec`, `makeEmbeddingProvider`, `makeRoPEProvider`) that
per-family headers call with hardcoded constants. The result is
identical — a fully configured `LLMModel` — but driven entirely by
config.

## What this example shows

Previously, adding a new model family required writing a per-family file
(`models/qwen3/qwen3.h`, `models/llama3/llama3.h`, …) that picks the
right chat template formatter and the right input providers. This example
shows that's no longer necessary for any LLM that:

- Can be represented by `LLMModel` (no custom decode loop, prefill flow,
  or KV management).
- Uses the standard embedding + RoPE input providers that every existing
  LLM family already uses.

The two blockers historically preventing a generic loader were:

1. **Chat template** had to be a `ChatTemplateFunc` bound at pipeline
   creation. The `Pipeline` class in [`auto_llm.h`](./auto_llm.h) lifts
   that constraint by reading the Jinja chat template from the bundle's
   `tokenizer_config.json` at runtime, via
   `geniex-proc::Tokenizer::apply_chat_template`.
2. **Input providers** had to be added by family-specific code. This
   example reads the embedding + RoPE configuration from the bundle's
   metadata via `geniex::makeEmbeddingProvider` /
   `geniex::makeRoPEProvider` and wires them automatically.

The result is one entry point that runs any text-only QAIRT LLM bundle:

```cpp
auto pipe = geniex::auto_llm::makePipeline(runtime_cfg, model_cfg);
auto out  = pipe->generateChat(messages, gen_cfg);
```

No model name in the caller's code, no model spec, no template binding.

## Bundle layout

`--model-dir` must point at a QAIRT export bundle containing:

| File                             | Required | Purpose                                  |
|----------------------------------|----------|------------------------------------------|
| `metadata.json`                  | yes      | Tensor shapes, shard wiring, model_id    |
| `genie_config.json`              | yes      | Dialog type, BOS/EOS, RoPE config        |
| `tokenizer.json`                 | yes      | HuggingFace fast tokenizer               |
| `tokenizer_config.json`          | yes      | Chat template (Jinja) + bos/eos strings  |
| `*.bin`                          | yes      | Compiled context-binary shards           |
| `htp_backend_ext_config.json`    | optional | HTP backend extensions                   |
| `embedding_weights.raw` / `embed_tokens.npy` | optional | CPU-side embedding LUT (for bundles that move embedding off-graph) |

This is the standard layout produced by AI Hub's QAIRT export today.

## Build

From the `geniex-qairt` repo root:

```pwsh
cmake -S . -B build -DGENIEX_BUILD_EXAMPLES=ON
cmake --build build --target auto_llm --config Release
```

The binary lands at `build\bin\Release\auto_llm.exe`.

## Run

Three modes, selected by which prompt flag you pass.

### Interactive REPL (default)

```pwsh
.\build\bin\Release\auto_llm.exe `
    --model-dir <path>\<to>\<model_dir> `
    --max-tokens 256 `
    --verbose
```

Multi-turn REPL: type a prompt, the model streams the reply, repeat. Type
`exit` / `quit` or send EOF to leave. Conversation history accumulates
across turns, and the KV cache holds the matching prefix.

### One-shot (`--prompt` / `--prompt-file`)

Answers a single prompt and exits — no history, no stdin.

```pwsh
.\build\bin\Release\auto_llm.exe `
    --model-dir <path>\<to>\<model_dir> `
    --prompt "What is gravity?" `
    --metrics-json out.json
```

Use `--prompt-file <path>` instead when the prompt is long or non-ASCII, to
avoid shell quoting and encoding damage; the file is read as UTF-8 and its
trailing newline trimmed.

### Batch (`--batch-file`)

Answers many prompts in **one process**, so the (multi-GB) bundle is loaded
exactly once. The KV cache is reset before each prompt, so every answer is
independent — identical to what a one-prompt-per-process run would produce,
without paying the load cost per prompt. This is what the
[runtime benchmark](../../runtime_benchmark/README.md) uses.

```pwsh
.\build\bin\Release\auto_llm.exe `
    --model-dir <path>\<to>\<model_dir> `
    --batch-file prompts.json `
    --metrics-json results.json --quiet
```

`prompts.json` is a JSON array (or `{"prompts": [...]}`), where each element is
either `{"id": <any>, "prompt": "<text>"}` or a bare string (in which case the
0-based index becomes the id). The `id` is opaque — it's echoed back in the
results so a harness can join them to its own suite.

```json
[
  {"id": 1, "prompt": "What is gravity?"},
  {"id": 2, "prompt": "What is a llama?"}
]
```

`--metrics-json` is rewritten after **every** prompt, so a killed or crashed
batch still leaves every completed answer on disk. The process exits non-zero
only if *every* prompt failed; individual failures are reported on stderr and
recorded with `"stop_reason": "error"`.

### `--metrics-json` shape

One object in one-shot mode, an array of them in batch mode (each with the
extra `id` and `prompt` fields echoed from the batch file):

```json
{
  "answer": "Gravity is ...",
  "prompt_tokens": 13,
  "generated_tokens": 128,
  "ttft_ms": 235.412,
  "decode_ms": 4901.233,
  "tokens_per_second": 26.1156,
  "stop_reason": "eos"
}
```

All timings are the pipeline's own measurements, not wall-clock observed from
outside, so process startup and model load never pollute them. `stop_reason` is
one of `eos` / `length` / `user` / `context_length` / `error`.

### Flags

| Flag | Description |
|------|-------------|
| `--model-dir <path>`        | **Required.** Bundle directory. |
| `--tokenizer-config <path>` | Override path to `tokenizer_config.json` (default: `<model-dir>/tokenizer_config.json`). |
| `--system <text>`           | System prompt. Applied once at startup in the REPL; prepended to every prompt in one-shot / batch mode. |
| `--max-tokens <n>`          | Max tokens generated per turn (default 512). |
| `--enable-thinking`         | Plumbs `{"enable_thinking":true}` into the Jinja context for reasoning models that read the field (Qwen3). No-op on templates that don't. |
| `--verbose`                 | Print TTFT / TPS metrics each turn. |
| `--prompt <text>`           | One-shot: answer this prompt and exit. |
| `--prompt-file <path>`      | One-shot: read the prompt from a UTF-8 file. |
| `--batch-file <path>`       | Batch: JSON array of `{id, prompt}`, model loaded once. |
| `--metrics-json <path>`     | Write answer + TTFT/TPS/token counts as JSON. Requires one of the three flags above. |
| `--quiet`                   | Suppress streamed output and banners. Only useful with `--metrics-json`, where the JSON file is the real output channel. Does not silence the runtime's own `[DEBUG]` logging. |

`--prompt`, `--prompt-file` and `--batch-file` are mutually exclusive.
