# Gemma4 (E2B / E4B) on geniex-qairt

**E2B status: validated on Snapdragon X Elite (X1E78100, Hexagon v73), 2026-07-21.**
Output matches Genie token-for-token under greedy decoding; decode ~25 tok/s vs Genie's ~22.

**E4B status: builds, untested — no E4B context binaries exist locally.** It shares this
directory's code verbatim; only bundle dimensions differ (see below).

## Why E2B and E4B share one source

Everything that differs between the two is either read from the bundle's `genie_config.json`
or inferred from the loaded graphs — so there is no per-variant inference logic.

| | E2B | E4B | comes from |
|---|---|---|---|
| `hidden_size` | 1536 | 2560 | `dialog.embedding.size` |
| `num_hidden_layers` | 35 | 42 | inferred from graph KV tensors |
| per-layer stream | 8960 | 10752 | `dialog.perlayer-embedding.size` (= layers × 256) |
| `num_key_value_heads` | 1 | 2 | inferred from graph KV tensors |

Identical across both: vocab 262144, bos 2, eos `[1, 106]`, pad 0, `head_dim` 256 /
`global_head_dim` 512, sliding window 512, and both RoPE blocks (global: proportional,
partial-rotary 0.25, θ=1e6; local/SWA: default, θ=1e4).

`models/dispatch.h` already routes any `gemma4_*` / `gemma3_*` `model_id` here, so E4B needs
no dispatch change either.

## Bundle layout

`modelConfigFromDirectory()` takes the bundle dir, but `bundleDirOf()` then resolves the bundle
as **the parent of `ctx-bins[0]`** — so the `.bin` shards must sit *directly* in the bundle dir,
not in a `models/` subdirectory, or the tokenizer and LUT lookups land one level too deep.

```
modelfiles/gemma4_e2b/
├── genie_config.json           # dialog config: RoPE, cache groups, LUT specs
├── tokenizer.json
├── tokenizer_config.json       # MUST contain an inline "chat_template" (see below)
├── chat_template.jinja
├── gemma4VL_..._1_of_2.serialized.bin      # hardlinks into the asset tree
├── gemma4VL_..._2_of_2.serialized.bin
├── embedding_int16_lut.bin                 # 805 MB, ufixed16
└── embed_token_int8_lut.bin                # 2.35 GB, ufixed8
```

The four large files are **NTFS hardlinks** into
`assets/gemma4-e2b-vl-v73-ce-notebook/assets_4B_int4_LPBQ_left_pad_name_fix/` — no duplication.

## Quantized embedding LUTs (the main core addition)

Gemma4's LUTs ship quantized, and dequantizing them into RAM is not an option: the E2B
per-layer table is 2.35 GB as int8 and **9.4 GB as float32**, on a 15.6 GB box.

`QuantizedLut` (`core/include/llm/quantized_lut.h`) memory-maps the table and converts rows on
demand; `EmbeddingInputProvider::setQuantization()` switches the provider onto it. Two paths:

- **Byte-copy fast path** — when the graph input's dtype and `(scale, offset)` equal the table's,
  stored bytes go straight into the tensor. This is what actually runs for both Gemma4 streams
  (`lut_enc == act_enc` in the export), and it is what makes the plugin bit-exact with Genie,
  which short-circuits identically.
- **Dequantize path** — otherwise, rows are dequantized (`real = scale * (stored + offset)`,
  the QNN convention) and `Graph::write` requantizes.

Encodings must be copied **verbatim** from the export's `embed_encodings.json` /
`embed_tokens_encodings.json`. A mismatch here is the classic silent "garbage tokens from
token 1" failure — it does not error, it just produces nonsense.

## Runtime requirements

- **QAIRT 2.48 DLLs.** The context binaries are 2.48-built; the previously vendored 2.45
  runtime rejects them with `Using newer context binary on old SDK` / `err 5000`.
  `third-party/windows/` now holds 2.48.0.260626 win-ARM64 libs + v73/v81 skels.
  The QNN headers in `qnn-api/include/` are deliberately left at their original version —
  QNN is backward compatible, and this repo already ran 2.45 DLLs against them.
- **No `htp_backend_ext_config.json`.** 2.48's `QnnHtpNetRunExtensions.dll` segfaults while
  parsing one through this code path. Without it the runtime uses `ModelConfig::perf_profile`
  (default `BURST`), which is what we want anyway — hence ~25 tok/s. If you re-add one, note
  that `llm_decode_burst` is rejected by this wrapper and `hmx_timeout_us` must be
  ≤ 1000000 µs.
- **`tokenizer_config.json` needs an inline `chat_template`.** Gemma4 ships it as a separate
  `chat_template.jinja`, which the loader does not look for.
- The bundled `chat_template.jinja` has one edit: three adjacent string literals inside a
  `raise_exception(...)` (a tool-calling error path) were joined into one. minja — the Jinja
  engine geniex-proc bundles — does not support Python-style implicit concatenation and fails
  the whole template otherwise. Behaviour is unchanged.

## Run

```powershell
build_v73.bat            # builds geniex_core + gemma4_e2b + gemma4_e4b

$B = "modelfiles\gemma4_e2b"
build-v73\bin\Release\gemma4_e2b.exe --model-dir $B --prompt "The capital of France is" --verbose
build-v73\bin\Release\gemma4_e2b.exe --model-dir $B --chat --prompt "tell me who you are" --verbose

# multi-round: KV is kept across turns, so round N only prefills its own text
build-v73\bin\Release\gemma4_e2b.exe --model-dir $B --verbose `
    --turn "What is Newton's 1st law of motion?" --turn "what about 2nd"
```

Interactive mode is also a conversation now (`reset` clears history). Genie has to re-send the
whole transcript each round because its `"type": "basic"` dialog is stateless; keeping KV is why
round 2 here prefills ~20 tokens instead of ~336.

## Validation vs Genie

Reference: `model-onboard/genie/gemma4-E2B-qairt-CN-enabled`, same assets, same box.
Compared under **greedy** decoding on both sides (Genie's shipped config samples at temp 0.8, so
it was re-run with `temp 0 / top-k 1`).

| Test | Genie (greedy) | Plugin (greedy) |
|------|----------------|-----------------|
| `The capital of France is` | ` Paris.\n\nParis is a city in France.\n\nParis is a major city in France…` | **identical** |
| `tell me who you are` (chat) | `I am Gemma 4, a Large Language Model developed by Google DeepMind. I am an open weights model.` | **identical** |
| Newton 1st → "what about 2nd" | Law of Inertia → F = ma | same, context carried |

> The stray `物体` ("object") token in long physics answers is **this CN-enabled asset's own
> behaviour under greedy decoding — Genie emits it at the identical position.** Not a plugin defect.

| Metric | Genie | Plugin |
|--------|-------|--------|
| Decode | 18–22 tok/s | **24–26 tok/s** |
| TTFT (short prompt) | ~100–140 ms | ~105–128 ms |
| Multi-round round-2 prefill | 336 tokens (full replay) | ~20 tokens (KV kept) |

---

## Vision (VLM)

Built only with `-DGENIEX_BUILD_VLM=ON` (`build_vlm.bat`). Two context binaries are used: the
VEG (Visual Embedding Generator, `vit/serialized/veg_xelite_v73.serialized.bin`) and the usual
LLM decoder bundle.

```
image -> Gemma4Processor (geniex-proc)  -> pixel_values [1,2520,768] + image_position_ids [1,2520,2]
      -> Gemma4VisionEncoder (VEG)      -> vision_embedding [1,256,1536]
      -> spliced into inputs_embeds at the 256 image-token positions
      -> the ordinary Gemma4 prefill/decode loop, unchanged
```

```powershell
build_vlm.bat
build-v73\bin\Release\gemma4_vlm.exe `
    --model-dir modelfiles\gemma4_e2b `
    --veg-dir  ...\gemma4-e2b-vl-v73-ce-notebook\vit\serialized `
    --image ...\images\images.jpg --prompt "describe this image" --verbose
```

### The per-layer PAD rule

Splicing vision into `inputs_embeds` is only half the job. `per_layer_inputs` must **also** be
redirected at those positions — to the **PAD** row, not the image token's own row.
`Gemma4Model.forward` rewrites multimodal positions before the per-layer lookup:

```python
llm_input_ids = torch.where(multimodal_mask, pad_token_id, llm_input_ids)
per_layer_inputs = self.language_model.get_per_layer_inputs(llm_input_ids, ...)
```

Getting this wrong corrupts the per-layer input at all 35 layers for every image position, and
the model answers *"Please provide the image you are referring to"* — it behaves as though no
image were attached at all, which reads like a broken vision encoder rather than a per-layer bug.
`EmbeddingInputProvider::setTokenSubstitution()` implements it; `Gemma4Model::setVisionEmbeddings()`
applies both halves together so they cannot drift apart.

### Validation vs Genie

Genie reference: the surrogate-LUT route in `model-onboard/genie/gemma4-E2B-qairt-CN-enabled`
(§4.2/§4.4 of its `VLM_E2E.md`). Both sides greedy (`temp 0 / top-k 1`), both fed the **same**
`vision_embedding`, on `images/images.jpg`.

| Check | Result |
|---|---|
| Prompt token stream vs HF `apply_chat_template` | **identical**, 270/270, 0 diffs |
| Vision rows entering the graph vs Genie's LUT | **bit-identical**, 393216/393216 elements |
| "What text appears in this image?" | both `"On-device AI is here"` |
| "Is there an animal…?" | both `No` |
| "Main color of the background?" | both `Blue` |
| "Photograph or digital illustration?" | both `Illustration` |
| "Mostly dark or mostly bright?" | both `Dark` |
| "Does this image contain any text?" | plugin `no`, Genie `No` (casing only) |
| "How many people…?" | plugin `0`, Genie `1` |
| Free-form `describe this image` | same subject, both read the on-image text; wording diverges |
| Decode | 21–22 tok/s, TTFT ~170 ms (270-token prompt) |

6 of 7 well-posed questions agree (5 token-exact, 1 casing-only). Note both runtimes share the
same quirk on "does this image contain any text" — both answer *no* even though both correctly
extract `"On-device AI is here"` when asked directly. Shared failure modes are evidence the two
are doing the same thing.

Inputs are bit-identical and short factual answers agree, so the vision plumbing is consistent
with Genie. Free-form wording and the (ambiguous) people-count still differ — but that is **not a
property of the vision path**, and it is worth being precise about why, because "the VLM disagrees
with Genie" is the wrong conclusion to draw from it:

| control | result |
|---|---|
| text-only, 72-token chat prompt (single prefill chunk), greedy | plugin and Genie agree for ~12 tokens, then diverge |
| text-only, 196-token chat prompt (two prefill chunks), greedy | agree for ~25 tokens, then diverge |
| same VLM prompt, `floatToTfN` reverted to truncation | output **byte-identical** to the rounding build |

So the divergence (a) reproduces with **no image involved at all**, (b) is not a multi-chunk
prefill artifact, and (c) predates the rounding fix. It is a pre-existing property of this runtime
versus Genie on longer generations — two independent decoder implementations over the same context
binaries, differing in mask construction, KV layout and host-side RoPE table computation — and it
was already present (and accepted) when the text-only path was validated. Short prompts with short
answers stay token-identical on both paths, which is why Goal-2's checks passed cleanly.

Root-causing that residual difference is decoder work, not vision work.

### Do not "fix" the partial-RoPE layout (tested, it regresses)

The obvious-looking suspect for the residual difference is `PartialRoPEInputProvider`. For the
global layers `position_ids_cos` is `[1,1,128,256]` while `halfDim()` is only
`global_head_dim(512) * partial_rotary_factor(0.25) / 2 = 64`, so `ropeCapacityRows()` treats the
tensor as **512 rows of 64** rather than 128 rows of 256, and each row ends up carrying positions
`4t..4t+3` instead of `t` followed by identity padding for the non-rotated dims. That reads like a
clear bug, and rewriting it to "128 rows of [64 rotated | 192 identity]" (the layout decode
already uses) looks strictly more principled.

**It was implemented and measured, and it regresses.** With the rewrite, `tell me who you are`
changes from Genie's exact `"…I am an open weights model."` to `"…I am an open weights Large
Language Model."`; an alternative `cat(freqs,freqs)` layout diverges further still. Neither
variant improved long-range verbatim recall (all three, including the current code, recall an
embedded access code across a 189-token two-chunk prompt). The exported graph's actual contract is
what the current code emits — the theory was wrong, the measurement decides. Reverted.

> **Known limitation, inherited from the export, not from this runtime.** Gemma4 wants *blockwise
> bidirectional* attention across image tokens (`create_masks_for_vision_model`); neither Genie nor
> this runtime implements it — the graphs build a purely causal mask. In shard 1 only layers 4/9/14
> are global and the other 12 of 15 are sliding-window, so ~80 % of layers read the image as a
> causal sequence. Low-level appearance and embedded text survive; fine spatial reasoning does not.

### Two traps worth knowing

* `qnn-net-run` **without `--use_native_input_files`** reads every input file as float32 and casts
  to the tensor dtype. For the INT_32 `image_position_ids` that silently turns every id into 0, so
  a "reference" VEG output generated that way is a run with no position information at all. Always
  pass the flag for this graph.
* `floatToTfN` used to truncate toward zero. On a whole quantized tensor that is a systematic
  −0.5 LSB bias rather than zero-mean rounding error (it moved 50 % of the vision elements). It now
  rounds to nearest, matching Genie and the QNN SDK's own `datautil::floatToTfN`; the vision rows
  are bit-identical to Genie's as a result. Pinned by `FloatToTfN.RoundsToNearest`.
