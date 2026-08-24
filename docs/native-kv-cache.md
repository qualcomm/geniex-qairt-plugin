# Native KV Cache

Study notes for [geniex#1445](https://github.com/qcom-ai-hub/geniex/issues/1445) — what native KV cache is,
why it exists, and what it takes to implement it driver-side.

All Genie references below are line-accurate against the vendored reference sources under
`Genie/src/qualla/engines/`.

- [1. What native KV cache is](#1-what-native-kv-cache-is)
- [2. What it is for](#2-what-it-is-for)
- [3. Driver-side implementation](#3-driver-side-implementation)
- [4. Recipes-side requirements](#4-recipes-side-requirements)
- [5. Gaps in geniex-qairt-plugin](#5-gaps-in-geniex-qairt-plugin)

---

## 1. What native KV cache is

"Native KV" means the KV cache buffers are stored **permanently in the HTP's HMX weight layout**
(`QNN_TENSOR_DATA_FORMAT_HMX_WEIGHT_LAYOUT`) instead of the logical flat layout, so the cache can be
fed to the HMX matmul units as a weight operand with **zero on-device re-layout**.

In Genie it is a `CacheManager` implementation (`NativeKV`) sitting beside the default `SmartMask`.

### Selection is automatic, not a config knob

| Step | Location |
|------|----------|
| Scan every graph variant's input specs; if any input has `dataFormat == QNN_TENSOR_DATA_FORMAT_HMX_WEIGHT_LAYOUT`, flip `_kv_update_method` to `KVManagerMode::NATIVE_KV` | `qnn-htp/nsp-model.cpp:377-388` |
| Per cache group, the same check on `key_in` (falling back to `key_out` for bert-cache models) picks `NativeKV` vs `SmartMask` | `qnn-htp/KVCache/kvmanager.cpp:81-89` |
| Record per `(AR, CL)` variant whether the KV *output* is also native | `qnn-htp/nsp-model.cpp:1108-1137`, `kvmanager.cpp:97-103` |

### The layout

`fromFlatOffset()` (`qnn-htp/KVCache/native-kv.cpp:53-81`):

```
[din, dout] -> [dout/N_TILE][din/32][(dout%N_TILE)/32][(din%32)/4][dout%32][din%4]
```

| Cache | `din` | `dout` | Tile |
|-------|-------|--------|------|
| Key   | `head_dim` | `ctx_size` | `K_TILE = 256` |
| Value | `ctx_size` | `head_dim` | `V_TILE = 64` |

The innermost `8 x 32 x 4` chunk is 1024 bytes = `KV_BLOCK_SIZE`.

### Constraints baked in

From `native-kv.cpp:18-29`:

- **uint8 / quantized only** — `State::error("Native KV only supports uint8.")`.
- **Clear value is 0, not 128** — *"Internally, HMX does not apply an offset for NativeKV tensor"*.
  `SmartMask` clears to `1 << 7` (`kvmanager.cpp:68-76`); `NativeKV` overrides to `0`.
- **New KV must land on a 32-aligned index** — `getIndexForNewKV()` rounds `n_valid_kv` up to 32, and
  `getCacheBudget()` returns `ctx_size - round32(variant)`. Violations are a hard error
  (`nsp-model.cpp:1950-1958`).

For contrast, `SmartMask` keeps the plain layout `key[n_heads, head_dim, ctx]` /
`value[n_heads, ctx, head_dim]` and writes contiguous rows (`smart-mask.cpp:60-90`).

---

## 2. What it is for

**The benefit.** Attention's `K^T . Q` and `P . V` run on HMX, which requires the tiled weight layout.
Without native KV the graph pays a layout conversion (transpose / re-tile of the whole cache) on every
forward pass — HTP cycles plus VTCM traffic and scratch space. Keeping the cache natively tiled removes
that entirely: a decode-throughput and TTFT win that also frees VTCM.

**The cost it moves to CPU.** The write-back scatter (graph KV output to KV input buffer) becomes
non-contiguous. Genie mitigates this two ways:

- If the graph's KV **outputs** are also native (`isKvOutputNativeFormat`) *and* `dst_idx`/`count` are
  both 32-aligned, `updateKV` takes a fast path (`native-kv.cpp:225-284`) of whole-tile `memcpy`s via
  `key_buffer_aligned_update` / `value_buffer_aligned_update`. This is exactly why the 32-alignment
  rule exists.
- Otherwise it falls back to per-element re-tiling (`native-kv.cpp:287-362`), and when the output is
  flat but the input native it also rebases `uint8 -> int8` (`- 128`, line 322). Genie warns about this
  case: *"The graph's KVCache has Native input and FlatBuffer output"*.

### Trade-offs

| Area | Impact |
|------|--------|
| Precision | 8-bit only; no fp16 / 16-bit KV |
| Cache utilization | 32-token write granularity leaves holes between `n_valid_kv` and `round32(n_valid_kv)` |
| Masking | The hole must be masked. `attention-mask.cpp:74-95` emits separate spans for `[past_idx, n_valid_kv)` and `[new_idx, ...)`, so the gap is naturally excluded |
| Eviction / long context | `reduceKV` / `moveKV` are per-element with an offset computation per byte — far more expensive than SmartMask's `memset` / `memcpy` |
| Save / restore | `loadCache` / `dumpCache` / `dumpHead` must de-tile and apply the +/-128 rebase to keep the on-disk format canonical |
| CL promotion | `reshapeCache` is a real per-head / per-tile move, not a stride tweak |

---

## 3. Driver-side implementation

Genie's shape is the reference: subclass the cache manager and override the layout-dependent hooks.
In dependency order:

### a. Detect and plumb the format

Read `QNN_TENSOR_GET_DATA_FORMAT()` off the KV *input* tensor per cache group at graph-load time
(`kvmanager.cpp:81-89`). Separately record, per `(AR, CL)` variant, whether the KV *output* is native
(`CacheGroup::m_isKvOutputNativeFormat`, `kvmanager.cpp:97-103`) — the fast path is gated on it. Set the
global mode flag before backend init so `QnnApi` sees it (`m_qnnApi->setKVUpdateMethod`).

### b. Buffer allocation is unchanged — only the layout differs

KV in and out share one fused allocation, with the output region immediately after the input:

- sizing: `qnn-api/QnnApi.cpp:997-1030`
- offset mapping: `registerTensorsWithBackend` -> `mapFusedBufferOffset` with `kv_offset`,
  `qnn-api/QnnApi.cpp:1060-1105`

That is the invariant every `NativeKV` method relies on: `cache.key_buf` is the input region, and the
output region is `key_buf + n_heads * head_dim * ctx_size` (`native-kv.cpp:219`, `234-235`).

### c. Implement the offset function, then the hook set

All pure virtuals of `CacheManager` (`kvmanager.hpp:105-199`):

| Hook | Native implementation |
|------|----------------------|
| `completeInit` | set clear value to 0; reject non-uint8 |
| `getIndexForNewKV` | `round32(n_valid_kv)` |
| `getCacheBudget` | `ctx - round32(variant)` |
| `clear` | plain `memset(0)` — layout-agnostic |
| `updateKV` | aligned whole-tile `memcpy` fast path + per-element fallback (incl. flat-output conversion) |
| `reduceKV` / `moveKV` | per-element via `fromFlatOffset`; no shortcuts available |
| `reshapeCache` | AR change is a **no-op** (all AR-n variants share a shape). CL change is a per-head (K) / per-head-x-tile (V) `memcpy`/`memmove` with zero-padding of the grown tail, iterating backwards when growing (`native-kv.cpp:414-520`) |
| `loadCache` / `dumpCache` / `dumpHead` | de-tile through a scratch buffer + `+/-128` rebase |

### d. Index and mask plumbing

- The scatter index tensor (`cache_index`) is filled `iota` from `new_idx`
  (`nsp-model.cpp:1961-1968`).
- `new_idx` must be validated against `ctx - ceil(AR/32)*32` before execute
  (`nsp-model.cpp:1950-1958`).
- Strategy planning must use the rounded budget: `m_isKvInputNativeFormat` changes `cacheBoundary` in
  `kvmanager.cpp:396-400`.
- The mask builder must mask both the unfilled tail and the alignment hole.

---

## 4. Recipes-side requirements

The other half of what makes native KV work — the compiled artifact must:

- Declare KV in/out with `dataFormat = HMX_WEIGHT_LAYOUT` and 8-bit ufixed data type.
- Use the `<prefix>_key/value_<idx>_in/_out` naming convention so `isKVTensor` / `getPrefix` /
  `parseLayerIndex` resolve them.
- Expose a `cache_index` input.
- Emit **native-format KV outputs** as well — otherwise execution lands on the slow per-element path
  with a warning.
- Keep all of the above consistent across every shard and every `(AR, CL)` variant of a cache group,
  since detection is a single global flip.

---

## 5. Driver support in geniex-qairt-plugin

Implemented on `feat/david/native_kv`. Detection is automatic per tensor — the same binary runs flat
and native bundles, with no build flag.

| Piece | Where |
|---|---|
| `dataFormat` surfaced onto every tensor | `TensorSpec::data_format` (`core/include/types.h`), populated in `Graph::buildSpecs` |
| Tiled addressing, conversion, restride, shift, clear, de-tile | `core/include/llm/kv_layout.h`, `core/src/llm/kv_layout.cpp` (`geniex::kv`) |
| Cache writes routed through it | `LLMModel::copyKV` / `shiftKVLeft` / `reshapeKV` / `initKVBuffers` |
| Detection, validation, rebase resolution + INFO log | `LLMModel::resolveKVLayout` |
| 32-granular reserved tail | `LLMModel::kvLen(phase, cl_idx)` |
| Tests | `tests/core/llm/kv_layout_test.cpp`, `NativeKV.*` in `tests/core/llm/llm_model_test.cpp` |
| Bundle inspector / cross-run diff | `examples/kv_layout_check/` |

Implementation notes worth knowing:

- **The unit of contiguity is 1024 bytes** — the innermost `[din_1:8][dout_1:32][din_2:4]` chunk spans
  32 `din` × 32 `dout` elements. Any copy or restride whose offsets and extents are 32-aligned reduces
  to whole-block `memcpy`s, which is what keeps `reshapeKV` (twice per `generate()`) affordable.
- **Tiled restride is pure `memmove`.** Keys are tile-major with a `kv_len`-independent tile stride
  (`head_dim * K_TILE`), so growing `kv_len` just appends tiles — one `memmove` per head. Values have a
  `kv_len`-dependent tile stride, so they need one `memmove` per `(head, tile)` region.
- **An unaligned shift of a tiled cache is slow** (element-wise re-tiling of the whole window). Only
  fixed-window `swa_*` caches shift; the runtime warns once if it happens.
- **Save/load (`saveKVCacheToFile`) stays a raw byte snapshot**, so its files are not portable between
  a native and a flat build. `kv_layout_check --dump` is the portable form.

### Measured against a real ENABLE_NATIVE_KV bundle — and Genie's formula does not match

Verified on `windows_bundle_x_elite_v245_nativekv` (Qwen3-4B eaglet target, natKV) against its flat
twin `windows_bundle_x_elite_v245`. Both exports are numerically identical — same tensor names,
dims, dtypes and quantization params for all 298 tensors; the **only** difference is `dataFormat` on
the 288 KV tensors. So a correct implementation must reproduce the flat bundle's output bitwise.

Harness: `examples/kv_layout_check/kv_layout_probe` runs a fixed prompt over the target engine and
dumps a chosen tensor. Compare only the *valid* rows — with a 5-token prompt in a 128-wide AR graph,
rows 5..127 are padding whose contents are unconstrained and differ freely between bundles.

**What is established:**

1. **The graphs are equivalent.** A single prefill chunk (cache present but fully masked) gives
   **bit-identical** `last_hidden_states` on the valid rows. So embeddings, RoPE, masking and
   execution are all fine; the KV cache bytes are the only variable.
2. **KV *outputs* are plain row-major**, despite carrying `HMX_WEIGHT_LAYOUT`. Both
   `past_key_0_out` and `past_value_0_out` ([8,1,128,128], 131072 B) are **byte-identical** to the
   flat export's. That includes `past_value_0_out`, whose `dout` of 128 spans two `V_TILE`s — so it
   is not an artefact of a single-tile shape. The runtime therefore treats KV outputs as flat
   (`GENIEX_KV_OUT_TILED=1` overrides).
3. **KV *inputs* are NOT row-major.** Forcing the whole flat path (`GENIEX_NATIVE_KV=0`) produces
   wrong output, so the input buffers really are laid out differently.

**Unresolved: the KV input byte order.** Calibrating against unrelated content (321215 differing
bytes of 368640) gives a noise floor; every candidate below scores at or slightly *worse* than that,
i.e. none is close:

| candidate for `past_key_N_in` / `past_value_N_in` | valid-row byte diffs |
|---|---|
| row-major, unchanged (`GENIEX_NATIVE_KV=0`) | 328317 |
| row-major + `-128` rebase | 340329 |
| Genie `fromFlatOffset`, tile 32 / 64 / 128 | 342217 (identical for all three) |
| Genie `fromFlatOffset`, nominal 256 with a narrowed trailing tile | 342052 |
| row-major within each tile, nominal 256 / 64 | 342537 |
| *(reference: unrelated content)* | *321215* |
| *(reference: correct)* | *0* |

Genie's `fromFlatOffset` cannot be right here as written: with `head_dim 128 / kv_len 1920 /
K_TILE 256` its fixed `din_0` stride addresses up to byte 258047 of a 245760-byte tensor, and
`getAlignedSize()` pads only to 8 bytes, so nothing absorbs the overrun. Genie only ever uses
full-CL (scatter) caches, where 256 divides the axis.

One structural observation not yet exploited: the runtime fuses each KV pair into one allocation
(`QnnApi.cpp`, `past_` prefix — `kv_in` then `kv_out`), and `1920 + 128 = 2016 + 32 = 2048 = CL`. So
the fused region is exactly a CL-wide cache. That is suggestive of Genie's scatter geometry, but the
fused buffer is `[all heads of kv_in][all heads of kv_out]` whereas a single CL-wide tiled tensor
would be head-major, so they are not the same object.

**What is needed to finish:** the authoritative byte layout the `ENABLE_NATIVE_KV` recipe expects for
`past_key_N_in` `[8,1,128,1920]` / `[8,1,128,2016]` and `past_value_N_in` `[8,1,1920,128]` /
`[8,1,2016,128]` — from the converter, or a QAIRT doc, or a dump of a known-good cache. Any candidate
can then be confirmed in about 15 s:

```pwsh
# reference (correct) and candidate, comparing chunk 2's 72 valid rows
kv_layout_probe --model-dir <flat_bundle>   --out ref.bin --tensor last_hidden_states --repeat-to 200
kv_layout_probe --model-dir <native_bundle> --out got.bin --tensor last_hidden_states --repeat-to 200
```

Until then the runtime logs a WARNING at init that the layout is unverified, and generation from a
native bundle is wrong.

### Why Genie's tile arithmetic cannot be transplanted as-is

Genie's `din_0` stride is `(tile / 32) * 1024`, reserving a full tile's worth of `dout` slots per `din`
block. That is fine when `N_TILE` divides the tiled axis — always true for Genie, whose native caches
are full-CL. It breaks for a non-scatter cache:

```
head_dim 128, kv_len 1920, K_TILE 256:
  din=127, dout=1919 -> 7*32768 + 3*8192 + 4095 = 258047
  tensor bytes/head   =  128 * 1920             = 245760   <- overruns by one tile's worth
```

`getAlignedSize()` pads only to 8 bytes, so nothing absorbs it. The real natKV export nevertheless
ships exactly these shapes (1920 for AR-128, 2016 for AR-32), so the export is using some other tile
arithmetic — which is precisely the unresolved question above. `kv::validateGeometry` asserts that
whatever layout is configured spans exactly `headStride()` bytes, so a mis-specified tile fails at
load with the numbers rather than corrupting memory.

---

## 6. Two different things are called "native KV cache"

Measuring two real ENABLE_NATIVE_KV bundles showed the name covers two independent
changes, and only the second one is implemented and verified here.

| | **Scatter cache** (implemented, verified) | **HMX tiled layout** (unverified) |
|---|---|---|
| What changes | Cache *management*: `kv_in` spans the whole CL and the graph places fresh KV inside it at `cache_index` | Cache *byte order*: bytes are tiled for direct HMX consumption |
| `dataFormat` | `FLAT_BUFFER` | `HMX_WEIGHT_LAYOUT` |
| Detected by | a `cache_index` input **and** `kv_len == CL` | `dataFormat` on the KV tensors |
| Seen in | `llama_v3_2_3b_instruct_ssd-geniex_qairt-w4a16` | `windows_bundle_x_elite_v245_nativekv` (Qwen3-4B) |
| Status | **works, matches Genie exactly** | see §5 — Genie itself fails on that bundle |

### The scatter cache

```
key_in  [1, 1, 128, 4096]   <- kv_len == CL, not CL - AR
value_in[1, 1, 4096, 128]
key_out [1, 1, 128, AR]
cache_index [1] int32       <- column where this pass's fresh KV goes
attention_mask [1,1,AR,4096]<- key axis is the whole cache, not kv_len + AR
```

The graph reads the full CL-wide cache, drops its freshly computed KV in at
`cache_index`, and attends over the whole axis. Compared to the concat cache the
driver must:

1. **Write `cache_index`** (= the write cursor `n_past`). Without it the graph
   scatters over the start of the cache — the single most damaging omission.
2. **Base the mask's fresh-key block at `n_past`** rather than at the end of the
   axis (`kvNewBase()` / `kvMaskWidth()`; `get_attention_mask`'s `row_len` and
   `new_base`).
3. **Stop reserving a tail.** `kvLen()` returns the full CL, which makes kv_len
   phase-independent and so turns the prefill<->decode restride into a no-op for
   free.

Everything else — the dense write cursor, capacity checks, CL promotion — falls out
of `kvLen()` unchanged.

### Other things this bundle needed

- **KV tensor naming.** It names the cache
  `past_nativekvcache__key_<layer>_head_<h>_in`, so the default `past_key_{}_in`
  pattern matched nothing: the model loaded, prefilled fine, and emitted gibberish
  because nothing was ever written back. `adoptKVNamingFromGraph()` now derives the
  prefix from a real tensor name when the declared patterns miss.
- **`head_dim` / `num_kv_heads` inference.** These were read by matching the
  literal `past_key_` prefix too. With the new naming they stayed 0, and a zero
  `head_dim` silently produces an *empty* RoPE table — so SSD's position-id
  override became a no-op and positions were off by the forecast prefix. Now taken
  from a resolved KV pair.
- **Three AR variants.** The bundle ships `ar1`, `ar32` and `ar128`; the runtime
  addresses two phases. `LLMSpec::min_decode_seq_len` lets a driver state the width
  its decode pass needs (SSD: 30 -> picks `ar32`) and the loader drops the rest.
- **`ParsedGenieConfig` never reached the base class** for SSD, so the provider
  chain used theta 10000 with no llama3 scaling. Also gave SSD's own RoPE table the
  same scaling the provider would apply.
- **Forecast-prefix load.** The file stores one tensor per layer holding all 8
  heads (`num_tensors = 2 * n_layers`); this export has one tensor per head. The
  loader read one head then skipped seven, misaligning the whole file. It now
  addresses the file by `(layer, head)`.
- **SSD double-committed the first token.** The init pass (whose only job is to
  produce the forecast logits that seed the first draft tree) also committed its KV
  and advanced `n_past_`, while the tree pass re-processes that same token as the
  tree root and commits it via `selectiveKVUpdate`. The token landed in the cache
  twice, shifting every later position — visible as a dropped token ("Lamas" for
  "Llamas"). Speculative decode must reproduce plain autoregressive decode exactly,
  which is the check that caught it.

### Verification

Against `genie-t2t-run` (QAIRT 2.45.0.260326, the SDK the bundle was built with),
same prompt, both greedy:

```
Genie   : Llamas are herbivores, eating grass, hay, and grains primarily.
GenieX  : Llamas are herbivores, eating grass, hay, and grains primarily.
```

Byte-identical. Genie counts 56 prompt tokens to our 55 because it prepends BOS to
a prompt that already starts with `<|begin_of_text|>`; feeding our runtime the same
56 tokens is what makes the outputs line up. Two further checks:

- **Teacher forcing.** Given the same prefix, our prefill predicts Genie's exact
  continuation (`"Llamas are herbivores,"` -> `" eating grass, hay, and"`).
- **Speculation is exact.** SSD and plain AR-1 decode produce identical text.

Performance, from Genie's own `--profile` versus ours on the same prompt:

| | TTFT | decode |
|---|---|---|
| Genie (ssd-q1) | 161.5 ms | 19.13 tok/s |
| GenieX (SSD) | 161.7 ms | 18.8 tok/s |

Reproduce with:

```pwsh
llama3_2_3b_ssd.exe --model-dir <bundle> --raw-prompt-file <bundle>\sample_prompt.txt --verbose
auto_llm.exe        --model-dir <bundle> --raw-prompt-file <bundle>\sample_prompt.txt   # plain AR-1
```
