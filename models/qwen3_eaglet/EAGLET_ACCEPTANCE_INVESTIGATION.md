# qwen3-8b eaglet acceptance investigation (handoff)

Two bundles investigated, both showing near-zero EAGLE speculative acceptance
on v73 despite producing correct, coherent output text. Written for a fixing
agent to pick up with full context — read this before touching code.

## Bundles

| | `-ce-train-` | `-iot-train-` |
|---|---|---|
| HF repo | `zhic-qcom/qwen3-8b-eaglet-ce-train-w4a16-v73-iot-export-ctx-bin` | `zhic-qcom/qwen3-8b-eaglet-iot-train-w4a16-v73-iot-export-ctx-bin` |
| Local path | `C:\Users\zhic\models\qwen3-8b-eaglet-w4a16-v73` | `C:\Users\zhic\models\qwen3-8b-eaglet-iot-train-w4a16-v73` |
| Draft vocab (confirmed via `metadata.json` / load log) | 151936 (full, matches target) | 32000 (trimmed) |
| Measured acceptance | ~1.0-1.18 tokens/round (after 2 live draft-checkpoint updates) | ~1.00-1.01 tokens/round |
| top1 exact-match | 0% across 200+ rounds | 0% across 1192+ rounds (incl. 8 real UltraChat prompts) |
| Draft confidence vs. uniform baseline | 85-380x (across checkpoint updates) | ~34000x (0.22-0.23 avg) |

Both bundles need the shipped `genie_config.json`'s `ctx-bins` arrays populated
manually (ship empty) before `eaglet_example` will load:
- target: `part1_of_6.bin` .. `part6_of_6.bin`
- draft: `draft_body.bin`, `draft_head.bin`

Reference ground truth for this checkpoint family:
`C:\Users\zhic\Downloads\Tutorial_for_Qwen3_8B_Eaglet_IoT\Tutorial_for_Qwen3_8B_Eaglet_IoT\EAGLET_IOT_ACCEPTANCE_RATE.md`
reports the IoT checkpoint's own **torch FP32 baseline: BE = 2.57 tokens/round**
(`trim_vocab=False`, depth=5, top_k=8, total_token=64, 50 xlam-holdout samples).
A separate pilot report (`configs/qwen3-8b.yaml`, mentioned in the original
`-ce-train-` investigation) cites **BE ≈ 3.21** for that checkpoint family. Both
references are far above anything measured on-device.

## What's ruled out

- **Prompt format/distribution.** Tested: general text, standard Qwen3
  `<tool_call>` format, synthetic XLAM-format, real XLAM holdout-adjacent
  records, 8 real UltraChat `test_sft` records rendered through the bundle's
  own chat template. All land at the same ~1.0-1.04 tokens/round. UltraChat is
  *not* a distinguishing test even though it's literally the frequency source
  used to build the `-iot-train-` vocab trim (see below) — the degradation is
  structural, not prompt-dependent.
- **The runtime's speculative-tree logic** (tree build, pruning, accept-walk,
  KV replay-seed pairing). Traced line-by-line against the EAGLE recipe;
  found correct. See "Round 3" reasoning in git history of this file / prior
  session transcript if needed — not re-summarized here since it's not a
  current suspect.
- **A `-ce-train-` vs `-iot-train-` input-tensor-count mismatch** (EAGLET v1 vs
  v2 feature inputs). `genie_config.json` declares `"eaglet-version": 1` for
  both bundles; the ground-truth export notebook only adds a second
  `target_hidden_states` input `if eaglet_version == 2`. Both bundles'
  `draft_body.bin` have exactly one feature input (`hidden_states`) —
  consistent, not a bug.

## Two confirmed, independent problems

### 1. `-iot-train-` only: missing draft-vocab remapping (`draft-token-map`)

**Confirmed, not hypothesized** — via both the on-device graph and the
ground-truth export recipe:

- `draft_head.bin`'s own `logits` output tensor is `[1, 128, 32000]`
  (`metadata.json`), and the runtime's `inferSpecFromGraphs` log reports
  `vocab_size=32000` for the draft while the target is `151936`.
- The export recipe that produces this is
  `Tutorial_for_Qwen3_8B_Eaglet_IoT/example1/qwen3_draft_model.ipynb` (cell 20):
  when `TRIMMED_VOCABULARY_PATH` is set, it loads a vocab-trim mapping file
  (produced by `eagletv1.5_qwen3-8b/evaluation/vocab_trim.py`, which scans
  **UltraChat** distillation data, keeps the top-32000 most-frequent tokens,
  and saves the sorted real-token-id list as
  `vocab_map_topk_32000_topp_None_freq_None.{npy,txt,json}`), then does
  `lm_head_trimed = model.lm_head.weight[trimmed_vocab_index]` — i.e. **the
  draft's logits column `k` means "the k-th most-frequent surviving token,"
  not "token id k."** The notebook prints `"Using trimmed vocab map from:
  ..."` when this is active, treating the mapping file as a required
  co-artifact of the export, not an optional extra.
- The deployed bundle ships the trimmed weights but **not** that mapping
  file, and `genie_config.json`'s `draft-token-map` field is empty
  (`parseEagletConfig` logs `token_map=0`).
- The runtime already has correct, working support for this field — see
  `core/src/llm/eagle_model.cpp:99`: when `cfg_.draft_token_map` is empty, it
  returns the raw softmax index as the token id directly (`return
  static_cast<int32_t>(raw);`). That fallback is only correct for a
  full-vocab draft. For this bundle it silently misreads every single draft
  proposal.
- Embedding table is **not** trimmed in either bundle (`embed_tokens` stays
  full-vocab 151936 per the notebook — only the output/logits side is
  trimmed), consistent with what's deployed. So this is purely an
  output-side mapping gap, not a broader export inconsistency.

**Important counter-evidence — this is not the shared root cause.** The
`-ce-train-` bundle's draft is confirmed full-vocab (`vocab_size=151936`,
re-verified directly from a fresh load log, not just the old doc's
assumption) — no vocab-trim mismatch is even possible there — and it *still*
shows the same ~1.0-1.18 floor. So vocab-mapping is a real, independently
confirmed bug in `-iot-train-` specifically, but fixing it should at best be
expected to bring `-iot-train-` up to whatever floor `-ce-train-` is already
stuck at, not past it, based on current evidence.

**Fix sketch (not yet implemented, needs the actual npy/json from the
training/export side — do not fabricate one):** obtain
`vocab_map_topk_32000_...json` for this checkpoint, convert it to the
`{string_index: token_id}` format `qwen3_eaglet.h`'s existing
`draft-token-map` parser expects (`models/qwen3_eaglet/qwen3_eaglet.h:107-131`),
reference it from `genie_config.json`'s draft engine `model.draft-token-map`
field, and re-measure.

### 2. Both bundles: target/draft feature-tensor quantization-encoding mismatch

**This is the better-supported candidate for the shared ~1.0 floor.**

`core/src/llm/eagle_model.cpp` passes the EAGLE "feature" (the target's
hidden-state output that seeds/conditions the draft) from target to draft via
a raw `std::memcpy` of quantized bytes — see call sites at approximately
lines 251-252, 426-437, 585, 638-641, 671 (`tgt.outputBytes(v_body,
cfg_.target_feature_output)` copied straight into the draft's input buffer
via `cfg_.draft_feature_input`). **No dequantize/requantize step exists on
this path.** This is only numerically correct if the target's output tensor
and the draft's input tensor share the same `(scale, zero_point)`.

Measured directly from each bundle's `metadata.json` (target shard
`part5_of_6.bin`'s `add_122687` output vs. `draft_body.bin`'s `hidden_states`
input):

| Bundle | target scale / zero_point | draft scale / zero_point | scale ratio |
|---|---|---|---|
| `-iot-train-` | 0.065026 / 11283 | 0.0044489 / 34570 | ~14.6x |
| `-ce-train-` | 0.117564 / 11528 | 0.0043186 / 31312 | ~27x |

Both bundles have a real, substantial encoding mismatch on this tensor pair.
This lines up with an explicit warning in the ground-truth export notebook
(`example1/qwen3_draft_model.ipynb`, cell 3):

```
WARNING: Since target_onnx_encodings is not provided, the encoding alignment
won't happen
```

`target_onnx_encodings` is documented as required "for encoding alignment
between Target and Draft model." The measured scale/zero-point mismatch in
both bundles is consistent with this alignment step having been skipped (or
not threaded through) during export.

**This is a stronger candidate for the shared floor than vocab-trim** because
it's structural to `core/` (applies to every EAGLE-family bundle, not just
`-iot-train-`), and both bundles — regardless of vocab-trim status — hit the
same acceptance floor.

**Status: CONFIRMED, with real on-device numbers (iot-train bundle).** Added
temporary diagnostic logging right after the `v_feat = tgt.outputBytes(...)`
read (~line 585) to dump the first 8 raw `uint16` values of `add_122687` for
the "What is gravity?" prompt, then reverted it (see "Diagnostic
instrumentation note" below for the pattern if you need it again). Captured:

```
raw add_122687 uint16 values (first 8): 11368 11069 11183 11327 11491 11624 11150 11310
```

Dequantized under each side's own `(scale, zero_point)`:

| raw | as target (correct) | as draft (actual, buggy) |
|---|---|---|
| 11368 | 5.5272 | -103.2234 |
| 11069 | -13.9156 | -104.5536 |
| 11183 | -6.5026 | -104.0464 |
| 11327 | 2.8611 | -103.4058 |
| 11491 | 13.5254 | -102.6762 |
| 11624 | 22.1739 | -102.0845 |
| 11150 | -8.6485 | -104.1932 |
| 11310 | 1.7557 | -103.4814 |

The target's real signal varies meaningfully (~36-unit range, -13.9 to
+22.2). Reinterpreted under the draft's `(scale, zero_point)` — which is what
actually happens on this `memcpy` path — the same 8 raw values collapse to a
~2.5-unit band (-102.1 to -104.6): the draft's `zero_point` (34570) sits so
far outside the raw-value range the target actually produces (~11069-11624)
that nearly the entire real signal is swallowed into a huge, almost-constant
offset. **The draft is effectively seeing a near-constant, saturated input
instead of the real conditioning feature.** This directly explains the full
observed symptom set: non-degenerate-looking draft confidence (it's
confidently reacting to *some* input, just not the real one), occasional
topN hits (a near-constant input still weakly correlates with generic
high-prior continuations), and 0% top1 (no real information about the actual
next token crosses this boundary).

This was only measured on `-iot-train-` so far; the same check on
`-ce-train-` (using its own target/draft scale+zp, see table above) should
show the same qualitative collapse — not yet executed, do this first if
picking this up.

### What the correct hidden-state hand-off actually looks like (ground truth)

**This is not a precision/bitwidth problem** — both target output and draft
input are 16-bit quantized activations (`activation_bitwidth=16` default on
both sides, confirmed `uint16` on-device for both). **It's an independent-
calibration problem**: each side computed its own `(scale, zero_point)` for
`last_hidden_states` from its own calibration data, and nothing forces them
to agree.

The ground-truth export notebook has a dedicated step for exactly this,
`example1/qwen3_draft_model.ipynb` cell 47 ("Align hidden_states encodings
across Draft and Target"), gated behind `if target_onnx_encodings:`:

1. Loads the **target model's own** post-quantization ONNX encodings file
   (`target_onnx_encodings`, produced when the target itself was quantized/
   exported — a separate, earlier step), and finds the entry named
   `"last_hidden_states"` in it.
2. Reads that entry's `bw` (bitwidth), `scale`, `offset`, and derives
   `encoding_min`/`encoding_max` from them.
3. **Force-sets the draft's hidden-state INPUT quantizer** to that exact
   range and bitwidth — `fch_Unsqueeze.input_quantizers[0]` for
   eaglet-v1/`dual_fc` (our case — `eaglet-version: 1` in both bundles), or
   `feature_fc_Unsqueeze.input_quantizers[0]` for v2 — then calls
   `allow_overwrite(False)` so the draft's own calibration pass (cell 53,
   `quantsim.compute_encodings(...)`) is **not allowed to override it**.

In other words: **the target's output encoding is the single source of
truth; the draft's input encoding must be pinned to match it exactly**, not
independently calibrated. The scale/zero_point mismatch we measured in both
bundles is exactly what you get when this cell either didn't run (no
`target_onnx_encodings` provided) or its result wasn't carried through to
the exported ctx-bin. This is why the fix in step 2 below has to pick one of
two forms — see the note there.

## How to confirm/deny #2 (the decisive next steps)

1. ~~Numeric check~~ — **done, see above.** Confirmed on `-iot-train-`;
   repeat on `-ce-train-` for completeness (same method, different
   scale/zero_point values from the table above).

2. **The decisive test (requires a real, revertable code change)**: patch
   the feature hand-off to properly dequantize using the target's own
   `(scale, zero_point)` into float, then requantize using the draft's own
   `(scale, zero_point)`, instead of the raw `memcpy`. Re-run the same
   prompts, compare `top1`/`topN`/`tokens-per-round` before/after. A
   meaningful jump toward the FP32 references (2.57 / 3.21) confirms this as
   the (or a) dominant shared cause. Little to no movement means the floor is
   coming from elsewhere (most likely w4a16 quantization of the draft head
   itself) and this mismatch, while real, isn't the main driver.

   Two different fixes are possible, pick based on where the real
   underlying problem is:
   - **Runtime-side dequant/requant** (described above) is a safe fix for
     *any* mismatched bundle, already-exported or not, but it adds a
     per-step float round-trip and never recovers precision the draft's
     encoding doesn't have room for.
   - **Re-export with cell 47's alignment actually applied** (pin the
     draft's input encoding to the target's output encoding, re-run
     `compute_encodings`/SeqMSE/export) is the ground-truth-correct fix and
     should be preferred if a re-export is feasible — it matches what the
     recipe always intended, and avoids a per-step runtime conversion. The
     runtime fix is the fallback for bundles that can't be re-exported.

3. **Isolate pure quantization-noise contribution** (only after #2 is
   fixed): compare the on-device BE against the torch FP32 baseline (2.57)
   using the same checkpoint and same prompts. Any remaining gap at that
   point is attributable to w4a16 weight quantization of the draft, not to
   the feature-crossing bug.

## How to verify the fix actually aligned hidden-state input/output

Whichever fix form is implemented (runtime dequant/requant, or re-export
with cell 47's alignment), do **all** of the following before trusting a new
acceptance-rate number — a fix that "looks plausible" but doesn't actually
align the tensors will just move the bug, not remove it:

1. **Static encoding check (no run needed).** Re-dump the quantization
   params exactly as in the table above:
   ```python
   import json
   d = json.load(open(r"<bundle>\metadata.json", encoding="utf-8"))
   d["model_files"]["part5_of_6.bin"]["outputs"]["add_122687"]["quantization_parameters"]
   d["model_files"]["draft_body.bin"]["inputs"]["hidden_states"]["quantization_parameters"]
   ```
   - If you re-exported with cell 47's alignment: `scale` and `zero_point`
     (`offset`) must now be **identical** (or within float rounding) on both
     sides. If they still differ, the alignment step did not actually run
     (check the `target_onnx_encodings` path was valid and
     `"Align encodings with target model"` was printed during export, not
     skipped).
   - If you only patched the runtime: the metadata will still show two
     different encodings — that's expected, since the fix now lives in the
     dequant/requant code, not the export. Skip to step 2 to verify the
     *runtime* actually does the conversion.

2. **Dynamic numeric check (repeat the exact method used to find the bug).**
   Re-add the same temporary diagnostic used above (dump raw values crossing
   the target→draft boundary — see "Diagnostic instrumentation note"), but
   this time dump **both** the raw target bytes and the actual values the
   draft receives after your fix runs. Recompute by hand:
   - `v_as_target = (raw - target_zp) * target_scale`
   - `v_as_draft_now = (value draft actually sees after your fix)`
   These two must now **track each other** (same shape/variance, not the
   collapsed near-constant band seen in the bug) — e.g. feed the same prompt
   used in the original capture (`"What is gravity?"`,
   `$TEMP/gravity_prompt.txt`) and confirm the draft-side values now span a
   comparable range to the target-side values (~36-unit range in the
   original capture), not the ~2.5-unit collapsed band. Revert the
   diagnostic logging afterward.

3. **End-to-end acceptance re-measurement.** Re-run the same prompt set used
   throughout this investigation (gravity + bicycle + the 8 UltraChat
   prompts — see "Repro commands") and compare `top1`/`topN`/
   `tokens-per-round` against this doc's baseline numbers (0% top1, ~1.0-1.01
   tokens/round). Expect a clear, non-marginal improvement if this was
   actually the (or a) dominant cause — a jump from 0% to single-digit-%
   top1 and tokens/round meaningfully above 1.0 is the signal to look for,
   not noise-level movement (e.g. 1.00 -> 1.02).

4. **Don't declare victory on `-iot-train-` alone.** Repeat step 3 on
   `-ce-train-` too (no vocab-trim confound there) — since both bundles
   share this bug, a real fix should move both. If only one moves, the fix
   is incomplete or something else is bundle-specific.



## Diagnostic instrumentation note

A prior session added temporary `DIAGNOSTIC (temporary)`-labeled
instrumentation to `core/include/llm/eagle_model.h` /
`core/src/llm/eagle_model.cpp` that logs per-generate-call level-0
draft/target agreement (`top1=`, `topN=`, `avg_top1_conf=`). It was reverted
to keep the working tree clean (`git diff` was empty against `origin/main`
after reverting). If you need that visibility again, it's small and
mechanical to re-add — grep this file's git history / prior session
transcripts for the exact diff, or just re-derive it: measure whether
`tree.tokens[0..n_branches)` (the anchor's level-0 proposals, always
pre-pruning-survivors) contains/equals the target's real argmax at that step.

## Repro commands

```powershell
Set-Location 'c:\Users\zhic\code\open_source_code\geniex-qairt-plugin\build\bin\Release'
.\eaglet_example.exe --model-dir 'C:\Users\zhic\models\qwen3-8b-eaglet-iot-train-w4a16-v73' `
  --raw-prompt-file <file> --max-tokens 150 --verbose
```

Genie config ctx-bins must be populated first (see "Bundles" above) — ships
empty from HF.

To inspect quantization params for any tensor:
```python
import json
d = json.load(open(r"<bundle>\metadata.json", encoding="utf-8"))
d["model_files"]["<shard>.bin"]["outputs"]["<tensor_name>"]["quantization_parameters"]
```
