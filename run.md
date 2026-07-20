# Running Gemma4-E2B on the QDC X2 Elite — device-local runbook

Everything below runs **entirely on the QDC X2 Elite box**, from its own PowerShell terminal. You do
**not** touch a host machine: SSH into the device once, then run every command locally with the
`C:\...` paths shown. No building, no copying, no host-side ssh.

Device: Snapdragon **X2 Elite** (`SC8480XP`, soc_id **87**, Hexagon **v81**). All assets/skels are the
`v81` variant — a v73 skel gives `err 5005` ("can't read future blob").

> **The device has no build toolchain** (no cmake / Rust / MSVC / git). You cannot compile here.
> Everything needed to *run* is already staged as prebuilt ARM64 binaries + assets (below). Source
> trees under `C:\src\` are for **reading only**.

Two independent, already-staged ways to run the same Gemma4-E2B W4A16 v81 model:

| Path | Binaries + assets | Config style |
|------|-------------------|--------------|
| **geniex-qairt** (this plugin) | `C:\geniex-gemma4\` | `genie_config.json` + `metadata.json` bundle |
| **genie** (stock `genie-t2t-run.exe`) | `C:\gemma4-v81\` | `htp-model-config-gemma4.json` |

## On-device map

```
C:\geniex-gemma4\                     ← geniex-qairt run tree
├─ bin\  gemma4_e2b.exe, geniex_core.dll, geniex-proc.dll,
│        VC++ rt (msvcp140*.dll, vcruntime140*.dll, concrt140.dll),
│        htp-files\  (all QNN HTP libs incl. QnnHtpV81Stub.dll + libQnnHtpV81Skel.so)
└─ modelfiles\gemma4_e2b\   genie_config.json, metadata.json, tokenizer.json,
                            tokenizer_config.json, chat_template.jinja, generation_config.json,
                            gemma4VL_..._{1,2}_of_2.serialized.bin,
                            embedding_fp32.bin, per_layer_fp32.bin

C:\gemma4-v81\                        ← genie run tree
├─ bin\   Genie.dll (v1.14.0, patched, ARM64) + genie-t2t-run.exe + 2.42 QNN DLLs + VC++ rt
├─ skel\  v81 unsigned skel (libQnnHtpV81Skel.so, …)   ← err-5005 fix
├─ run\   run_gemma4_v81.ps1, htp-model-config-gemma4.json, htp_backend_ext_config.json, prompt.txt
├─ ar128_ar1_cl4096_v81\  gemma4VL_..._{1,2}_of_2.serialized.bin
├─ embedding_int16_lut.bin (805 MB) + embed_token_int16_lut.bin (4.70 GB) + *_encodings.json
└─ tokenizer\  tokenizer.json (+ generation_config.json)

C:\src\                               ← SOURCE, reference only (cannot build on device)
├─ geniex-qairt-plugin\   this repo (no build/.git/third-party)
├─ example3\              patched Genie source + .diff files + genie_config
└─ genie-build-242\       build.bat / build_t2t.bat / rebuild.bat (used on the HOST to build Genie)
```

---

# 1. geniex-qairt inference

SSH into the device, then:

```powershell
cd C:\geniex-gemma4\bin

# single-shot with perf metrics (TTFT, decode toks/s)
.\gemma4_e2b.exe --model-dir C:\geniex-gemma4\modelfiles\gemma4_e2b --prompt "The capital of France is" --verbose

# limit length
.\gemma4_e2b.exe --model-dir C:\geniex-gemma4\modelfiles\gemma4_e2b --prompt "Who are you?" --max-tokens 64

# interactive REPL (no --prompt): type prompts, 'exit' to quit
.\gemma4_e2b.exe --model-dir C:\geniex-gemma4\modelfiles\gemma4_e2b

# instruct-tuned checkpoint: apply the chat template
.\gemma4_e2b.exe --model-dir C:\geniex-gemma4\modelfiles\gemma4_e2b --chat --prompt "Explain me Newton's 1st law of motion."
```

Notes:
- Run from `C:\geniex-gemma4\bin` (or put it on PATH) so `geniex_core.dll`, `geniex-proc.dll`, and the
  VC++ runtime resolve. `geniex_core` finds the HTP libs (incl. the v81 skel) inside `htp-files\` next
  to itself — **no `ADSP_LIBRARY_PATH` needed** for this path.
- Flags: `--model-dir <path>` (default `.\modelfiles\gemma4_e2b`), `--prompt <text>`, `--max-tokens <n>`
  (default 256), `--chat`, `--verbose`, `--help`.
- **Success:** prints `Model loaded.` then a coherent completion (first token `Paris`).

---

# 2. genie inference

SSH into the device, then use the launcher (it sets PATH + `ADSP_LIBRARY_PATH` and runs the exe):

```powershell
# default prompt ("Paris is the capital of France. London ... Tokyo is the capital of")
powershell -ExecutionPolicy Bypass -File C:\gemma4-v81\run\run_gemma4_v81.ps1

# + profiling (writes + prints C:\gemma4-v81\run\genie_profiling_stats.json)
powershell -ExecutionPolicy Bypass -File C:\gemma4-v81\run\run_gemma4_v81.ps1 -Profiling
```

The launcher deletes any stale `genie_profiling_stats.json` before each profiled run (`genie-t2t-run`
refuses `--profile` if the output file already exists — `ERROR: … already exists`) and prints the JSON
when done. Key fields: `prompt-processing-rate`, `time-to-first-token`, `token-generation-rate`,
`num-generated-tokens`, and the `GenieDialog_create` init time.

**Custom prompt** — write `prompt.txt` as UTF-8 **without BOM** (a BOM corrupts the first token), then
run the launcher with no `-Prompt`:

```powershell
[System.IO.File]::WriteAllText("C:\gemma4-v81\run\prompt.txt", "The capital of France is",
                               (New-Object System.Text.UTF8Encoding($false)))
powershell -ExecutionPolicy Bypass -File C:\gemma4-v81\run\run_gemma4_v81.ps1
```

### Manual (equivalent to the launcher)

```powershell
$env:PATH = "C:\gemma4-v81\bin;$env:PATH"       # patched Genie.dll + genie-t2t-run.exe + QNN DLLs + VC++ rt
$env:ADSP_LIBRARY_PATH = "C:\gemma4-v81\skel"   # v81 unsigned skel — REQUIRED, fixes err 5005
cd C:\gemma4-v81\run
genie-t2t-run.exe -c .\htp-model-config-gemma4.json --prompt_file .\prompt.txt
# timing: delete any old stats first (the exe won't overwrite), then pass --profile
if (Test-Path .\genie_profiling_stats.json) { Remove-Item .\genie_profiling_stats.json -Force }
genie-t2t-run.exe -c .\htp-model-config-gemma4.json --prompt_file .\prompt.txt --profile .\genie_profiling_stats.json
```

### Expected output

```
Using libGenie.so version 1.14.0
[INFO]  "Using create From Binary List Async"
[INFO]  "Allocated total size = 88113152 across 3 buffers"
[PROMPT]: The capital of France is
[BEGIN]:  Paris. Paris is a city in France. …[END]
```

Reference perf (`--profile`, measured on-device): init ~1.6 s, prompt ~105 toks/s, TTFT ~57 ms,
decode ~49 toks/s (rates vary with prompt/generation length).

---

# Troubleshooting (both paths)

| Symptom | Cause | Fix |
|---------|-------|-----|
| `positional-encoding is not an object` (genie) | wrong RoPE schema for this Genie | config must use `positional-encoding` **object** + sibling `local-positional-encoding` object (NOT the array form) |
| `inputs_embeds - Wrong input shape` | E4B dims in config | this is **E2B**: `embedding.size=1536`, `perlayer-embedding.size=8960` |
| `err 5005` / "Can't read future blob" | wrong/absent skel | genie: `ADSP_LIBRARY_PATH=C:\gemma4-v81\skel`; geniex: use the bundled `htp-files\` (has the v81 skel) |
| `'…exe' is not recognized` | PATH not set | genie: use `run_gemma4_v81.ps1`; geniex: `cd C:\geniex-gemma4\bin` first |
| DLL load fails (`0xc000007b` / missing msvcp140) | VC++ runtime absent | keep `msvcp140.dll`/`vcruntime140*.dll` next to the exe (already bundled in both `bin\`) |
| `[PROMPT]: 'The` (truncated, genie) | inline `-Prompt` over SSH quoting | write `prompt.txt` (UTF-8, no BOM) and run without `-Prompt` |
| garbage multilingual tokens from token 1 | **wrong asset** (old CN-enabled 2.48 export) | use the staged v81 assets (`C:\gemma4-v81\` / `C:\geniex-gemma4\`), not any old CN-enabled bins |

---

# If the box was re-imaged (assets missing) — restage from host

Only needed if `C:\geniex-gemma4\` / `C:\gemma4-v81\` are gone (they are **not** rebuildable on the
device — no toolchain). From the host, over the tunnel:

```bash
ssh -i "C:\Users\zackli\OneDrive - Qualcomm\Documents\qdc_zack_li.pem" \
    -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
    -L 5555:localhost:3389 -p 2222 hcktest@localhost
# scp the built bin\ + skel\ + assets from the host (qualcomm-code\gemma4-v81\, genie-build-242\deploy\,
# and the geniex build\bin\Release\ tree). See model-onboard\llm_notebook_ce\Gemma4\zack-export.md §12.
```

---

# What each asset is

Gemma4-**E2B** is a *split decoder*: one logical model compiled into **2 QNN context shards**
(`_1_of_2`, `_2_of_2`) plus a handful of side files. The two run trees carry the same model with two
different asset *encodings* (genie keeps the embedding LUTs int16; geniex pre-dequantizes them to fp32).

| Asset | Path(s) | What it is |
|-------|---------|------------|
| **`*_1_of_2.serialized.bin` / `*_2_of_2.serialized.bin`** | `ar128_ar1_cl4096_v81\` (genie) · `modelfiles\gemma4_e2b\` (geniex) | The compiled model — every transformer block (attention + MLP), the final norm and the LM head — serialized as **QNN HTP context binaries** for Hexagon **v81**. Weight-shared across the two AR modes (`ar128` prefill, `ar1` decode) at `cl4096` context. **This is the only thing that runs on the NPU.** Weights are **4-bit (W4)**, compute activations **16-bit (A16)**. |
| **`embedding_int16_lut.bin` (805 MB)** / **`embedding_fp32.bin`** | genie / geniex | **Main token-embedding table** (`inputs_embeds`). `vocab 262144 × hidden 1536`. genie stores it **uint16** (`ufixed16`, 2 B/elem → 805 MB); geniex stores the same values pre-dequantized to **float32** (4 B/elem → ~1.6 GB). A plain **lookup table**, not a matmul. |
| **`embed_token_int16_lut.bin` (4.70 GB)** / **`per_layer_fp32.bin`** | genie / geniex | **Per-layer embedding table** (`per_layer_inputs`) — a Gemma3/4-specific *second* embedding stream injected into every block. `vocab 262144 × 8960` (= layers × per-layer-dim). Same dtype story: genie **uint16** (4.70 GB), geniex **float32**. |
| **`embed_encodings.json` / `embed_tokens_encodings.json`** | genie | Quant params (`bw:16`, per-tensor `scale`+`offset`) for the two LUTs above — used to dequantize uint16 → float at load. |
| **`genie_config.json` / `htp-model-config-gemma4.json`** | geniex / genie | Runtime + model config: context size 4096, vocab 262144, BOS/EOS/pad, sampler, the two RoPE blocks, the KV `cache-groups`, and the `embedding`/`perlayer-embedding` LUT declarations. |
| **`htp_backend_ext_config.json`** | both | HTP backend knobs: `soc_id 87`, `dsp_arch v81`, `burst` perf, `unsigned` PD session, shared-buffer memory. |
| **`metadata.json`** | geniex | Just the bundle's `model_id` (`gemma4_e2b`) — geniex uses it to pick the model family. |
| **`tokenizer.json` / `tokenizer_config.json`** | both | The SentencePiece/BPE tokenizer (262144-token vocab). CPU-side, before/after the graph. |
| **`chat_template.jinja` / `generation_config.json`** | geniex | Chat formatting (used with `--chat`) and default sampling params. |
| **`bin\` (Genie.dll / gemma4_e2b.exe + QNN DLLs + VC++ rt)** | both | The runtime executables/libraries. Not model data. |
| **`skel\` (`libQnnHtpV81Skel.so`, …)** | genie | The **v81 Hexagon skel** the NPU actually executes — the err-5005 fix. (geniex ships its copy inside `bin\htp-files\`.) |

## Embedding — 16-bit or 32-bit? CPU or NPU?

**The embedding is a 16-bit quantized lookup table, evaluated on the CPU.** Details:

- **Stored precision:** **16-bit unsigned fixed-point** (`ufixed16`, i.e. `uint16`), `bw = 16`, asymmetric
  per-tensor quant — one `scale` + `offset` for the whole table (see `genie_config.json` /
  `embed_encodings.json`). The byte sizes prove it: `262144 × 1536 × 2 B = 805 MB` and
  `262144 × 8960 × 2 B = 4.70 GB`.
- **The two run trees differ only in on-disk storage, not in numeric values:**
  - **genie** keeps the LUT as `uint16` and dequantizes to float **on the CPU at lookup time** (using the
    `scale`/`offset`).
  - **geniex** ships a LUT already dequantized to **float32** on disk (`*_fp32.bin`, ~2× the size) — same
    numbers, no runtime dequant. This is why the geniex LUTs are bigger.
- **Where it runs:** **CPU, always.** Embedding is a *gather* (row lookup by token id) over a 262k-row
  table (0.8–4.7 GB), which is memory-bound, not compute-bound. Keeping it off the NPU avoids loading a
  multi-GB table into Hexagon memory and lets both run paths share the same compiled graphs. The looked-up
  hidden vector is then handed to the NPU as the **A16 activation `inputs_embeds`** (and the per-layer
  stream as `per_layer_inputs`).
- **So:** the *trained* embedding is **16-bit** regardless of path; geniex merely *carries* it as 32-bit
  floats on disk for convenience. Either way it's a CPU table lookup, not an NPU matmul.

## KV cache and everything else

- **KV cache — 16-bit.** This is a **W4A16** export: weights 4-bit, activations 16-bit. The KV cache *is*
  a stored activation, so it lives at **A16 (16-bit fixed-point)** inside the compiled graph on the
  **NPU** — it is not a separate configurable dtype in these configs, it follows the A16 activation
  scheme. Gemma4 keeps **two** caches: the **global** cache (`past_*`, `kv-dim 512`) for full-attention
  layers and a **sliding-window** cache (`swa_*`, `kv-dim 256`, `window-size 512`) for local-attention
  layers (see the `cache-groups` in `htp-model-config-gemma4.json`).
- **All transformer layers (attention Q/K/V/O + MLP + norms + LM head)** run on the **NPU** from the
  `serialized.bin` shards: **4-bit weights, 16-bit activations**.
- **CPU-side helpers fed into the graph each step** (computed in **float32**): RoPE `cos`/`sin` (both the
  global partial-rotary θ=1e6 and the local full-rotary θ=1e4 tables), `position_ids`, and the causal +
  sliding-window `attention_mask`s. These are small and control-flow-y, so they stay on the CPU.

## Quantization & inference-processor summary

| Component | Bit-width / dtype | Quant scheme | Runs on |
|-----------|-------------------|--------------|---------|
| Transformer weights (attn + MLP + LM head) | **4-bit** (W4) | per-channel weight quant | **NPU** (HTP v81) |
| Compute activations (matmul I/O) | **16-bit** (A16) | fixed-point | **NPU** |
| **KV cache** — global (`past_*`) & sliding-window (`swa_*`) | **16-bit** (A16) | fixed-point activation | **NPU** |
| **Token embedding LUT** (`inputs_embeds`, 1536-d) | **16-bit** `ufixed16` (genie) · **fp32** on-disk (geniex) | per-tensor `scale`+`offset` | **CPU** (gather + dequant) |
| **Per-layer embedding LUT** (`per_layer_inputs`, 8960-d) | **16-bit** `ufixed16` (genie) · **fp32** on-disk (geniex) | per-tensor `scale`+`offset` | **CPU** (gather + dequant) |
| RoPE cos/sin, position_ids, attention masks | **fp32** (int32 ids) | — (not quantized) | **CPU** |
| Tokenizer (encode/decode) | — | — | **CPU** |
| Sampler (temp/top-k/top-p over logits) | fp32 logits | — | **CPU** |

> **In one line:** the model body (weights 4-bit, activations + KV cache 16-bit) runs on the **NPU** from
> the two `serialized.bin` shards; the embeddings are **16-bit** lookup tables evaluated on the **CPU**
> (geniex just stores them as fp32 on disk); RoPE/masks/tokenizer/sampler are CPU float helpers.
