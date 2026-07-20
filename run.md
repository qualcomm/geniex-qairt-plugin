# Running Gemma4-E2B inference on the QDC X2 Elite (Hexagon v81)

Two ways to run the **same** Gemma4-E2B W4A16 v81 asset on the QDC X2 Elite laptop:

1. **geniex-qairt inference** — the `gemma4_e2b.exe` example from *this* plugin (`geniex_core`).
2. **genie inference** — Qualcomm's stock `genie-t2t-run.exe` (patched Genie built against QAIRT 2.42).

Both assume you have a **PowerShell terminal open on the QDC X2 Elite box** (Snapdragon X2 Elite,
`SC8480XP` / soc_id 87 / Hexagon **v81**). If you're driving it over SSH from your host, the tunnel is:

```bash
ssh -i "C:\Users\zackli\OneDrive - Qualcomm\Documents\qdc_zack_li.pem" \
    -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
    -L 5555:localhost:3389 -p 2222 hcktest@localhost
```

> Device arch is **v81** — that is why every asset/skel is the `v81` variant. Using a v73 skel gives
> `err 5005` ("can't read future blob").

---

# geniex-qairt inference

Runs the plugin's own pipeline (`geniex_core.dll`) via the per-model example exe `gemma4_e2b.exe`.

## What you need on the device

```
<geniex-run>\
├─ gemma4_e2b.exe          # built from models/gemma4/gemma4_e2b_example.cpp
├─ geniex_core.dll         # the plugin core
├─ geniex-proc.dll         # tokenizer / preprocessing
├─ htp-files\              # QNN HTP runtime libs (bundles ALL archs incl. QnnHtpV81Stub.dll,
│                          #   libQnnHtpV81Skel.so) — copied next to geniex_core by the build
└─ modelfiles\gemma4_e2b\  # the QAIRT bundle (--model-dir), see below
```

`gemma4_e2b.exe`, `geniex_core.dll`, `geniex-proc.dll`, and `htp-files\` all come straight from the
Windows ARM64 build tree: `build\bin\Release\`. Keep them together — `geniex_core` locates the HTP
libs by looking for `htp-files\` next to itself (see [core/include/runtime.h](core/include/runtime.h)),
so **no `ADSP_LIBRARY_PATH` is needed** for this path; the skel is inside `htp-files\`.

The `--model-dir` bundle is a QAIRT export directory containing (per
[core/include/llm/llm_spec_loader.h](core/include/llm/llm_spec_loader.h)):

- the context-binary shards (`*.serialized.bin`),
- `genie_config.json` — ctx-bin ordering + dialog/RoPE/sampler,
- `metadata.json` — export metadata (shard shapes; mainly used by the VLM path),
- `tokenizer.json`,
- the embedding LUTs (`embedding_int16_lut.bin`, `embed_token_int16_lut.bin`) + their encodings.

> If you only have the stock Genie asset staged at `C:\gemma4-v81\` (the §12/§13 layout from
> `zack-export.md`), it is **not** yet a geniex `--model-dir` bundle — geniex reads
> `genie_config.json` + `metadata.json`, not `htp-model-config-gemma4.json`. Build the
> `modelfiles\gemma4_e2b\` bundle before using this path.

## Build (host: x86_64 Windows, MSVC ARM64 cross — or on the ARM64 device itself)

```powershell
cd C:\Users\zackli\zackli\qualcomm-code\geniex-qairt-plugin
cmake -B build -A ARM64 -DGENIEX_BUILD_EXAMPLES=ON
cmake --build build --config Release --target gemma4_e2b -j32
# output: build\bin\Release\gemma4_e2b.exe (+ geniex_core.dll, geniex-proc.dll, htp-files\)
```

## Run

```powershell
cd <geniex-run>          # the dir holding gemma4_e2b.exe + geniex_core.dll + htp-files\

# single-shot
.\gemma4_e2b.exe --model-dir .\modelfiles\gemma4_e2b --prompt "The capital of France is" --max-tokens 64

# with perf metrics (TTFT, decode toks/s)
.\gemma4_e2b.exe --model-dir .\modelfiles\gemma4_e2b --prompt "The capital of France is" --verbose

# interactive REPL (no --prompt): type prompts, 'exit' to quit
.\gemma4_e2b.exe --model-dir .\modelfiles\gemma4_e2b

# instruct-tuned checkpoint: apply chat template
.\gemma4_e2b.exe --model-dir .\modelfiles\gemma4_e2b --chat --prompt "Explain gravity in one sentence."
```

Flags (from [gemma4_e2b_example.cpp](models/gemma4/gemma4_e2b_example.cpp)):
`--model-dir <path>` (default `.\modelfiles\gemma4_e2b`), `--prompt <text>`, `--max-tokens <n>`
(default 256), `--chat`, `--verbose`, `--help`.

**Success signal:** `Model loaded.` then a coherent completion (first token = `Paris`). A DLL-load
failure (`0xc000007b`) means the ARM64 VC++ runtime (`msvcp140.dll`, `vcruntime140*.dll`) isn't
alongside the exe.

---

# genie inference

Runs Qualcomm's stock `genie-t2t-run.exe` against the same v81 asset, using a **patched Genie built
against QAIRT 2.42**. This is the "already staged" path — everything lives under `C:\gemma4-v81\`
(the state left by the working export; see §12–§13 of
`model-onboard\llm_notebook_ce\Gemma4\zack-export.md`).

## What's on the device (`C:\gemma4-v81\`)

```
C:\gemma4-v81\
├─ ar128_ar1_cl4096_v81\   gemma4VL_..._1_of_2.serialized.bin (288 MB) + _2_of_2 (1.04 GB)
├─ embedding_int16_lut.bin        (805 MB, main token-embedding LUT, int16)
├─ embed_token_int16_lut.bin      (4.70 GB, per-layer embedding LUT, int16)
├─ embed_encodings.json / embed_tokens_encodings.json   (LUT scale/offset)
├─ tokenizer\tokenizer.json (+ generation_config.json)
├─ bin\    Genie.dll (v1.14.0, patched, ARM64) + genie-t2t-run.exe (ARM64)
│          + 2.42 QNN DLLs (QnnHtp.dll, QnnHtpV81Stub.dll, QnnSystem.dll, …)
│          + ARM64 VC++ runtime (msvcp140.dll, vcruntime140*.dll)
├─ skel\   v81 unsigned skel (libQnnHtpV81Skel.so, …)   ← err-5005 fix
└─ run\    htp-model-config-gemma4.json, htp_backend_ext_config.json,
           prompt.txt, run_gemma4_v81.ps1
```

## Run (easiest — the launcher does PATH + skel + invocation)

```powershell
# default prompt ("Paris is the capital of France. London ... Tokyo is the capital of")
powershell -ExecutionPolicy Bypass -File C:\gemma4-v81\run\run_gemma4_v81.ps1

# + profiling (writes run\genie_profiling_stats.json)
powershell -ExecutionPolicy Bypass -File C:\gemma4-v81\run\run_gemma4_v81.ps1 -Profiling
```

**Custom prompt** — write `prompt.txt` as UTF-8 **without BOM** (a BOM corrupts the first token), then
run the launcher with no `-Prompt`:

```powershell
[System.IO.File]::WriteAllText("C:\gemma4-v81\run\prompt.txt", "The capital of France is",
                               (New-Object System.Text.UTF8Encoding($false)))
powershell -ExecutionPolicy Bypass -File C:\gemma4-v81\run\run_gemma4_v81.ps1
```

> Do **not** pass the prompt inline as `-Prompt "..."` over SSH→PowerShell — the quoting gets mangled
> (you'll see `[PROMPT]: 'The` and garbage). Always set `prompt.txt` and run without `-Prompt`.

## Run (manual — equivalent to what the launcher does)

```powershell
$env:PATH = "C:\gemma4-v81\bin;$env:PATH"       # patched Genie.dll + genie-t2t-run.exe + QNN DLLs + VC++ rt
$env:ADSP_LIBRARY_PATH = "C:\gemma4-v81\skel"   # v81 unsigned skel — REQUIRED, fixes err 5005
cd C:\gemma4-v81\run
genie-t2t-run.exe -c .\htp-model-config-gemma4.json --prompt_file .\prompt.txt
# add:  --profile .\genie_profiling_stats.json   for timing
```

## Expected output (success signals)

```
Using libGenie.so version 1.14.0
[INFO]  "Using create From Binary List Async"
[INFO]  "Allocated total size = 88113152 across 3 buffers"
[PROMPT]: The capital of France is
[BEGIN]:  Paris. Paris is a city in France. …[END]
```

- `GenieDialog_create` succeeds with no "Non-identical quantization parameters" / "Wrong input shape".
- No "T2E conversion overflow" → the patched Genie's `size_t` LUT index handles the 4.70 GB int16 LUT.
- First token is a correct word (`Paris` / `Japan` / `4`) → graph numerics are sound.
- Reference perf (`--profile`): init ~1.6 s, prompt ~237 toks/s, TTFT ~59 ms, decode ~44 toks/s.

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `positional-encoding is not an object` | wrong RoPE schema for this Genie | config must use `positional-encoding` **object** + sibling `local-positional-encoding` object (NOT the array form) |
| `inputs_embeds - Wrong input shape` | E4B dims in config | this is **E2B**: `embedding.size=1536`, `perlayer-embedding.size=8960` |
| `err 5005` / "Can't read future blob" | wrong/absent skel | `ADSP_LIBRARY_PATH` must point at `C:\gemma4-v81\skel` (v81 unsigned skel) |
| `'genie-t2t-run.exe' is not recognized` | PATH not set | run via `run_gemma4_v81.ps1`, or add `C:\gemma4-v81\bin` to PATH first |
| DLL load fails (`0xc000007b` / missing msvcp140) | VC++ runtime absent | keep `msvcp140.dll`/`vcruntime140*.dll` next to `Genie.dll` (already in `bin\`) |
| `[PROMPT]: 'The` (truncated) | inline `-Prompt` over SSH quoting | write `prompt.txt` (UTF-8, no BOM) and run without `-Prompt` |
| garbage multilingual tokens from token 1 | **wrong asset** (old CN-enabled 2.48 export) | use the re-exported v81 asset at `C:\gemma4-v81\`, not the old CN-enabled bins |
