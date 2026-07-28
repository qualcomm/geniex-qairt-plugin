# Collecting runtime-quality benchmark answers

This is the **collection** half of the workflow. You run the prompt suite
through both runtimes on a Snapdragon device, producing a results CSV + an
answers JSON, then score those with Claude Code (see
[SCORE_WITH_CLAUDE.md](SCORE_WITH_CLAUDE.md)).

The script does no scoring; it just collects raw answers, so the run can be
entirely non-interactive once kicked off.

> Because this repo *is* the runtime under test, collection normally happens on
> the same ARM64 Windows box you build on. A remote Qualcomm QDC device works
> identically — copy the repo (or just a built `auto_llm.exe` plus its DLLs, the
> bundle, and this directory) over and run the same command.

## Prerequisites

1. **A built `auto_llm.exe`.** This is the geniex side of the benchmark — the
   script shells out to it, so it must reflect the plugin code you want to
   measure. Rebuild it after every plugin change:

   ```pwsh
   cmake -B build -A ARM64          # first time only
   cmake --build build --config Release --target auto_llm -j32
   ```

   The binary lands at `build\bin\Release\auto_llm.exe`, alongside the
   `geniex_core.dll` / `geniex-proc.dll` it needs. The script auto-discovers it
   there; override with `--auto-llm <path>`.

   Sanity check: `.\build\bin\Release\auto_llm.exe --help`

2. **QAIRT SDK installed**, with `genie-t2t-run` on `PATH` and
   `ADSP_LIBRARY_PATH` pointing at the right Hexagon architecture for the
   device (v73 for X Elite, v75 for QCS8275, v81 for X2 Elite). See
   [README.md § Running genie-t2t-run by hand](README.md#running-genie-t2t-run-by-hand)
   for the exact environment setup.

   Sanity check: `genie-t2t-run --help`

   Skip this entirely with `--skip-genie` if you only want the geniex side.

3. **Python 3.10+.** Stdlib only — no third-party packages required. (If the
   `tokenizers` package happens to be installed, the script uses it to backfill
   genie's generated-token count when its profile JSON omits it. Purely
   optional.)

4. **A QAIRT export bundle** on disk. This replaces the old model-cache /
   `geniex pull` step: there is no cache in this repo, you point the script
   straight at a directory. It must contain at minimum:

   | File                          | Used by                                     |
   | ----------------------------- | ------------------------------------------- |
   | `genie_config.json`           | both (genie reads it; supplies the bos-token that picks the genie-side template) |
   | `metadata.json`               | geniex (`auto_llm` builds the LLM spec from it) |
   | `tokenizer.json`              | both                                        |
   | `tokenizer_config.json`       | geniex (the Jinja `chat_template`)          |
   | `*.bin`                       | both (compiled context-binary shards)       |
   | `htp_backend_ext_config.json` | optional                                    |
   | `embedding_weights.raw` / `embed_tokens.npy` | optional CPU-side embedding LUT |

   Bundles committed to this repo live under `modelfiles/`; anything else on
   disk works too — `--model-dir` takes any absolute or relative path.

## One-shot run

From this directory:

```powershell
$env:PYTHONIOENCODING = "utf-8"
python runtime_quality_benchmark.py --model-dir ..\modelfiles\<bundle>
```

That's the whole command. The script will:

- Verify `--model-dir` contains a `genie_config.json` and read it to pick the
  genie-side chat template (Llama-3 vs Qwen), failing fast if the directory
  isn't a bundle.
- Locate `auto_llm.exe` (under `<repo>/build/bin/`, then `$PATH`).
- Write the pending prompts to a temp `batch.json` and invoke `auto_llm` **once**
  with `--batch-file`, so the bundle is loaded a single time. `auto_llm` resets
  the KV cache between prompts and renders the bundle's own chat template for
  each, then writes every answer plus its TTFT / decode / throughput numbers to
  a temp `--metrics-json` file — rewritten after every prompt, so a crash
  partway still yields the completed answers.
- Iterate every prompt in `testing_prompts.md`, pairing each geniex answer with
  a fresh `genie-t2t-run` subprocess fed the hand-built formatted prompt.
- Persist the running results to `results/<slug>.csv` after **every** prompt so
  a crashed or interrupted run can be resumed.
- After the last prompt, also write `results/<slug>.answers.json` — the file the
  scoring agent reads.

`<slug>` is the bundle directory's name, lowercased, with `-` and `.` turned
into `_`. Examples:

| `--model-dir`                          | `<slug>`                  |
| -------------------------------------- | ------------------------- |
| `..\modelfiles\Qwen3-4B-Instruct-2507` | `qwen3_4b_instruct_2507`  |
| `..\modelfiles\gemma4_e2b_nonqat`      | `gemma4_e2b_nonqat`       |

Override it with `--slug` (handy when benchmarking the same bundle across
plugin revisions) or replace the whole path with `--out`.

Expected wall-clock: ~30 minutes for a 1B model, ~2 hours for a 4B model on
Snapdragon X Elite (NPU). By default both runtimes generate until their natural
EOS / context-exhaustion stop (no token cap); the prompt suite has 100 prompts.
Pass `--max-tokens N` to truncate the geniex side for quick smoke tests.

Smoke-test before committing to a full run:

```powershell
python runtime_quality_benchmark.py --model-dir ..\modelfiles\<bundle> `
    --limit 2 --max-tokens 40 --skip-genie --out $env:TEMP\smoke.csv
```

## Useful flags

| flag                 | default                                 | what it does                                                           |
| -------------------- | --------------------------------------- | ---------------------------------------------------------------------- |
| `--model-dir`        | _(required)_                            | QAIRT export bundle directory. Both runtimes read this same directory.  |
| `--prompts`          | `testing_prompts.md` next to the script | Different prompt suite. Same markdown shape as the bundled file.        |
| `--auto-llm`         | auto-discovered under `<repo>/build/bin/`, then `$PATH` | Path to the built `auto_llm` executable.                |
| `--tokenizer-config` | `<model-dir>/tokenizer_config.json`     | Override where `auto_llm` reads the Jinja chat template from.           |
| `--enable-thinking`  | off                                     | Pass `--enable-thinking` to `auto_llm`, plumbing `{"enable_thinking":true}` into the Jinja context. No-op on templates that don't read the field. |
| `--slug`             | the bundle directory name               | Override the `<slug>` used to name output files.                       |
| `--out`              | `results/<slug>.csv`                    | Override the CSV path (the answers.json is always derived from it).     |
| `--max-tokens`       | 0 (no cap)                              | Cap on geniex tokens per prompt. `0` = run to natural EOS / context stop, like genie. Set positive to truncate. |
| `--timeout`          | 600                                     | Per-prompt timeout in seconds for the `genie-t2t-run` subprocess. The geniex side is one batch process, so its budget is `(timeout x prompt count) + 900s` of slack for the model load. |
| `--limit N`          | 0 (all)                                 | Run only the first N prompts — handy for smoke tests.                  |
| `--resume`           | off                                     | Skip prompts whose `id` already appears in `--out`. Use to continue.    |
| `--skip-genie`       | off                                     | Skip the genie-t2t-run pass; only run `auto_llm`. Default `--out` becomes `results/<slug>_geniex_only.csv`. See "Geniex-only re-runs" below. |
| `--stream-geniex`    | off                                     | Let `auto_llm` stream its answers + debug logs to this console instead of running `--quiet`. Handy for watching a live run. |

## Chat templates

The two runtimes reach the same formatted prompt by different routes, and this
is the part most likely to silently diverge — check it first if answers look
wrong on one side only.

- **geniex side.** `auto_llm` renders the bundle's own Jinja `chat_template`
  from `tokenizer_config.json`, over a `[system, user]` message list. The system
  message comes from `--system`, which the script sets to
  `"You are a helpful AI assistant."`
- **genie side.** `genie-t2t-run` does no templating and takes a fully-formatted
  string, so the script hand-builds one from the `TEMPLATES` table at the top of
  `runtime_quality_benchmark.py`, with the same system message baked in. The
  family is picked from `genie_config.json`'s `dialog.context.bos-token`
  (`128000` → llama3, anything else → qwen).

If you benchmark a bundle whose family isn't covered, add an entry to
`TEMPLATES` / `detect_template`. A mismatch shows up as garbled genie answers
with visible `<|im_start|>` / `<|begin_of_text|>` markers.

## Geniex-only re-runs

When **only the plugin has changed** (the bundle on disk and the
`genie-t2t-run` binary are the same as last time), there's no point re-running
the genie side: for a given (bundle, prompt, template), `genie-t2t-run` is
deterministic, so its answers will be byte-identical to the prior full run.
Pass `--skip-genie` to drop the genie pass entirely:

```powershell
$env:PYTHONIOENCODING = "utf-8"
python runtime_quality_benchmark.py --model-dir ..\modelfiles\<bundle> --skip-genie
```

Remember to rebuild `auto_llm` first — otherwise you re-measure the old plugin.

Behavior changes vs. a full run:

- **No `genie-t2t-run` subprocess is spawned.** Roughly halves wall-clock.
- **Default `--out` switches** to `results/<slug>_geniex_only.csv` so the prior
  full-run CSV (`results/<slug>.csv`) is left untouched.
- **The `genie_*` columns** in the new CSV (`genie_answer`, `genie_ttft_ms`,
  `genie_tps`, `genie_error`) are written **empty** — the scoring agent
  backfills them from the prior `<slug>.csv` at score time.
- **The companion `<slug>_geniex_only.answers.json`** carries a
  `{"meta": {"skip_genie": true, ...}, "rows": [...]}` wrapper instead of the
  bare list a full run produces, so downstream tooling can tell the two shapes
  apart.

When you hand the results to the scoring agent, give it both the new
`<slug>_geniex_only.answers.json` and the prior full run's
`<slug>.answers.json`. The agent scores `geniex_answer` from the new file and
`genie_answer` from the prior file (matched by `id`); see
[SCORE_WITH_CLAUDE.md § Shape B](SCORE_WITH_CLAUDE.md#shape-b--geniex-only-re-run-paired-with-a-prior-full-run).

## Resumption

Because the CSV is rewritten after every prompt, an interrupted run can just be
restarted with the same arguments plus `--resume`:

```powershell
$env:PYTHONIOENCODING = "utf-8"
python runtime_quality_benchmark.py --model-dir ..\modelfiles\<bundle> --resume
```

Only the still-missing prompts go into the new `auto_llm` batch, so the resumed
run pays the model load once more but re-generates nothing. The first time the
answers.json is written is when the run finishes, so if you crash mid-run the
CSV is the source of truth — `--resume` will fill in the missing rows and emit a
fresh answers.json once everything is done.

## Files to keep for scoring

After a successful run, only two files are needed:

```
results/<slug>.csv               # full row data (answers, perf, errors)
results/<slug>.answers.json      # 5-field-per-row payload the scoring agent reads
```

The `.csv` is the one that gets edited during scoring (see
SCORE_WITH_CLAUDE.md); the `.answers.json` is what the agent ingests, so keep
both around. If you collected on a remote device, these two files are all you
need to copy back.

## Performance columns

Besides answers, the CSV records per-runtime, per-prompt metrics:

| column                 | meaning                                                   |
| ---------------------- | --------------------------------------------------------- |
| `*_ttft_ms`            | time to first token, in milliseconds                      |
| `*_tps`                | legacy column: decode-only on the geniex side, genie's reported rate on the genie side. Kept so historical CSVs still load. |
| `*_tps_decode`         | decode-only throughput — the apples-to-apples pair        |
| `*_tps_with_ttft`      | throughput including TTFT: `N / (ttft + decode)`           |
| `*_generated_tokens`   | tokens generated                                           |
| `*_decode_ms`          | decode-phase wall time                                     |

- **geniex** values come from `auto_llm`'s `--metrics-json` output, which are
  the pipeline's own `GenerateResult` numbers (`ttft_ms`, `decode_ms`,
  `tokens_per_second`, `generated_tokens`).
- **genie** values come from the JSON file genie-t2t-run writes via
  `--profile <file>`. The script points it at a fresh temp file per prompt (in a
  throwaway temp dir, since genie-t2t-run refuses to overwrite an existing
  profile path) and reads the `GenieDialog_query` event's `time-to-first-token`
  (→ ms) and `token-generation-rate` (toks/sec). If the profile file is missing
  or malformed the cells are left blank rather than guessed.

Both are reported by the runtimes themselves, so they isolate model performance
from per-prompt overhead (e.g. genie's process startup, or the one-time bundle
load on the geniex side). Note this means the geniex batch's wall time is *not*
in the CSV — only the per-prompt runtime metrics are.

## Troubleshooting

- **`could not find a built auto_llm executable`** — build it
  (`cmake --build build --config Release --target auto_llm`) or pass
  `--auto-llm <path>`. If you copied the binary to another machine, copy
  `geniex_core.dll`, `geniex-proc.dll` and the `htp-files/` directory next to
  it too.
- **`missing <dir>/genie_config.json — is this a QAIRT export bundle?`** —
  `--model-dir` points somewhere that isn't a bundle. Check for the files listed
  in prerequisite 4.
- **`auto_llm exit <n>` / `WARNING: auto_llm returned no result for N prompt(s)`**
  — the batch process died. The script keeps whatever answers `auto_llm` had
  already written and marks the rest with the failure reason, so re-run with
  `--resume` after fixing the cause. Reproduce interactively with
  `--stream-geniex`, or run `auto_llm` by hand on the failing prompt:
  `auto_llm.exe --model-dir <dir> --prompt "<text>" --verbose`.
- **`auto_llm timeout`** — the whole batch exceeded
  `(--timeout x prompts) + 900s`. Raise `--timeout`, or split the run with
  `--limit` + `--resume`.
- **Garbled geniex output** — the bundle's own chat template is being applied,
  so a garbled *geniex* answer usually means the bundle itself is bad (wrong
  quantization, mismatched context binaries) rather than a template problem.
  Confirm by running `auto_llm` interactively on the same bundle; if the REPL
  is equally garbled, it's the bundle. Check the plugin's own debug logs, which
  `--stream-geniex` surfaces.
- **Garbled genie output starting with `<|im_start|>` or `<|begin_of_text|>`** —
  the genie-side hand-built template is wrong for this family. See
  [Chat templates](#chat-templates).
- **Answers differ hugely between runtimes on a QAIRT-version bump** — the
  context binaries in the bundle were built against a different QAIRT version
  than the plugin is linked against. `genie-t2t-run` is more lenient about
  version skew, so this shows up as geniex-only failures. Rebuild the bundle or
  run against a matching QAIRT.
- **UnicodeEncodeError on Windows** — the `$env:PYTHONIOENCODING = "utf-8"` line
  in the example commands is load-bearing; without it Python's default cp1252
  stdout will choke on emoji/CJK in model output.
