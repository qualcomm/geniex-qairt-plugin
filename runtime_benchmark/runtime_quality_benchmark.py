"""Run a prompt suite through both the GenieX QAIRT plugin and `genie-t2t-run`
on the same model bundle, then write the answers to a CSV ready for scoring.

Both runtimes are driven as subprocesses over the same QAIRT export bundle:

- **geniex** — this repo's `auto_llm` example (`build/bin/Release/auto_llm.exe`),
  invoked once in `--batch-file` mode so the multi-GB bundle is loaded a single
  time and reused across every prompt, with the KV cache reset between prompts
  so each answer is independent. `auto_llm` applies the bundle's own Jinja chat
  template (from `tokenizer_config.json`) and reports its own TTFT / decode /
  throughput numbers via `--metrics-json`.
- **genie-t2t-run** — Qualcomm QAIRT SDK reference runner. One subprocess per
  prompt (it has no batch mode), fed a hand-built formatted prompt because it
  does no chat templating of its own.

Same bundle, same prompts, same chat template — so quality differences reflect
the runtime, not the inputs.

Usage (minimal — runs the bundled prompt suite, writes results into ./results/):
    python runtime_quality_benchmark.py --model-dir ../modelfiles/<bundle>

Usage (full):
    python runtime_quality_benchmark.py \
        --model-dir ../modelfiles/<bundle> \
        --prompts testing_prompts.md \
        --auto-llm ../build/bin/Release/auto_llm.exe \
        --max-tokens 512 \
        --out results/my_run.csv

Usage (geniex-only re-run, when only the plugin code changed):
    python runtime_quality_benchmark.py --model-dir ../modelfiles/<bundle> --skip-genie

`--skip-genie` skips the genie-t2t-run pass entirely — useful when the plugin
has changed but the model bundle under test hasn't, since genie's answers for
the same bundle are deterministic across runs. The geniex-only output is written
to results/<slug>_geniex_only.csv by default to avoid clobbering a prior full
run; pair the two CSVs at scoring time (see SCORE_WITH_CLAUDE.md).

`<slug>` defaults to the bundle directory's name (lowercased, `-`/`.` → `_`);
override the whole output path with --out.

The script does NOT score answers — scoring is a separate pass (see
SCORE_WITH_CLAUDE.md). Once the run finishes, the script also writes a
companion <out>.answers.json that the scoring agent consumes.

Requires a built `auto_llm.exe` (see README.md § Build) for the geniex side,
and the `genie-t2t-run` CLI on PATH for the genie side (skip it with
--skip-genie). Python 3.10+, stdlib only.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path


# ---------------------------------------------------------------------------
# Prompt parsing
# ---------------------------------------------------------------------------

CATEGORY_RE = re.compile(r"^\s*#\s*---\s*(.+?)\s*---\s*$")
PROMPT_RE = re.compile(r"^\s*-\s+(.*?)\s*$")


@dataclass
class Prompt:
    id: int
    category: str
    text: str


def load_prompts(path: Path) -> list[Prompt]:
    prompts: list[Prompt] = []
    category = ""
    pid = 0
    for raw in path.read_text(encoding="utf-8").splitlines():
        cat_m = CATEGORY_RE.match(raw)
        if cat_m:
            category = cat_m.group(1).strip()
            continue
        if raw.lstrip().startswith("#"):
            continue
        m = PROMPT_RE.match(raw)
        if not m:
            continue
        text = m.group(1).strip()
        # Strip surrounding quotes if present (the source file uses them
        # whenever a bullet contains a comma or apostrophe).
        if (text.startswith('"') and text.endswith('"')) or (
            text.startswith("'") and text.endswith("'")
        ):
            text = text[1:-1]
        pid += 1
        prompts.append(Prompt(id=pid, category=category, text=text))
    return prompts


# ---------------------------------------------------------------------------
# Chat templates
# ---------------------------------------------------------------------------

# System prompt shared by both runtimes. Passed to auto_llm via --system, and
# baked into the hand-built genie templates below, so the geniex side (which
# applies the bundle's real Jinja chat template) and the genie side (which does
# no templating) see the same system message.
SYSTEM_PROMPT = "You are a helpful AI assistant."

# Only genie-t2t-run needs a hand-built prompt: it does NO chat templating and
# takes a fully-formatted string. The geniex side does NOT use these — auto_llm
# renders the bundle's own `chat_template` from tokenizer_config.json. We detect
# the model family from genie_config.json's bos-token to pick the genie template.

TEMPLATES = {
    "qwen": (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        "<|im_start|>user\n{prompt}<|im_end|>\n"
        "<|im_start|>assistant\n"
    ),
    "llama3": (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        f"{SYSTEM_PROMPT}<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
    ),
}


def detect_template(genie_config: dict) -> str:
    bos = genie_config.get("dialog", {}).get("context", {}).get("bos-token")
    # Qwen3 / Qwen2 uses 151643. Llama 3 uses 128000.
    if bos == 128000:
        return "llama3"
    return "qwen"


# ---------------------------------------------------------------------------
# Output cleaning
# ---------------------------------------------------------------------------

ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]|\x1b\[\?[0-9;]*[a-z]")


def strip_ansi(text: str) -> str:
    return ANSI_RE.sub("", text)


def clean_genie(text: str) -> str:
    """Extract the assistant response from genie-t2t-run output.

    genie-t2t-run prints headers, then `[BEGIN]: <answer>[END]`. The answer can
    span many lines (the [BEGIN]/[END] markers literally bracket it).
    """
    text = strip_ansi(text)
    m = re.search(r"\[BEGIN\]:(.*?)\[END\]", text, flags=re.DOTALL)
    if m:
        return m.group(1).strip()
    # No [END] (model truncated by max-tokens or context exhaustion).
    m = re.search(r"\[BEGIN\]:(.*)$", text, flags=re.DOTALL)
    if m:
        return m.group(1).strip()
    return text.strip()


# genie-t2t-run writes profiling data to the file given by `--profile FILE` as
# JSON (artifact_type "GENIE_PROFILE"). The per-query metrics live in the
# `GenieDialog_query` event of the `dialog` component:
#
#   "time-to-first-token":  {"value": 155464, "unit": "us"}
#   "token-generation-rate":{"value": 13.07,  "unit": "toks/sec"}
#   "num-tokens-generated": {"value": 96}             (when present)
#
# We read those, normalise TTFT to milliseconds, and (when the token count is
# available) derive a decode-only TPS that can be compared apples-to-apples
# with the geniex side, which reports decode-only TPS by construction.


def _us_value_to_ms(field: dict | None) -> float | None:
    """Convert a {'value': N, 'unit': 'us'|'ms'|'s'} profile field to ms."""
    if not isinstance(field, dict) or "value" not in field:
        return None
    value = float(field["value"])
    unit = str(field.get("unit", "us")).lower()
    if unit == "us":
        return value / 1000.0
    if unit == "s":
        return value * 1000.0
    return value  # already ms (or unknown — leave as-is)


def _value_to_int(field: dict | None) -> int | None:
    if not isinstance(field, dict) or "value" not in field:
        return None
    try:
        return int(field["value"])
    except (TypeError, ValueError):
        return None


@dataclass
class GenieProfile:
    ttft_ms: float | None = None
    # Genie's reported rate. Empirically this is decode/decode-time on a
    # warm path but can include some constant cost on short answers — we
    # surface it as-is and compute a derived decode-only TPS alongside.
    tps_reported: float | None = None
    generated_tokens: int | None = None
    # Wall-clock total time for the GenieDialog_query event, when the
    # profile carries it. Used to back out decode time = total - ttft.
    total_ms: float | None = None


def parse_genie_profile(profile_path: Path) -> GenieProfile:
    """Read the GenieDialog_query event from a genie-t2t-run --profile JSON.

    Pulls TTFT (→ ms), token-generation-rate (as Genie reports it),
    num-tokens-generated, and (if the event carries a duration field) total
    wall time. Missing fields stay ``None`` so a partial profile degrades
    gracefully instead of aborting the run."""
    out = GenieProfile()
    try:
        data = json.loads(profile_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return out
    for component in data.get("components", []):
        for event in component.get("events", []):
            if event.get("type") != "GenieDialog_query":
                continue
            out.ttft_ms = _us_value_to_ms(event.get("time-to-first-token"))
            rate = event.get("token-generation-rate")
            if isinstance(rate, dict) and "value" in rate:
                out.tps_reported = float(rate["value"])
            out.generated_tokens = _value_to_int(event.get("num-tokens-generated"))
            # The event itself usually carries a duration via either
            # "duration"/"total-time"/"value" — try a few common names.
            for key in ("duration", "total-time", "value"):
                ms = _us_value_to_ms(event.get(key)) if isinstance(event.get(key), dict) else None
                if ms is not None:
                    out.total_ms = ms
                    break
            return out
    return out


def derive_tps(generated_tokens: int | None, ttft_ms: float | None,
               total_ms: float | None, decode_ms: float | None) -> tuple[float | None, float | None]:
    """Return (tps_decode_only, tps_with_ttft).

    tps_decode_only = (N - 1) / decode_ms — matches geniex's native metric.
    tps_with_ttft   = N / (ttft + decode_ms) — matches what genie-t2t-run's
    `token-generation-rate` reports on short answers.

    Either is None when the inputs needed to compute it aren't available."""
    n = generated_tokens or 0
    if decode_ms is None and total_ms is not None and ttft_ms is not None:
        decode_ms = max(total_ms - ttft_ms, 0.0)
    decode_only = None
    if n > 1 and decode_ms and decode_ms > 0:
        decode_only = (n - 1) / (decode_ms / 1000.0)
    with_ttft = None
    if n > 0:
        denom_ms = None
        if ttft_ms is not None and decode_ms is not None:
            denom_ms = ttft_ms + decode_ms
        elif total_ms is not None:
            denom_ms = total_ms
        if denom_ms and denom_ms > 0:
            with_ttft = n / (denom_ms / 1000.0)
    return decode_only, with_ttft


# ---------------------------------------------------------------------------
# Runners
# ---------------------------------------------------------------------------

@dataclass
class RunResult:
    answer: str
    seconds: float
    error: str | None = None
    # Time-to-first-token in milliseconds and decode throughput in tokens/sec.
    # None when the runtime didn't report them (e.g. on error/timeout).
    ttft_ms: float | None = None
    # tps == tps_decode (kept for backward compat with existing CSVs).
    tps: float | None = None
    # Decode-only throughput: (N - 1) / decode_time. Matches what the
    # geniex pipeline reports natively.
    tps_decode: float | None = None
    # Throughput with TTFT included: N / (ttft + decode_time). Matches what
    # `genie-t2t-run` surfaces in its profile JSON.
    tps_with_ttft: float | None = None
    generated_tokens: int | None = None
    decode_ms: float | None = None


def _csv_num(v: float | None) -> str:
    """Format a metric for the CSV cell ('' when unavailable)."""
    return f"{v:.2f}" if v is not None else ""


def _csv_int(v: int | None) -> str:
    return str(v) if v is not None else ""


def _fmt_ms(v: float | None) -> str:
    return f"{v:6.0f}ms" if v is not None else "    -  "


def _fmt_tps(v: float | None) -> str:
    return f"{v:5.1f}t/s" if v is not None else "    -   "


# auto_llm's --max-tokens is a hard cap with no "unlimited" sentinel. To match
# the genie side — which has no token cap and stops only on EOS / context
# exhaustion — we pass a value larger than any bundle's context window when the
# user asks for no limit (--max-tokens 0). The decode loop only uses this as an
# upper bound (no preallocation) and stops earlier on EOS or context_length, so
# an oversized value is safe.
UNLIMITED_MAX_TOKENS = 1_000_000


def run_geniex_batch(
    auto_llm: Path,
    model_dir: Path,
    prompts: list[Prompt],
    max_tokens: int,
    timeout: int,
    tokenizer_config: Path | None,
    enable_thinking: bool,
    verbose: bool,
) -> dict[int, RunResult]:
    """Run every prompt through `auto_llm --batch-file`, in ONE process.

    Batch mode loads the bundle once and resets the KV cache between prompts,
    so each answer is independent while the (very expensive) model load is paid
    a single time. auto_llm rewrites --metrics-json after every prompt, so if it
    crashes or is killed partway we still recover the completed answers.

    Returns {prompt id -> RunResult}. Prompts missing from auto_llm's output get
    an error RunResult so the caller can record the failure rather than silently
    dropping the row.
    """
    effective_max_tokens = max_tokens if max_tokens > 0 else UNLIMITED_MAX_TOKENS
    work_dir = Path(tempfile.mkdtemp(prefix="geniex_batch_"))
    batch_path = work_dir / "batch.json"
    metrics_path = work_dir / "metrics.json"
    batch_path.write_text(
        json.dumps([{"id": p.id, "prompt": p.text} for p in prompts], ensure_ascii=False),
        encoding="utf-8",
    )

    cmd = [
        str(auto_llm),
        "--model-dir", str(model_dir),
        "--batch-file", str(batch_path),
        "--metrics-json", str(metrics_path),
        "--system", SYSTEM_PROMPT,
        "--max-tokens", str(effective_max_tokens),
    ]
    if tokenizer_config is not None:
        cmd += ["--tokenizer-config", str(tokenizer_config)]
    if enable_thinking:
        cmd.append("--enable-thinking")
    if not verbose:
        cmd.append("--quiet")

    # Timeout is per-prompt in spirit but the process is shared, so scale it by
    # the prompt count (plus generous slack for the one-time model load).
    total_timeout = timeout * len(prompts) + 900
    start = time.time()
    proc_error: str | None = None
    try:
        cp = subprocess.run(
            cmd,
            capture_output=not verbose,
            text=True,
            timeout=total_timeout,
            encoding="utf-8",
            errors="replace",
        )
        if cp.returncode != 0:
            tail = ((cp.stderr or "") + (cp.stdout or ""))[-400:].strip() if not verbose else ""
            proc_error = f"auto_llm exit {cp.returncode}: {tail}"
    except subprocess.TimeoutExpired:
        proc_error = f"auto_llm timeout after {total_timeout}s"
    elapsed = time.time() - start

    # Read whatever auto_llm managed to write, even on failure — it rewrites the
    # metrics file after every prompt, so a partial run is still usable.
    rows: list[dict] = []
    if metrics_path.is_file():
        try:
            rows = json.loads(metrics_path.read_text(encoding="utf-8"))
        except ValueError as e:
            proc_error = proc_error or f"malformed metrics JSON: {e}"
    shutil.rmtree(work_dir, ignore_errors=True)

    # auto_llm reports per-prompt runtime metrics but not per-prompt wall time;
    # spread the batch wall time evenly for the progress display only (it is
    # not written to the CSV).
    per_prompt_seconds = elapsed / len(prompts) if prompts else 0.0

    out: dict[int, RunResult] = {}
    for row in rows:
        try:
            pid = int(row["id"])
        except (KeyError, TypeError, ValueError):
            continue
        ttft_ms = row.get("ttft_ms") or None
        decode_ms = row.get("decode_ms") or None
        tps_decode = row.get("tokens_per_second") or None
        gen_tok = row.get("generated_tokens") or None
        stop_reason = row.get("stop_reason") or ""
        _, tps_with_ttft = derive_tps(gen_tok, ttft_ms, None, decode_ms)
        out[pid] = RunResult(
            answer=(row.get("answer") or "").strip(),
            seconds=per_prompt_seconds,
            error="generation error" if stop_reason == "error" else None,
            ttft_ms=ttft_ms,
            tps=tps_decode,
            tps_decode=tps_decode,
            tps_with_ttft=tps_with_ttft,
            generated_tokens=gen_tok,
            decode_ms=decode_ms,
        )

    missing = [p.id for p in prompts if p.id not in out]
    if missing:
        reason = proc_error or "no result emitted by auto_llm"
        print(
            f"WARNING: auto_llm returned no result for {len(missing)} prompt(s) "
            f"(ids {missing[:10]}{'...' if len(missing) > 10 else ''}): {reason}",
            file=sys.stderr,
            flush=True,
        )
        for pid in missing:
            out[pid] = RunResult(answer="", seconds=0.0, error=reason)
    return out


# Cached per-config-dir tokenizer for backfilling Genie's generated-token
# count when the profile JSON omits it. Loaded lazily so the benchmark
# still runs (with `genie_tps_decode` empty, just like before) on hosts
# without the `tokenizers` package installed.
_TOKENIZER_CACHE: dict[Path, object | None] = {}


def _load_tokenizer(config_dir: Path) -> object | None:
    if config_dir in _TOKENIZER_CACHE:
        return _TOKENIZER_CACHE[config_dir]
    tok_path = config_dir / "tokenizer.json"
    tok: object | None = None
    if tok_path.is_file():
        try:
            from tokenizers import Tokenizer  # type: ignore[import-not-found]
            tok = Tokenizer.from_file(str(tok_path))
        except Exception:
            tok = None
    _TOKENIZER_CACHE[config_dir] = tok
    return tok


def _count_tokens(tokenizer: object | None, text: str) -> int | None:
    if tokenizer is None or not text:
        return None
    try:
        return len(tokenizer.encode(text).ids)  # type: ignore[attr-defined]
    except Exception:
        return None


def run_genie(config_dir: Path, formatted_prompt: str, timeout: int) -> RunResult:
    start = time.time()
    # genie-t2t-run writes its profiling JSON to the path given via --profile,
    # but it *refuses to overwrite* an existing file. So we make a fresh temp
    # DIRECTORY per call and point --profile at a not-yet-created file inside
    # it (absolute path, so it's unaffected by the cwd=config_dir we run from).
    profile_dir = Path(tempfile.mkdtemp(prefix="genie_profile_"))
    profile_path = profile_dir / "profile.log"
    # genie-t2t-run resolves ctx-bins / tokenizer relative to cwd, so we
    # must invoke it from the model directory.
    try:
        cp = subprocess.run(
            [
                "genie-t2t-run",
                "-c",
                "genie_config.json",
                "-p",
                formatted_prompt,
                "--profile",
                str(profile_path),
            ],
            capture_output=True,
            text=True,
            cwd=str(config_dir),
            timeout=timeout,
            encoding="utf-8",
            errors="replace",
        )
    except subprocess.TimeoutExpired:
        shutil.rmtree(profile_dir, ignore_errors=True)
        return RunResult(answer="", seconds=time.time() - start, error="timeout")
    out = (cp.stdout or "") + (cp.stderr or "")
    cleaned = clean_genie(out)
    err = None
    if cp.returncode != 0 and not cleaned:
        err = f"exit {cp.returncode}: {out[-200:].strip()}"
    gp = parse_genie_profile(profile_path)
    shutil.rmtree(profile_dir, ignore_errors=True)

    # If genie-t2t-run's profile didn't carry `num-tokens-generated`
    # (depends on QAIRT version), fall back to tokenising the recorded
    # answer with the bundle's own tokenizer.json.
    gen_tok = gp.generated_tokens
    if gen_tok is None and cleaned:
        gen_tok = _count_tokens(_load_tokenizer(config_dir), cleaned)

    # Genie's `token-generation-rate` is what we'll treat as `tps_with_ttft`
    # (it's the per-event total-time / token count). We then derive a
    # decode-only number when num-tokens-generated + (total_ms or
    # tps_reported) give us enough to back out decode time.
    decode_ms = None
    if gp.total_ms is not None and gp.ttft_ms is not None:
        decode_ms = max(gp.total_ms - gp.ttft_ms, 0.0)
    elif gp.tps_reported and gen_tok and gp.ttft_ms is not None:
        # Genie's rate ≈ tokens / total_ms. Solve for decode_ms so we can
        # report decode-only TPS.
        total_ms = (gen_tok / gp.tps_reported) * 1000.0
        decode_ms = max(total_ms - gp.ttft_ms, 0.0)
    tps_decode, tps_with_ttft = derive_tps(gen_tok, gp.ttft_ms, gp.total_ms, decode_ms)
    if tps_with_ttft is None:
        tps_with_ttft = gp.tps_reported  # fall back to whatever Genie said
    return RunResult(
        answer=cleaned,
        seconds=time.time() - start,
        error=err,
        ttft_ms=gp.ttft_ms,
        tps=tps_decode if tps_decode is not None else gp.tps_reported,
        tps_decode=tps_decode,
        tps_with_ttft=tps_with_ttft,
        generated_tokens=gen_tok,
        decode_ms=decode_ms,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_PROMPTS = SCRIPT_DIR / "testing_prompts.md"
DEFAULT_RESULTS_DIR = SCRIPT_DIR / "results"

# Where `cmake --build build --config Release --target auto_llm` puts the binary
# on each platform. Probed in order; override with --auto-llm.
AUTO_LLM_CANDIDATES = (
    REPO_ROOT / "build" / "bin" / "Release" / "auto_llm.exe",
    REPO_ROOT / "build" / "bin" / "auto_llm.exe",
    REPO_ROOT / "build" / "bin" / "auto_llm",
)


def slugify_dir(model_dir: Path) -> str:
    """../modelfiles/Qwen3-4B-Instruct-2507 -> qwen3_4b_instruct_2507."""
    return model_dir.resolve().name.replace("-", "_").replace(".", "_").lower()


def find_auto_llm() -> Path | None:
    """Locate a built auto_llm binary, preferring the standard build output."""
    for c in AUTO_LLM_CANDIDATES:
        if c.is_file():
            return c
    found = shutil.which("auto_llm")
    return Path(found) if found else None


def main() -> int:
    p = argparse.ArgumentParser(
        description="Compare the GenieX QAIRT plugin vs genie-t2t-run answer "
        "quality on a fixed prompt suite, over the same model bundle.",
    )
    p.add_argument(
        "--model-dir",
        type=Path,
        required=True,
        help="QAIRT export bundle directory (contains genie_config.json, "
        "tokenizer.json, tokenizer_config.json and the *.bin context shards). "
        "Both runtimes read this same directory.",
    )
    p.add_argument(
        "--prompts",
        type=Path,
        default=DEFAULT_PROMPTS,
        help=f"Prompt-suite markdown (default: {DEFAULT_PROMPTS.name} alongside this script)",
    )
    p.add_argument(
        "--auto-llm",
        type=Path,
        default=None,
        help="Path to the built auto_llm executable (default: auto-discovered "
        "under <repo>/build/bin/, then $PATH).",
    )
    p.add_argument(
        "--tokenizer-config",
        type=Path,
        default=None,
        help="Override the tokenizer_config.json auto_llm reads the Jinja chat "
        "template from (default: <model-dir>/tokenizer_config.json).",
    )
    p.add_argument(
        "--enable-thinking",
        action="store_true",
        help="Pass --enable-thinking to auto_llm, plumbing "
        "{\"enable_thinking\":true} into the Jinja context. Only affects "
        "templates that read the field (Qwen3 reasoning models).",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="CSV output path (default: results/<slug>.csv next to this script, "
        "where <slug> is the bundle directory name)",
    )
    p.add_argument(
        "--slug",
        default=None,
        help="Override the <slug> used to name output files (default: the "
        "bundle directory name). Useful when benchmarking the same bundle "
        "across plugin revisions.",
    )
    p.add_argument(
        "--max-tokens",
        type=int,
        default=0,
        help="Cap on tokens generated per prompt on the geniex side. Default 0 "
        "= no cap: auto_llm runs to its natural EOS / context-exhaustion stop, "
        "matching the genie side (which has no token cap). Pass a positive "
        "value to truncate (e.g. for quick smoke tests). The genie side has no "
        "equivalent flag and always stops on EOS / context exhaustion.",
    )
    p.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="Per-prompt timeout in seconds for the genie-t2t-run subprocess. "
        "The geniex side runs as ONE batch process, so its timeout is "
        "(--timeout x prompt count) + 900s of slack for the model load.",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=0,
        help="If >0, only run the first N prompts (useful for smoke tests)",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="Skip prompts whose id already appears in --out",
    )
    p.add_argument(
        "--skip-genie",
        action="store_true",
        help="Skip the genie-t2t-run pass entirely (only run the plugin via "
        "auto_llm). Use this when only plugin code changed — genie's answers "
        "for the same bundle are deterministic and were captured in a previous "
        "full run. The genie_* columns in the output CSV are left blank, and "
        "the default output path becomes <slug>_geniex_only.csv so it doesn't "
        "clobber the prior full run. Pair the new file with the prior full run "
        "when scoring (see SCORE_WITH_CLAUDE.md).",
    )
    p.add_argument(
        "--stream-geniex",
        action="store_true",
        help="Let auto_llm stream its answers and banners to this console "
        "(drops --quiet). Handy for watching a live run; output is noisier.",
    )
    args = p.parse_args()

    model_dir = args.model_dir.resolve()
    if not model_dir.is_dir():
        print(f"ERROR: --model-dir is not a directory: {model_dir}", file=sys.stderr)
        return 2
    # genie_config.json is required by both sides: genie-t2t-run reads it
    # directly, and it supplies the bos-token that picks the genie-side
    # hand-built template. Fail fast here rather than deep in a subprocess.
    cfg_path = model_dir / "genie_config.json"
    if not cfg_path.is_file():
        print(f"ERROR: missing {cfg_path} — is this a QAIRT export bundle?", file=sys.stderr)
        return 2

    auto_llm = args.auto_llm.resolve() if args.auto_llm else find_auto_llm()
    if auto_llm is None or not auto_llm.is_file():
        print(
            "ERROR: could not find a built auto_llm executable. Build it with\n"
            "  cmake --build build --config Release --target auto_llm\n"
            "or pass --auto-llm <path>.",
            file=sys.stderr,
        )
        return 2
    print(f"Using auto_llm: {auto_llm}", flush=True)
    print(f"Bundle: {model_dir}", flush=True)

    slug = args.slug or slugify_dir(model_dir)
    if args.out is None:
        DEFAULT_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        suffix = "_geniex_only.csv" if args.skip_genie else ".csv"
        args.out = DEFAULT_RESULTS_DIR / f"{slug}{suffix}"
        print(f"Auto-derived output path: {args.out}", flush=True)
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)

    if args.skip_genie:
        print("--skip-genie: genie-t2t-run pass disabled; only auto_llm will run.", flush=True)

    prompts = load_prompts(args.prompts)
    if args.limit:
        prompts = prompts[: args.limit]
    print(f"Loaded {len(prompts)} prompts from {args.prompts}", flush=True)

    genie_cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    template_key = detect_template(genie_cfg)
    template = TEMPLATES[template_key]
    print(f"genie-side chat template: {template_key}", flush=True)

    done_ids: set[int] = set()
    rows: list[dict] = []
    if args.resume and args.out.exists():
        with args.out.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                rows.append(row)
                try:
                    done_ids.add(int(row["id"]))
                except (KeyError, ValueError):
                    pass
        print(f"Resume: {len(done_ids)} prompts already in {args.out}", flush=True)

    fieldnames = [
        "id",
        "category",
        "prompt",
        "genie_answer",
        "geniex_answer",
        "genie_score",
        "geniex_score",
        "note",
        "genie_ttft_ms",
        "geniex_ttft_ms",
        # `*_tps` is the legacy column (decode-only on the geniex side, the
        # genie-reported rate on the genie side). Kept so existing scoring
        # tooling and historical CSVs still load. The two columns below are
        # the apples-to-apples pair: decode-only on both sides, and total
        # (TTFT + decode) on both sides.
        "genie_tps",
        "geniex_tps",
        "genie_tps_decode",
        "geniex_tps_decode",
        "genie_tps_with_ttft",
        "geniex_tps_with_ttft",
        "genie_generated_tokens",
        "geniex_generated_tokens",
        "genie_decode_ms",
        "geniex_decode_ms",
        "genie_error",
        "geniex_error",
    ]

    def write_all() -> None:
        with args.out.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k, "") for k in fieldnames})

    pending = [pr for pr in prompts if pr.id not in done_ids]
    if not pending:
        print("Nothing to do — every prompt is already in the output CSV.", flush=True)

    # The geniex side runs as a SINGLE auto_llm process over all pending
    # prompts: the bundle is loaded once and the KV cache reset between
    # prompts. auto_llm rewrites its metrics file after every prompt, so a
    # crash partway still yields the completed answers.
    geniex_results: dict[int, RunResult] = {}
    if pending:
        print(
            f"Running {len(pending)} prompt(s) through auto_llm (batch mode, "
            "model loaded once) ...",
            flush=True,
        )
        geniex_results = run_geniex_batch(
            auto_llm=auto_llm,
            model_dir=model_dir,
            prompts=pending,
            max_tokens=args.max_tokens,
            timeout=args.timeout,
            tokenizer_config=args.tokenizer_config,
            enable_thinking=args.enable_thinking,
            verbose=args.stream_geniex,
        )
        ok = sum(1 for r in geniex_results.values() if not r.error)
        print(f"  auto_llm done: {ok}/{len(pending)} prompts answered", flush=True)

    for prompt in prompts:
        if prompt.id in done_ids:
            continue
        gx = geniex_results.get(prompt.id) or RunResult(
            answer="", seconds=0.0, error="missing from auto_llm output"
        )
        print(f"\n[{prompt.id:03d}] ({prompt.category}) {prompt.text[:80]}", flush=True)
        print(
            f"  geniex: ttft={_fmt_ms(gx.ttft_ms)}  "
            f"tps={_fmt_tps(gx.tps_decode)} (decode) "
            f"{_fmt_tps(gx.tps_with_ttft)} (w/ttft)  "
            f"err={gx.error or '-'}",
            flush=True,
        )
        if args.skip_genie:
            gn = RunResult(answer="", seconds=0.0)
            print("  genie : skipped (--skip-genie)", flush=True)
        else:
            # genie-t2t-run needs the hand-built formatted string; the geniex
            # side applies the bundle's own Jinja template inside auto_llm.
            genie_formatted = template.format(prompt=prompt.text)
            gn = run_genie(model_dir, genie_formatted, args.timeout)
            print(
                f"  genie : {gn.seconds:5.1f}s  ttft={_fmt_ms(gn.ttft_ms)}  "
                f"tps={_fmt_tps(gn.tps_decode)} (decode) "
                f"{_fmt_tps(gn.tps_with_ttft)} (w/ttft)  "
                f"err={gn.error or '-'}",
                flush=True,
            )

        rows.append(
            {
                "id": prompt.id,
                "category": prompt.category,
                "prompt": prompt.text,
                "genie_answer": gn.answer,
                "geniex_answer": gx.answer,
                "genie_score": "",
                "geniex_score": "",
                "note": "",
                "genie_ttft_ms": _csv_num(gn.ttft_ms),
                "geniex_ttft_ms": _csv_num(gx.ttft_ms),
                "genie_tps": _csv_num(gn.tps),
                "geniex_tps": _csv_num(gx.tps),
                "genie_tps_decode": _csv_num(gn.tps_decode),
                "geniex_tps_decode": _csv_num(gx.tps_decode),
                "genie_tps_with_ttft": _csv_num(gn.tps_with_ttft),
                "geniex_tps_with_ttft": _csv_num(gx.tps_with_ttft),
                "genie_generated_tokens": _csv_int(gn.generated_tokens),
                "geniex_generated_tokens": _csv_int(gx.generated_tokens),
                "genie_decode_ms": _csv_num(gn.decode_ms),
                "geniex_decode_ms": _csv_num(gx.decode_ms),
                "genie_error": gn.error or "",
                "geniex_error": gx.error or "",
            }
        )
        # Persist after every prompt — long runs are expensive to lose.
        rows.sort(key=lambda r: int(r["id"]))
        write_all()

    # Companion JSON: minimal payload the scoring agent reads. Same basename
    # as the CSV, with `.answers.json` appended so `<slug>.csv` /
    # `<slug>.answers.json` stay paired.
    answers_path = args.out.parent / (args.out.stem + ".answers.json")
    rows_payload = [
        {
            "id": int(r["id"]),
            "category": r.get("category", ""),
            "prompt": r.get("prompt", ""),
            "genie_answer": r.get("genie_answer", ""),
            "geniex_answer": r.get("geniex_answer", ""),
        }
        for r in rows
    ]
    # Wrap with a meta block so downstream tooling (SCORE_WITH_CLAUDE.md) can
    # tell a geniex-only run apart from a full run without diffing every row.
    # Keep backward compat: a top-level list (the legacy shape) is still
    # readable by the scoring agent — the meta block is purely additive.
    answers_payload: list[dict] | dict
    if args.skip_genie:
        answers_payload = {
            "meta": {
                "model_dir": str(model_dir),
                "slug": slug,
                "runtimes": ["geniex"],
                "skip_genie": True,
            },
            "rows": rows_payload,
        }
    else:
        answers_payload = rows_payload
    answers_path.write_text(
        json.dumps(answers_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"\nWrote {args.out} ({len(rows)} rows)", flush=True)
    print(f"Wrote {answers_path} (for scoring agent)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
