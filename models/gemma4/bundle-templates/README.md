# Gemma4 bundle config templates

`modelfiles/` is gitignored, so the `genie_config.json` that makes a bundle work is not
tracked anywhere. These are reference copies.

| File | Status |
|------|--------|
| `genie_config.e2b.json` | **Validated** — the exact config used for the X Elite / v73 run that matches Genie token-for-token. |
| `genie_config.e4b.json` | **Untested template** — dimensions from `gemma-4-E4B-it-qat-mobile-ct-FP16/config.json`; no E4B binaries exist yet. |

To build a bundle: copy the matching file to `modelfiles/<variant>/genie_config.json`, then add
`tokenizer.json`, a `tokenizer_config.json` **with an inline `chat_template`**, and the `.bin`
shards + LUTs *directly* in that directory (not a subdirectory — `bundleDirOf()` resolves the
bundle as the parent of `ctx-bins[0]`).

**The `quant-param` blocks are the part you must not copy blindly.** They have to match the
export's own `embed_encodings.json` / `embed_tokens_encodings.json` exactly. A mismatch does not
error — it silently produces garbage tokens from the first token onward.

See `../README.md` for the full E2B↔E4B dimension table and the runtime requirements.
