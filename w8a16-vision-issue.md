# Gemma4-E2B VEG: W8A16 vision encoder fails on v73, FP16 works

**TL;DR** — Use the **FP16** vision encoder (`vision_fp16/veg_xelite_v73.serialized.bin`). The **W8A16**
variant (`vision_w8a16/veg_xelite_v73_w8a16.serialized.bin`) loads but **fails at NPU execution** with a
skel-side DMA error (`err 1003`). The fix is upstream in the AI Hub Workbench export, not in this plugin.

Assets: `C:\Users\zackli\zackli\qualcomm-code\gemma4-e2b-v73-aihub-recipe-exported-v2\` (soc_id 60,
Hexagon **v73**, QAIRT **2.45**).

---

## What works vs. what doesn't

| Dir | `pixel_values` / `vision_embedding` dtype | Size | On-device result |
|-----|-------------------------------------------|------|------------------|
| `vision_fp16/veg_xelite_v73.serialized.bin` | `FLOAT_32` | 420 MB | ✅ runs (coherent output) |
| `vision_w8a16/veg_xelite_v73_w8a16.serialized.bin` | `UFIXED_POINT_16` (quantized I/O) | 252 MB | ❌ `err 1003` / hang |

FP16 verified twice: `image … -> 256 soft tokens`, `vision_embedding [256,1536]`, coherent
golden-retriever description, EXIT=0, TTFT ~345 ms, decode ~30 tok/s.

## The failure

The W8A16 graph **loads** with the same I/O geometry (`[1,2520,768] -> [1,256,1536]`), then graph
**execution** on the v73 HTP either hangs indefinitely or dies with:

```
[QNN] <E> Internal error handing: Dma execution failed on the skel side. result = 1003 transport error = 0
[QNN] <E> Graph graph_… failed in execution with err 1003
"Failed to execute graph. Error 1003"
```

`err 1003` is a **skel-side (Hexagon) DMA / transport failure** — the AI-Hub-quantized W8A16 ViT graph
does not execute on this v73 skel. It is a hardware/graph-level runtime failure, not an I/O-marshalling
bug the runtime can paper over.

## What the difference actually is (measured, not guessed)

Ran `qnn-context-binary-utility --context_binary <bin> --json_file <out>` (QAIRT 2.45,
`x86_64-windows-msvc`) on **both** binaries and diffed the graph I/O:

| Tensor | FP16 dtype | W8A16 dtype |
|--------|-----------|-------------|
| `pixel_values` (in) | `QNN_DATATYPE_FLOAT_32` | `QNN_DATATYPE_UFIXED_POINT_16` (SCALE_OFFSET) |
| `image_position_ids` (in) | `QNN_DATATYPE_INT_32` | `QNN_DATATYPE_INT_32` (same) |
| `vision_embedding` (out) | `QNN_DATATYPE_FLOAT_32` | `QNN_DATATYPE_UFIXED_POINT_16` (SCALE_OFFSET) |

So the W8A16 export quantized both the float **I/O tensors** (AI Hub `--quantize_io`) *and* the internal
weights/activations to INT8/INT16. FP16 keeps the whole graph in float.

## This is NOT a plugin bug

- The plugin's `Graph::write(float*)` / `Graph::read(float*)` paths already handle
  `QNN_DATATYPE_UFIXED_POINT_16` (they dequantize via `scale`/`offset` — see
  `core/src/graph.cpp` `writeFloatLike` / `Graph::read`). So marshalling float↔uint16 I/O is not the
  problem.
- The encoder loads fine (I/O introspection succeeds, shapes validate). The failure is strictly at
  `graph.execute()` on the NPU. That points at the compiled graph itself, not the host runtime.

## The wedge trap (important operational gotcha)

A **hung or failed W8A16 session leaves the HTP context in a stuck state**, so the *next* run — even the
known-good `vision_fp16` command — then also fails with `err 1003`. This is what made FP16 look broken:
it wasn't; a leftover wedged W8A16 process was poisoning the shared HTP context.

**Recovery:** kill any leftover process and re-run — FP16 recovers immediately.

```powershell
taskkill /IM gemma4_vlm.exe /F
```

Rule of thumb: if FP16 suddenly throws `err 1003` right after a W8A16 attempt, it's the wedge, not FP16.

## Correct run command (FP16)

```powershell
cd C:\Users\zackli\zackli\qualcomm-code\geniex-qairt-plugin\build-v73\bin\Release
.\gemma4_vlm.exe --model-dir C:\Users\zackli\zackli\qualcomm-code\gemma4-e2b-new-llm\modelfiles\gemma4_e2b `
                 --veg-dir   C:\Users\zackli\zackli\qualcomm-code\gemma4-e2b-v73-aihub-recipe-exported-v2\vision_fp16 `
                 --image     C:\Users\zackli\zackli\qualcomm-code\GenieX\tests\assets\quality_dog.jpg `
                 --prompt    "describe this image" --max-tokens 256 --verbose
```

## Learnings / know-how

1. **`err 1003` == skel-side DMA failure**, i.e. the graph doesn't run on the Hexagon skel. Distinguish
   it from load-time errors: `5000` = "failed to get context blob meta info" (SDK-version/serialization
   mismatch), `5005` = "can't read future blob" (wrong/newer skel). 1003 is a *runtime* execution
   failure — the binary is loadable but the graph is bad for this hardware/skel.
2. **`qnn-context-binary-utility --json_file` is the fast way to diff two context binaries** — dtypes,
   quant encodings, I/O shapes, buildId, backend/core API versions — without running them. Put the SDK
   `lib/x86_64-windows-msvc` on PATH so the DLLs resolve.
3. **Native stdout is fully buffered when piped/redirected** — a hung exe shows *nothing* through
   `| tail` until it exits. To watch a possibly-hung QNN run, redirect to a file and poll the file; the
   QNN `[INFO]/[ERROR]` logs go to stderr and are less buffered.
4. **HTP context is shared/stateful across processes** — one wedged NPU session can fail unrelated later
   runs. Always kill stragglers before concluding a *different* config is broken.
5. **FP16 VEG is only 420 MB and runs at ~30 tok/s end-to-end** — the W8A16 size win (252 MB) buys
   nothing here because it doesn't run. No reason to pursue W8A16 for this deployment.

## What to fix in the Workbench export process

The bug is upstream in the AI Hub `gemma_4_e2b_it` `vision_compile` W8A16 path. To fix / work around:

1. **Reproduce minimally on-device:** run the W8A16 context binary through `qnn-net-run` (not just the
   plugin) with dummy inputs to confirm `err 1003` is intrinsic to the graph, independent of this
   runtime. This isolates it as a compile/quantization defect.
2. **Drop `--quantize_io`** so `pixel_values` / `vision_embedding` stay `FLOAT_32` while the graph body
   is still W8A16. Quantized *I/O* tensors on a ViT with a 2520-patch input may be what trips the skel
   DMA path; float I/O + int internals is the safer combination to try first.
3. **Try a newer/older QAIRT for the compile+link** (2.46 / 2.48 are installed locally). 1003 can be a
   version-specific HTP codegen bug; the FP16 path linking cleanly on 2.45 while W8A16 fails suggests a
   quantized-graph codegen issue in that toolchain.
4. **Revisit the W8A16 calibration** — the encoder was PTQ'd with 100-image Imagenette calibration. If a
   layer's activation range collapses, the generated DMA descriptors can be malformed. Inspect the
   `model.encodings` for degenerate (zero/NaN) scales on the ViT attention/projector layers.
5. **Until fixed, ship FP16.** Update any export README / recipe docs to mark the W8A16 vision encoder as
   **not runnable on v73** and default consumers to `vision_fp16/`.

## References

- Run instructions + wedge-trap note: `run.md` (VLM section, end of file).
- Plugin I/O dtype handling: `core/src/graph.cpp` (`writeFloatLike`, `Graph::read`).
- VEG wrapper: `models/gemma4/gemma4_vision.h`.
- Export details / jobs: `gemma4-e2b-v73-aihub-recipe-exported-v2/README.md`.

---

## UPDATE (2026-07-23, second session) — common root cause with the LLM, and the LLM is now FIXED

### `--quantize_full_type w8a16` (the qairt-quantizer) is the common culprit
Both broken assets were compiled with `--quantize_full_type w8a16`:
- **W8A16 vision** → `err 1003` skel DMA fail at execute (this doc).
- **LLM `part2_of_3.bin`** → garbage output; its output hidden `add_29930` came out ~10× attenuated (amax≈18.7 vs healthy 176.8). Its AI Hub **link** failed 8× (exit 14). Reproduced locally: 2.45 compose → error 1002 ("Received nullptr for graphsInput"); 2.48.1 compose → `validateOpConfig 3110` / `Failed to validate op node_Conv_4567 error 0xc26`, even with `fp16_relaxed_precision + O2`.

Colleague's tip (confirmed): the HUB job used the **qairt-quantizer because of `--quantize_full_type w8a16`**, and that path emits ops/DMA layouts the v73 HTP rejects. **Dropping `--quantize_full_type` fixes the LLM** (see below). The same fix should be tried for the vision encoder (options in "What to fix", plus §7 below).

### LLM fix that WORKED (reference for fixing the vision encoder the same way)
1. Recompile each (part, graph) **directly to a context binary** (`--target_runtime qnn_context_binary`), **WITHOUT `--quantize_full_type`**, one graph per job:
   ```
   --target_runtime qnn_context_binary --output_names add_29930 --quantize_io
   --qairt_version 2.45 --qnn_options context_enable_graphs=<single graph>
   ```
   Single-graph compile without `--quantize_full_type` → **SUCCESS** (vs exit 14 with it).
2. The weight-share **LinkJob still fails (exit 14)** even on the no-qft DLCs → weight-sharing is separately broken for this graph. So emit each of the 6 (part{1,2,3} × {prompt,token}) as its **own single-graph context binary**.
3. **QNN loader constraint:** all context files in one model must have the **same graph count** — else `"Different len(graphs) found in different context files. Found 1 vs 2"`. So ALL 6 parts must be single-graph (can't mix shipped 2-graph part1/3 with 1-graph part2).
4. Result: working bundle `gemma4-v2-fixed/gemma4_e2b/` (6 single-graph ctxbins + config + notebook embedding LUTs). Verified: "capital of France"→"Paris.", "who are you"→"I am Gemma 4… by Google", and **full VLM E2E** with `vision_fp16` → coherent golden-retriever caption.

### Vision-encoder recompile attempt left running
Submitted an AI Hub job to recompile the W8A16 vision encoder directly to a context binary (fused, from ONNX `mqe4l934m`): job **`jgz4k1j6p`** (still `--quantize_full_type w8a16`, so it may still 1003 — if so, retry per §7 option 2: **drop `--quantize_io`** to keep FLOAT_32 I/O with w8a16 weights). Left running server-side when this session stopped; check status with `qai_hub.get_job('jgz4k1j6p')`, download the target model, drop it in a `vision_w8a16/` dir, and test with `gemma4_vlm.exe --veg-dir …`.

### AI Hub access
`qai-hub` is authenticated globally (`~/.qai_hub/client.ini`). Recover any compile artifact:
`hub.get_job(<id>).get_target_model().download("out.bin")`. Key model/job ids: vision ONNX `mqe4l934m`; LLM part2 AIMET `mnl02g3kn`; working LLM single-graph ctxbin jobs — part1 `jgz4k1dzp`/`j5w1nj6zg`, part2 `jp84vn1k5`/`jg9xer1mg`, part3 `jg9xe6nqg`/`jp1vxrzkp`.
