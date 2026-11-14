# Windows ARM64 validation

Validated on a Snapdragon X Elite X1E80100 laptop with the GenieX
`sdk/cmake/arm64-windows-llvm.cmake` toolchain, Clang 22.1.8, Release, and QAIRT
2.45.0.260326. The base of this contribution is `ab24abb` on the QAIRT plugin's
main branch. Contexts were compiled for SC8380XP; both Phonon 31M and 40M used
their matching complete bundles and the Gradium text frontend at
`4e1bc12ccb0806c6ca756d1945fabaa5cbf98fb7`.

## Checks performed

- Full standalone build and **399/399 CPU unit tests** passed.
- Five streaming-pipeline tests, two bundle tests and two profile-event tests
  cover input validation, request serialization, cancellation, exceptions,
  path confinement, device selection and accelerator-event accounting.
- Instrumented pipeline tests: **100% line, function, region and branch coverage**
  of `core/include/pipeline/tts_pipeline.h` (LLVM source coverage).
- Real-device lifecycle test passed separately for each model: two cancellation
  methods, three subsequent complete requests, invalid voice recovery, finite
  streamed PCM, per-request accelerator counters and callback failure invalidation.
- The companion GenieX SDK built with this pipeline under its existing `qairt`
  plugin. Its two C API/path contract suites and **29 Python device checks**
  passed. Checks included explicit CPU/GPU rejection, wrong SoC, missing native
  frontend, conflicting runtime selection, multi-chunk text, callback exceptions,
  cancellation/reuse and per-request profiling deltas.
- Inspected the Python process's loaded modules: HTP and System came from the
  selected QAIRT 2.45 runtime; no QNN CPU or GPU provider was loaded.
- Eight WAVs generated through the SDK passed PCM-format and nonsilence checks.
  For each model the first WAV matched the Gradium PR-273 reference runner
  with the same input, seed and temperature: identical sample count, cosine
  similarity above 0.99999996 and maximum absolute PCM16 difference of **1**.
  This checks acoustic-loop parity, not a subjective listening-quality score.

## Measured observations

Input: `Hello from Phonon, running locally through GenieX on the Snapdragon NPU.`
Default voice, seed 0, temperature 0.3. Each model stayed loaded for four SDK
requests. Median values below use the last three requests; first request is
excluded from the warm median.

| Model | Audio length | Warm first PCM callback | Warm RTF | Model initialization |
| --- | ---: | ---: | ---: | ---: |
| Phonon 31M | 5.04 s | 6.99 ms | 0.0540 | 345 ms |
| Phonon 40M | 4.96 s | 9.02 ms | 0.0617 | 408 ms |

Every successful benchmark request reported positive QNN accelerator event time
for **both** `prefill` and `step`. First PCM is submission to the Python callback;
RTF is generation milliseconds divided by audio milliseconds. Both exclude model
initialization and speaker playback. These are observations on one machine and
one input, not performance guarantees or a comparison across devices.

There is no CPU/GPU backend retry. Text normalization, tokenization, state
management, random-noise preparation, callbacks and audio writing remain host
operations. Android/Linux hardware, VENTUNO Q, alternate SoCs and concurrent
LLM/TTS execution have **not** been validated by this revision.
