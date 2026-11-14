# Streaming Phonon TTS on QAIRT

This optional pipeline generates mono float PCM from a Gradium Phonon QNN
bundle. `PhononModel` uses the existing `Model`, `Graph`, buffer allocator,
QNN HTP backend, runtime selection and logging. There is no second backend loader
or GenieX plugin. The model-independent `TTSPipeline` owns request serialization
and cancellation; Phonon's acoustic scheduling stays under `models/phonon/`.

## Inputs

- A QNN bundle exported by [Gradium xn-ptts](https://github.com/gradium-ai/xn-ptts),
  containing `metadata.json`, a compiled context with `prefill` and `step` graphs,
  tokenizer, BOS data and voice KV files. Both Phonon 31M and 40M were tested.
- An explicit SoC identifier present in `runtime.context_binaries.*.soc_models`.
  A GGUF file alone is insufficient. Use a context compiled for your device and
  a compatible QAIRT version; this code does not convert or download weights.
- The trusted native text frontend built from Gradium's `qnn/text-ffi`, currently
  tested at commit `4e1bc12ccb0806c6ca756d1945fabaa5cbf98fb7` (PR 273).
  Build with `cargo build --release --locked --manifest-path qnn/text-ffi/Cargo.toml`.
  On Windows the resulting library is `ptts_text.dll`; pass its absolute path.
  The executable-library path in bundle metadata is deliberately ignored.
- QNN HTP drivers and runtime libraries. Respect the model's access terms.
  Models, credentials and SDK binaries are not included in this contribution.

## Build and run

Enable `GENIEX_BUILD_TTS=ON` and `GENIEX_BUILD_EXAMPLES=ON` with the normal project
toolchain. On Windows ARM64 use ClangCL or GenieX's
`sdk/cmake/arm64-windows-llvm.cmake`, not plain MSVC:

```powershell
cmake -S . -B build -A ARM64 -T ClangCL `
  -DGENIEX_BUILD_TTS=ON -DGENIEX_BUILD_EXAMPLES=ON -DGENIEX_BUILD_TESTS=ON
cmake --build build --config Release --target phonon_tts phonon_device_test
& ./build/bin/Release/phonon_tts.exe `
  --bundle C:/models/phonon-40m-qnn --soc-model SC8380XP --lang en `
  --text-library C:/trusted/ptts_text.dll `
  --qairt C:/Qualcomm/AIStack/QAIRT/2.45.0.260326 `
  --text 'Hello from the shared QAIRT speech pipeline.' --out phonon.wav
```

For Linux/Android use the repository's normal cross toolchain and the matching
text frontend `.so`. Those platforms have **not** been run on hardware for this
contribution. Windows ARM64/X Elite is the tested target. Device-specific context
binaries and runtime/driver compatibility remain necessary on every platform.

The CLI uses strict NPU verification by default. Each successful request must
report positive accelerator event timings for **both** graphs. Missing contexts,
runtime errors and unsupported tensor layouts fail; no CPU or GPU backend is
retried. Text normalization, tokenization, acoustic scheduling, state copies and
WAV writing are CPU operations, while the neural graphs execute on HTP.

`--repeats 4` retains the model and writes four WAVs. JSON output reports
per-request graph counts/timings, first PCM, total generation time and RTF.
These timings exclude model initialization and audio-device playback. Callback
execution is synchronous and is included in generation time.

## C++ API

```cpp
#include "phonon/phonon_model.h"
auto model = std::make_unique<geniex::PhononModel>(
    geniex::PhononOptions{bundle, soc, "en", trusted_text_library, true});
geniex::QnnRuntimeConfig runtime;
runtime.htp_dir = qairt_sdk;
if (!model->load(runtime)) throw std::runtime_error("Cannot initialize HTP");
geniex::TTSPipeline pipe(std::move(model));
geniex::SpeechConfig config;
config.collect_audio = false;
auto result = pipe.synthesize("Hello", config,
    [](const float* pcm, size_t samples, int sample_rate) {
        // Consume or copy this borrowed mono float PCM block before returning.
        return true; // false cancels this request
    });
```

Link `geniex_tts`. One pipeline handles one synthesis at a time. `cancel()` may
be called from the callback or another thread. Cancellation preserves reuse;
a graph or C++ callback exception invalidates the model. Do not destroy a
pipeline while synthesis is active. The native text library stays loaded for
Rust TLS safety. The Gradium MIT notice is included alongside this code.

## Tests

CPU suites: `TTSPipeline.*`, `ExecutionProfile.*`, `PhononBundle.*`.
`tts_pipeline_test` is included in the standard patch-coverage surface.

To enable the real-device lifecycle test, set these before starting CTest:

```powershell
$env:PHONON_TEST_BUNDLE = 'C:/models/phonon-40m-qnn'
$env:PHONON_TEST_SOC = 'SC8380XP'
$env:PHONON_TEST_TEXT_LIBRARY = 'C:/trusted/ptts_text.dll'
$env:PHONON_TEST_QAIRT = 'C:/Qualcomm/AIStack/QAIRT/2.45.0.260326'
ctest --test-dir build -C Release -R '^PhononDevice\.' --output-on-failure
```

It checks real streamed PCM, both cancellation methods, subsequent successful
requests, per-request positive accelerator timings, invalid voice recovery and
failure invalidation. With no model environment it is skipped, not simulated.
