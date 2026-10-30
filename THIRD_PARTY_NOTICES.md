# Third-Party Notices

## Phonon acoustic generation and text frontend

The acoustic scheduling in `models/phonon/phonon_model.cpp` is adapted from
Gradium's `xn-ptts/qnn/runner/phonon.cpp` at revision
`4e1bc12ccb0806c6ca756d1945fabaa5cbf98fb7`:
https://github.com/gradium-ai/xn-ptts/tree/4e1bc12ccb0806c6ca756d1945fabaa5cbf98fb7.
It uses GenieX Model/Graph rather than linking Gradium's QNN loader or engine.
The original MIT license is retained in `models/phonon/LICENSE-xn-ptts-MIT`.

The explicitly supplied native `ptts_text` library is built from that project's
`qnn/text-ffi` and provides text normalization, chunking, and tokenization
through its C ABI. No weights, compiled contexts, or native text libraries are
redistributed here. Model/voice assets retain upstream licenses and terms.

GenieX-QAIRT-plugin is licensed under the BSD 3-Clause License (see
[`LICENSE.txt`](LICENSE.txt)). It ships with or depends on the following
third-party components, each governed by its own license.

## 1. Qualcomm AI Runtime (QAIRT) SDK

**Location in tree:** `qnn-api/` (both `src/` and `include/`, including
`include/HTP/` and `include/System/`).

**Origin:** Files extracted verbatim from the Qualcomm AI Runtime SDK
(QAIRT), downloadable from
https://www.qualcomm.com/developer/software/qualcomm-ai-engine-direct-sdk.
SDK version at time of extraction: v2.50.40.260831140417, per
`qnn-api/include/QnnSdkBuildId.h`; these headers declare QNN C API 2.39.0.
See [`qnn-api/README.md`](qnn-api/README.md) for the independent runtime API
floor and compatibility rationale.

**License:** Qualcomm AI Runtime SDK End User License Agreement (EULA)
as distributed with the SDK download. These files are proprietary
Qualcomm Technologies, Inc. code and are **not** relicensed under
BSD-3-Clause. The original per-file copyright and "Confidential and
Proprietary" markings are preserved.

See [`qnn-api/README.md`](qnn-api/README.md) for refresh instructions.

## 2. QAIRT Prebuilt Runtime Libraries

**Location in tree:**
- `third-party/windows/` — Windows ARM64 runtime libraries.
- `third-party/android/` — Android ARM64 runtime libraries.
- `third-party/linux-gcc11.2/` — Linux aarch64 (gcc 11.2 ABI) runtime libraries.

**Origin:** Prebuilt `.dll` / `.so` binaries shipped as part of a QAIRT SDK
release. Version: v2.50.40.260831 — matching the QAIRT SDK release used to
refresh the headers in item 1.

**License:** Qualcomm AI Runtime SDK EULA (same as item 1).

## 3. geniex-proc

**Location in tree:** `third-party/geniex-proc/` (git submodule).

**Origin:** Sibling project providing tokenizer and preprocessing
utilities, vendored as a submodule.

**License:** BSD 3-Clause License. See
`third-party/geniex-proc/LICENSE.txt` for the full text.

This submodule transitively vendors additional libraries under
`third-party/geniex-proc/third-party/` (xtensor, xtensor-blas, xtensor-io,
xtl, tokenizers-cpp, msgpack, etc.); each carries its own license file in
its respective directory.
