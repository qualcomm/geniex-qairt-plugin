# qnn-api/

This directory contains files **extracted verbatim from the Qualcomm AI
Runtime (QAIRT) SDK**. They are *not* part of the BSD-3-Clause licensed
portion of this project.

## Source

- **SDK:** Qualcomm AI Runtime SDK (QAIRT), also referred to as the
  Qualcomm AI Engine Direct SDK.
- **Download:** https://www.qualcomm.com/developer/software/qualcomm-ai-engine-direct-sdk
- **Version at extraction:** v2.36.1.250708151608_123266, per
  `include/QnnSdkBuildId.h` — the authoritative record, since it ships with the
  headers. These headers declare QNN C API 2.27.0 (`include/QnnCommon.h`).

  Deliberately *not* the newest SDK: the plugin negotiates the C API at load
  time and accepts any runtime at or above the version it compiled against, so
  the oldest headers we support give the widest runtime range. Do not bump this
  to match the bundled runtime libraries — those are a separate, newer release
  (`GENIEX_QAIRT_VERSION`, v2.45.0.260326) and are tracked in
  [`../THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md) §2.

## License

These files are governed by the **QAIRT SDK End User License Agreement**
shipped with the SDK download, not by this project's BSD-3-Clause license.
The original per-file Qualcomm copyright and "Confidential and Proprietary"
markings are preserved intentionally — they accurately describe the
licensing status of these files.

Do **not** rewrite or strip these headers. If you need to update the files,
replace them from a fresh SDK download (see *Refreshing* below).

See [`../THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md) for the full
third-party component list.

## Layout

The SDK's `QNN/` level is dropped on extraction — the public headers sit
directly in `include/`, because the build puts `include/`, `include/HTP` and
`include/System` on the include path and sources include them unqualified
(`#include "QnnCommon.h"`).

- `include/*.h` — QAIRT public C API headers (`QnnCommon.h`, `QnnTypes.h`, etc.).
- `include/HTP/` — HTP (Hexagon Tensor Processor) backend headers, plus
  `HTP/core/` internals.
- `include/System/` — QNN System API headers (context binary introspection).
- `include/*.hpp` — additional SDK-provided C++ helpers (`MmappedFile`,
  `MmappedReader`).
- `src/*.cpp` — SDK-provided wrapper implementations, since modified.

## Refreshing

To update these files to a newer SDK version:

Read the note on **Version at extraction** above first — refreshing to a newer
SDK *narrows* the range of runtimes the build accepts, so it is rarely what you
want. Bumping the bundled runtime libraries (step 5) does not require it.

1. Download the target QAIRT SDK from the link above.
2. From the extracted SDK (paths relative to its `include/QNN/`), copy:
   - `QNN/*.h` → `qnn-api/include/` (flatten; drop the `QNN/` level)
   - `QNN/HTP/**` → `qnn-api/include/HTP/`
   - `QNN/System/**` → `qnn-api/include/System/`
   - The matching sample/wrapper `.cpp` / `.hpp` files from
     `examples/Genie/Genie/src/qualla/engines/qnn-api/` and
     `.../qualla/MmappedFile/include/MmappedFile/` → `qnn-api/src/` and
     `qnn-api/include/`. These are modified in-tree, so merge rather than
     overwrite.
3. Update the version recorded in this file (from the new
   `include/QnnSdkBuildId.h`) and in `../THIRD_PARTY_NOTICES.md` §1, and update
   `GENIEX_QNN_API_VERSION` in the top-level `CMakeLists.txt` to the new
   `QNN_API_VERSION_MAJOR.MINOR` from `include/QnnCommon.h`.
4. Rebuild and run smoke tests against an existing model.
5. Update the bundled runtime binaries under `third-party/{windows,android,linux-gcc11.2}/` if they are from the same SDK release. These carry their own version — `GENIEX_QAIRT_VERSION` and `../THIRD_PARTY_NOTICES.md` §2 — which is independent of the headers above.
