# qnn-api/

This directory contains files **extracted verbatim from the Qualcomm AI
Runtime (QAIRT) SDK**. They are *not* part of the BSD-3-Clause licensed
portion of this project.

## Source

- **SDK:** Qualcomm AI Runtime SDK (QAIRT), also referred to as the
  Qualcomm AI Engine Direct SDK.
- **Download:** https://www.qualcomm.com/developer/software/qualcomm-ai-engine-direct-sdk
- **Version at extraction:** v2.50.40.260831140417, per
  `include/QnnSdkBuildId.h` — the authoritative record, since it ships with the
  headers. These headers declare QNN C API 2.39.0 (`include/QnnCommon.h`). The
  bundled runtime libraries are also from QAIRT v2.50.40.260831 (tracked in
  [`../THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md) §2).

  The runtime negotiation floor is deliberately independent of the header
  version: `kMinApiMinor` in `src/QnnApi.cpp` remains 27 (the QNN System API
  floor remains 4). The loader accepts providers at or above those floors and
  copies only the floor-era interface prefixes, leaving later API members null.
  QAIRT 2.45/2.48/2.49 compatibility was verified before this header refresh;
  revalidate those older runtimes before claiming compatibility with this build.

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

Read the note on **Version at extraction** above first. A header refresh changes
compile-time declarations, but the runtime API floor remains the explicit
`kMinApiMinor` check; test older runtimes after any refresh. Bumping only the
bundled runtime libraries (step 5) does not require a header refresh.

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
   `include/QnnSdkBuildId.h`) and in `../THIRD_PARTY_NOTICES.md` §1. Preserve
   `kMinApiMinor` in `src/QnnApi.cpp` unless API-use analysis requires raising
   the runtime floor; the compile-time `static_assert` only ensures the headers
   declare at least that API level.
4. Rebuild and run smoke tests against the new runtime and representative older
   runtimes; use an existing model for device-level validation when available.
5. Update matching bundled runtime binaries under
   `third-party/{windows,android,linux-gcc11.2}/` when the SDK release supplies
   them. Runtime metadata (`GENIEX_QAIRT_VERSION` and
   `../THIRD_PARTY_NOTICES.md` §2) is independent of the header build ID.
