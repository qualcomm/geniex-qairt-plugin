# Tests

CPU-only unit tests for `geniex-qairt`. They run in ordinary CI with **no NPU**
and no QNN runtime — the device boundary is replaced by a link-time `QnnApi`
stub. Built with GoogleTest (fetched via CMake) and run under CTest.

## Build & run

```pwsh
cmake -B build -A ARM64 -DGENIEX_BUILD_TESTS=ON
cmake --build build --config Release -j
ctest --test-dir build -C Release -L unit --output-on-failure
```

Run a single binary or case directly:

```pwsh
.\build\bin\Release\graph_test.exe --gtest_filter=GraphIO.*
```

## Layout

Mirrors `core/src/`; tests are grouped by the unit under test.

| Path | Covers |
|---|---|
| `core/utils_test.cpp`  | `utils.cpp` — fp16 ↔ fp32 conversion, `totalMs`, `mergeTimeLogs`. |
| `core/graph_test.cpp`  | `graph.cpp` — spec building and write→execute→read round-trips (fp32, fp16, quantized uint8/uint16, int32), plus overflow handling. |
| `testing/graph_info_builder.hpp` | Builds `GraphInfo_t` / `Qnn_Tensor_t` fixtures with CPU client buffers. |
| `testing/stub_qnnapi.cpp` | Link-time `QnnApi` stub; `graphExecute` copies input→output (identity). |

## How it works

- **Pure-logic suites** (e.g. `utils_test`) compile only the `core/` source under
  test — no device dependency, no doubles.
- **`graph_test`** drives the real `Graph` against a real `ClientBuffer`-backed
  `IOTensor` (`BufferAlloc::DEFAULT`, CPU heap buffers) and a link-time `QnnApi`
  stub. The real `QnnApi.cpp` is not linked; `graphExecute` is the only `QnnApi`
  method `Graph` calls.

## Adding a test

Add `core/<unit>_test.cpp` and register it in [`CMakeLists.txt`](CMakeLists.txt)
via `geniex_add_unit_test(<name> SOURCES ...)`, listing the `core/` translation
units under test. Tests are discovered per-case and labeled `unit;cpu`.

## Scope

These tests cover host-side logic above the QNN boundary. `Model::initialize`
(QNN backend bring-up) is out of scope — it requires the full QNN runtime and is
covered by on-device integration testing. Some `graph.cpp` quant/cast kernels are
currently `static` and tested indirectly through `Graph` round-trips; extracting
them for direct testing is tracked in qcom-ai-hub/geniex#1021.
