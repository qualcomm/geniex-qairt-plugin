# kv_layout_check

Reports the physical KV cache layout of a bundle: whether an `ENABLE_NATIVE_KV`
recipe actually emitted `HMX_WEIGHT_LAYOUT`, and whether the shapes it chose are
representable in that layout.

Family-agnostic on purpose — it loads the graphs and reads tensor metadata only,
with no spec inference, provider wiring or execution. So it works on any bundle,
including eaglet/speculative ones whose generic spec loading needs a
family-specific driver.

```pwsh
.\build\bin\Release\kv_layout_check.exe --model-dir C:\path\to\bundle
```

Real output, Qwen3-4B eaglet bundle (`windows_bundle_x_elite_v245`):

```
bundle: C:/Users/.../windows_bundle_x_elite_v245
  ctx-bin: "weight_sharing_model_ar128_ar32_cl2048_2_of_3_socid60_archv73.serialized.bin"
  ctx-bin: "weight_sharing_model_ar128_ar32_cl2048_3_of_3_socid60_archv73.serialized.bin"

graph 'ar128_cl2048_2_of_3'
  x36   in  key    FLAT_BUFFER  ufixed8  [8,1,128,1920]
  x36   in  value  FLAT_BUFFER  ufixed8  [8,1,1920,128]
  x36   out key    FLAT_BUFFER  ufixed8  [8,1,128,128]
  x36   out value  FLAT_BUFFER  ufixed8  [8,1,128,128]

graph 'ar32_cl2048_2_of_3'
  x36   in  key    FLAT_BUFFER  ufixed8  [8,1,128,2016]
  x36   in  value  FLAT_BUFFER  ufixed8  [8,1,2016,128]
  x36   out key    FLAT_BUFFER  ufixed8  [8,1,128,32]
  x36   out value  FLAT_BUFFER  ufixed8  [8,1,32,128]

== summary ==
  HMX_WEIGHT_LAYOUT tensors : 0
  FLAT_BUFFER tensors       : 288
  -> flat bundle; the runtime takes the original strided-copy path
```

A native bundle shows `HMX_WEIGHT_LAYOUT` on the `in` rows. If a declared shape
cannot be represented in that layout (e.g. a key `kv_len` that is not a multiple
of `K_TILE` = 256), the row is flagged `NOT REPRESENTABLE` with the reason and the
tool exits 2 — the runtime would refuse to load that bundle. Note the shapes
above: `1920` and `2016` are exactly that case, so this model's `kv_len` has to
change before it can go native.

At runtime the model also logs the detected layout and derived rebase at INFO.

## Overriding the rebase

The runtime derives the output→cache byte bias from tensor metadata. To force it
while diagnosing a mismatch:

```pwsh
$env:GENIEX_NATIVE_KV_REBASE = "0"   # no rebase
$env:GENIEX_NATIVE_KV_REBASE = "1"   # -128 (the derived default)
```
