# ABI stability

`libmlxforge`'s C ABI (`src/capi/mlxforge.h`) is the product's contract — Node,
Swift, Rust, and any future binding depend on it. This is the policy that keeps
it trustworthy, plus the automated guard that enforces it.

## Versioning

- **Pre-1.0 (now):** the ABI may still change between minor versions while the
  surface settles; `mlxforge_abi_version()` stays at `1`.
- **From 1.0:** the C ABI is **append-only within a major version**:
  - new functions may be added;
  - new fields may be appended to the **end** of a struct (callers zero-init, so
    a new trailing field defaults to a disabled value — this is how
    `mlxforge_sampling.json_schema` was added);
  - existing functions are never removed, renamed, or have their signature
    changed; struct fields are never reordered or repurposed.
  - A breaking change means a new major and, where needed, a new symbol name
    (e.g. `mlxforge_submit_chat2`) so the old one keeps working.
- `MLXFORGE_ABI_VERSION` / `mlxforge_abi_version()` is bumped on any **additive**
  change so a consumer can feature-detect at runtime.

## Contract rules (already enforced by the header)

- No C++ types cross the boundary; all strings are UTF-8 and caller-freed
  (`mlxforge_string_free` / `mlxforge_floats_free`).
- Every fallible call reports through a `char** err`; a C++ exception is never
  allowed to unwind across `extern "C"`.

## The automated guard

`scripts/check-abi.sh` runs in the release workflow (and can be run locally). It:

1. fails if the library references **cpp-httplib** (the server harness leaking
   into the product library);
2. diffs the dylib's exported `mlxforge_*` symbols against the committed baseline
   `cmake/abi-baseline.txt` and **fails if any baseline symbol is missing** —
   i.e. a removal/rename. New symbols are reported but allowed.

```sh
cmake -S . -B build && cmake --build build --target mlxforge_shared
scripts/check-abi.sh build/libmlxforge.dylib
```

## Adding to the ABI (checklist)

1. Append the function/struct-field to `src/capi/mlxforge.h` (fields at the end).
2. Implement it in `src/capi/mlxforge.cpp` (try/catch → `err`).
3. Bump `MLXFORGE_ABI_VERSION`.
4. Refresh the baseline: `UPDATE_BASELINE=1 scripts/check-abi.sh build/libmlxforge.dylib`
   and commit `cmake/abi-baseline.txt`.
5. Extend the bindings (Node/Swift/Rust) and a `tests/capi` case.
