# Contributing & maintainer guide

This is the practical guide to building, testing, and safely modifying mlxforge.
Read the [architecture](./architecture.md) and [forward-pass](./llm-architecture.md)
docs first for the *what*; this document is the *how*.

## Prerequisites

- **Apple Silicon** — the MLX Metal backend is the only supported target; the
  build hard-errors on non-Apple platforms.
- The Xcode **Metal Toolchain**: `xcodebuild -downloadComponent MetalToolchain`.
- **CMake ≥ 3.24** and a C++17 compiler (Apple clang).
- *(Optional, only to regenerate golden fixtures)* Python 3.12 + `mlx-lm`.

All C++ dependencies are fetched and pinned by CMake; you don't install them
manually. See `cmake/Dependencies.cmake` for the exact pins (MLX v0.31.2,
cpp-httplib v0.46.1, doctest v2.5.2, spdlog v1.15.3; nlohmann/json is reused
transitively from MLX). Bump pins deliberately. The tokenizer is our own C++
byte-level BPE, so there is no Rust/`cargo` requirement.

## Build, test, run

```sh
cmake -S . -B build                       # configure (fetches + pins deps)
cmake --build build --parallel            # build mlxforge, mlxforge-cli, mlxforge_tests
cmake --build build --parallel --target mlxforge_tests   # tests only (faster)
ctest --test-dir build --output-on-failure               # run all tests
ctest --test-dir build -R kv_cache                       # run a subset by name
./build/tests/mlxforge_tests --test-case="*forward*"         # run by name (verbose)
```

- The test binary is `build/tests/mlxforge_tests` (note the `tests/` subdir).
- MLX's Metal kernels make the **first** build slow (minutes); incremental
  rebuilds of `mlxforge_tests` are fast.

### Adding a source file or a test

- **Source:** add it under `src/` and to the `mlxforge_core` source list in the
  top-level `CMakeLists.txt`.
- **Test:** add it under `tests/` (mirroring the module path, e.g.
  `tests/cache/foo_test.cpp`) and to the `mlxforge_tests` source list in
  `tests/CMakeLists.txt`. Tests use **doctest**; every `TEST_CASE` is
  auto-registered with `ctest` via `doctest_discover_tests`.

## The golden-reference discipline (most important)

The failure mode here is **silent numerical garbage, not a crash**. Every
numerically-sensitive stage is gated against an `mlx-lm` golden reference: `.npy`
tensors dumped from `mlx-lm` on the *same* weights the C++ engine loads,
committed under `reference/fixtures/`.

```sh
# regenerate fixtures (rarely needed)
python3.12 -m venv reference/.venv
reference/.venv/bin/pip install mlx-lm numpy
reference/.venv/bin/python reference/dump_ref.py              # Llama (default)
```

- The throwaway venv (`reference/.venv`) and the HF model cache are gitignored;
  the tiny `.npy` fixtures are committed.
- The C++ side compares with `tests/support/reference.h`: `assert_close` (fp16
  rel ~1e-2) and `assert_tokens_equal` (exact).
- **Two test tiers.** Pure-logic unit tests (config parsing, key sanitize,
  KV-cache bookkeeping, sampler math, request/response (de)serialization, SSE
  framing) always run. The golden-reference / integration tests need the model
  present locally; CMake globs the HF cache for the snapshot dir
  (`MLXFORGE_MODEL_DIR`, `MLXFORGE_MODEL_DIR_4BIT`)
  and the tests **self-skip** (pass with a message) if it's absent. A green
  `ctest` *without* the model has only run the pure-logic units — download a
  model to exercise the numerical and scheduler paths.

**Debugging a numerical mismatch:** extend `reference/dump_ref.py` to emit the
intermediate tensor you suspect, and assert against it from C++. Bisecting the
forward pass this way (embedding → post-norm → RoPE'd Q/K → block output →
logits) is how the front-half bugs were originally localized.

## Coding conventions

- C++17, 2-space indent, ~100 columns.
- Use the `mx::` alias (`namespace mx = mlx::core`).
- Prefer spelled-out MLX ops (`mx::add`, `mx::multiply`) over operators, for
  clarity and to match existing code.
- Match the comment density and naming of the surrounding code. Comments explain
  **why** (conventions, gotchas, non-obvious invariants), not **what**.
- After a change, the repo habit is: run the `code-simplifier` pass over the new
  code, then verify the build and `ctest` before committing.

## Hard-won numerical gotchas

Read these before touching the forward pass, the KV cache, the worker, or the
tokenizer. Each one is a bug that does not announce itself.

- **`array::data<T>()` reads the raw row-major buffer — it ignores strides.**
  Calling it on a lazy *view* (e.g. a `transpose`) reads the wrong elements.
  Always `mx::contiguous(...)` (or `astype` to a fresh array) before `data<T>()`.
  The compare harness does this.
- **MLX arrays are thread-bound.** GPU work must run on the thread that created
  the arrays. The `Worker` therefore *loads the model on its own thread* and is
  the only thread that calls `eval`/`async_eval`. Other threads touch only their
  own `Request`. Never call an MLX op off the worker thread.
- **One `async_eval` per decode step, over the whole batch** — never per-row,
  never per-layer. This is the highest-severity invariant; `Worker::decode_steps()`
  exists to prove it (the step count stays far below the token count under load).
- **The tokenizer is our own byte-level BPE** (`tokenizer/bpe.{h,cpp}`),
  validated to byte-match the HF tokenizer against committed mlx-lm golden ids
  (`reference/fixtures/tokenizer_corpus.json`). It is pure/`const`/thread-safe
  (no mutex) and currently supports Llama-3.2-style byte-level BPE only —
  `Tokenizer::from_file` throws on other families (e.g. SentencePiece-based ones).
- **Masks are additive fp16, never boolean** (avoids MLX bug #2894). See
  `LlamaModel::batch_mask`.
- **RoPE is llama3-scaled.** `compute_rope_freqs` mirrors `mlx_lm`'s `Llama3RoPE`
  exactly and feeds `fast::rope` via `freqs` (the analytic base disabled). It is
  validated against the reference; don't "simplify" the math.
- **Decode-with-cache vs full-recompute logits differ by fp16 accumulation
  order.** Compare argmax / exact tokens, not raw logits at a tight tolerance.
- **Left-padding is load-bearing.** The batched KV cache left-pads so every active
  row appends its new token at the same physical column; the per-row `offset`
  array carries each row's true RoPE position. Don't switch to right-padding.

## Where to look

- Module responsibilities: the table in [architecture.md](./architecture.md#module-map)
  (and the one in the top-level `README.md`).
- The forward pass and KV cache: [llm-architecture.md](./llm-architecture.md).
- The fixture catalogue (shapes/dtypes/what each gates): `reference/README.md`.
- Project-level working notes for Claude Code: the top-level `CLAUDE.md`.
