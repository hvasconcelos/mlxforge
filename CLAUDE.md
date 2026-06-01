# CLAUDE.md

Guidance for working in this repo with Claude Code.

## What this is

`xllm` — a from-scratch LLaMA inference engine in **C++ on Apple MLX** (the C++
core library, not `mlx-lm`), served behind an **OpenAI-compatible HTTP API** with
**continuous batching**. Apple Silicon only (Metal backend). C++17.

The defining constraint: the failure mode is **silent numerical garbage, not a
crash**. Anything touching the forward pass, KV cache, or sampling must be
validated against the `mlx-lm` golden reference, not eyeballed.

`SPECIFICATION.md` is the design; `STORIES.md` is the 25-story plan (all done).

## Build / test / run

```sh
cmake -S . -B build                       # configure (fetches + pins deps)
cmake --build build --parallel            # build xllm, xllm-cli, xllm_tests
cmake --build build --parallel --target xllm_tests   # tests only (faster)
ctest --test-dir build --output-on-failure           # run all tests
ctest --test-dir build -R XLLM-008                   # run one story's tests
./build/tests/xllm_tests --test-case="XLLM-013:*"    # run by name (verbose)
```

- The test binary is `build/tests/xllm_tests` (note the `tests/` subdir).
- MLX's Metal kernels and the Rust tokenizer crate make the *first* build slow
  (minutes); incremental rebuilds of `xllm_tests` are fast.
- To add a source file: add it to `src/` and to the `xllm_core` list in the
  top-level `CMakeLists.txt`; add tests to `tests/` and to `tests/CMakeLists.txt`.

## Environment / model

- Dependencies are pinned in `cmake/Dependencies.cmake` (MLX v0.31.2,
  cpp-httplib, doctest, tokenizers-cpp; nlohmann/json comes transitively from
  MLX). Bump deliberately.
- Needs the Xcode **Metal Toolchain** (`xcodebuild -downloadComponent
  MetalToolchain`) and `cargo`/Rust (tokenizers-cpp builds a Rust crate).
- Model: `mlx-community/Llama-3.2-1B-Instruct-bf16` (fp16) or `-4bit`. The fp16
  HF repo is **gated**; the `-bf16` repo is public and used as the fp16 source
  (cast to fp16 on load). The C++ engine and the Python reference load the
  **same** weights, keeping the golden reference self-consistent.
- Integration tests find the model by globbing the HF cache in
  `tests/CMakeLists.txt` (`XLLM_MODEL_DIR`, `XLLM_MODEL_DIR_4BIT`). If the model
  is absent they **self-skip** (they pass with a `MESSAGE`), so a green
  `ctest` without the model only ran the pure-logic units — download the model
  to exercise the numerical/scheduler paths.

## Golden-reference workflow (critical)

Numerically-sensitive code is gated against `.npy` fixtures dumped from `mlx-lm`,
committed under `reference/fixtures/`.

```sh
# regenerate fixtures (rarely needed)
python3.12 -m venv reference/.venv
reference/.venv/bin/pip install mlx-lm numpy
reference/.venv/bin/python reference/dump_ref.py
```

- The throwaway venv (`reference/.venv`) and the model cache are gitignored; the
  tiny `.npy` fixtures are committed.
- The C++ side compares with `tests/support/reference.h`: `assert_close`
  (fp16 rel ~1e-2) and `assert_tokens_equal` (exact). When debugging a numerical
  mismatch, extend `dump_ref.py` to emit the intermediate tensor and assert
  against it — that's how the front-half (embeddings / post-norm / RoPE'd Q/K)
  bugs were localized.

## Conventions

- C++17, 2-space indent, ~100 col. Use the `mx::` alias (`namespace mx =
  mlx::core`). Prefer spelled-out MLX ops (`mx::add`, `mx::multiply`) over
  operators for clarity, matching existing code.
- Match the comment density / naming of surrounding code. Comments explain
  *why* (conventions, gotchas), not *what*.
- Tests use **doctest**, named `"XLLM-NNN: ..."` so `ctest -R` filters per story.
- After implementing a change, the repo's habit is: run `code-simplifier` on the
  new code, then verify build + `ctest` before committing. Commit messages start
  with the story id (`XLLM-NNN: ...`) and the trailer
  `Co-Authored-By: Claude ...`.

## Hard-won gotchas (read before touching these areas)

- **`array::data<T>()` reads the raw row-major buffer.** It ignores strides, so
  calling it on a lazy *view* (e.g. a `transpose`) reads the wrong elements.
  Always `mx::contiguous(...)` (or `astype` to a fresh array) before `data<T>()`.
  The compare harness does this.
- **MLX arrays are thread-bound.** GPU work must run on the thread that created
  the arrays. The `Worker` therefore *loads the model on its own thread* and is
  the only thread that calls `eval`/`async_eval`. Other threads touch only their
  own `Request`. Don't call MLX ops off the worker thread.
- **One `async_eval` per decode step, over the whole batch** — never per-row or
  per-layer. This is the highest-severity invariant; `Worker::decode_steps()`
  exists to prove it (steps ≪ tokens under load).
- **`tokenizers-cpp` is not thread-safe** — it stashes the last encode/decode
  result in the handle. `Tokenizer` guards `encode`/`decode` with a shared mutex;
  keep it.
- **Masks are additive fp16, never boolean** (avoids MLX bug #2894). See
  `LlamaModel::batch_mask`.
- **RoPE is llama3-scaled** — `compute_rope_freqs` mirrors `mlx_lm`'s
  `Llama3RoPE` exactly and feeds `fast::rope` via `freqs` (base disabled). It's
  validated against `reference/fixtures/rope_freqs.npy`; don't "simplify" the
  math.
- **Decode-with-cache vs full-recompute logits differ by fp16 accumulation
  order** — compare argmax / exact tokens, not raw logits at tight tolerance.

## Where things live

`src/{core,model,cache,sample,scheduler,runtime,server,tokenizer}`, apps in
`apps/` (`xllm` server, `xllm-cli`), tests mirror the module path under `tests/`,
shared test helpers in `tests/support/`. See the table in `README.md` for the
per-module responsibilities.
