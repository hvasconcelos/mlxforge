# CLAUDE.md

Guidance for working in this repo with Claude Code.

## What this is

`mlxforge` — a from-scratch LLaMA inference engine in **C++ on Apple MLX** (the C++
core library, not `mlx-lm`), served behind an **OpenAI-compatible HTTP API** with
**continuous batching**. Apple Silicon only (Metal backend). C++17.

The defining constraint: the failure mode is **silent numerical garbage, not a
crash**. Anything touching the forward pass, KV cache, or sampling must be
validated against the `mlx-lm` golden reference, not eyeballed.

The `doc/` folder is the design reference: `doc/architecture.md` (engine
internals + threading), `doc/llm-architecture.md` (the transformer forward
pass), `doc/supported-models.md`, `doc/applications.md`, and
`doc/contributing.md` (the maintainer guide).

## Build / test / run

```sh
cmake -S . -B build                       # configure (fetches + pins deps)
cmake --build build --parallel            # build mlxforge, mlxforge-cli, mlxforge_tests
cmake --build build --parallel --target mlxforge_tests   # tests only (faster)
ctest --test-dir build --output-on-failure           # run all tests
ctest --test-dir build -R kv_cache                   # run a subset by ctest name
./build/tests/mlxforge_tests --test-case="*forward*"     # run by name (verbose)
```

- The test binary is `build/tests/mlxforge_tests` (note the `tests/` subdir).
- MLX's Metal kernels make the *first* build slow (minutes); incremental
  rebuilds of `mlxforge_tests` are fast.
- To add a source file: add it to `src/` and to the `mlxforge_core` list in the
  top-level `CMakeLists.txt`; add tests to `tests/` and to `tests/CMakeLists.txt`.

## Environment / model

- Dependencies are pinned in `cmake/Dependencies.cmake` (MLX v0.31.2,
  cpp-httplib, doctest, spdlog v1.15.3; nlohmann/json comes transitively from
  MLX). Bump deliberately. System **libcurl** is a `find_package(CURL)` (not
  FetchContent), linked PRIVATE into `mlxforge_core` for HuggingFace downloads.
- Needs the Xcode **Metal Toolchain** (`xcodebuild -downloadComponent
  MetalToolchain`). The tokenizer is our own C++ byte-level BPE, so there is no
  longer a `cargo`/Rust build requirement.
- Model spec resolution: the CLI/server take a model **spec** (local dir *or* HF
  repo id) and call `mlxforge::resolve_model_dir` (`src/core/model_source.{h,cpp}`)
  once before loading. It uses a local dir as-is, auto-resolves an HF cache parent
  (`models--org--name`) to its `snapshots/<rev>/`, reuses the HF hub cache, then
  falls back to downloading via `src/core/hf_download.{h,cpp}` (libcurl, the only
  HTTP *client* in the tree) into `$MLXFORGE_CACHE` (default `~/.cache/mlxforge`).
  Env helpers live in `src/core/env.{h,cpp}` (`env_or`/`env_long`).
- Model: `mlx-community/Llama-3.2-1B-Instruct-bf16` (fp16) or `-4bit`. The fp16
  HF repo is **gated**; the `-bf16` repo is public and used as the fp16 source
  (cast to fp16 on load). The C++ engine and the Python reference load the
  **same** weights, keeping the golden reference self-consistent.
- Integration tests find the model by globbing the HF cache in
  `tests/CMakeLists.txt` (`MLXFORGE_MODEL_DIR`, `MLXFORGE_MODEL_DIR_4BIT`). If the model
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
- Tests use **doctest**, with descriptive `TEST_CASE` names; `ctest -R <regex>`
  filters by the discovered test name.
- Logging goes through `core/logging.h` (`mlxforge::log::{debug,info,warn,error}`,
  spdlog with fmt-style `{}`), not `printf`/`fprintf` — those stay only for
  actual program output on **stdout** (the CLI's generated text / weight dump).
  Logs go to **stderr**; level/file/pattern are env-driven (`MLXFORGE_LOG_LEVEL`,
  `MLXFORGE_LOG_FILE`, `MLXFORGE_LOG_PATTERN`). Hot-path detail is `debug`. Call
  `log::init()` once at the top of `main()`; it's idempotent.
- After implementing a change, the repo's habit is: run `code-simplifier` on the
  new code, then verify build + `ctest` before committing.

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
- **The tokenizer is our own byte-level BPE** (`tokenizer/bpe.{h,cpp}`,
  `BpeTokenizer`), validated to byte-match the HF tokenizer against committed
  mlx-lm golden ids (`reference/fixtures/tokenizer_corpus.json`, regenerated by
  `dump_ref.py`). It is pure/`const`/thread-safe (no mutex). It targets
  Llama-3.2-style byte-level BPE only; `Tokenizer::from_file` throws on other
  families (e.g. SentencePiece-based tokenizers, not yet implemented).
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
`apps/` (`mlxforge` server, `mlxforge-cli`), tests mirror the module path under `tests/`,
shared test helpers in `tests/support/`. See the module table in `README.md` and
`doc/architecture.md` for the per-module responsibilities.
