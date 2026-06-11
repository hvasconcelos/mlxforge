# CLAUDE.md

Guidance for working in this repo with Claude Code.

## What this is

`mlxforge` — a from-scratch Local Inference engine in **C++ on Apple MLX** (the C++
core library, not `mlx-lm`), with **continuous batching**. Apple Silicon only (Metal
backend). C++17. Runs LLaMA-family decoder models (Llama-3.2, Qwen3 dense/MoE,
Qwen3.5 hybrid) and **Qwen3-VL** vision-language (image → text: a from-scratch ViT
encoder + image merge + interleaved M-RoPE + DeepStack, served single-stream).

The defining constraint: the failure mode is **silent numerical garbage, not a
crash**. Anything touching the forward pass, KV cache, or sampling must be
validated against the `mlx-lm` (or `mlx-vlm`, for Qwen3-VL) golden reference, not
eyeballed.

**The product is `libmlxforge` — the engine as an embeddable library.** mlxforge is
the only MLX project that is a *complete, batched* LLM engine (scheduler + continuous
batching + own tokenizer/GGUF/chat templates, golden-reference-gated) meant to be
**bound from other languages** through a stable C ABI (Node first, then Swift/Rust),
not just another Python/Swift app. The single-stream MLX libs (`node-mlx`, Apple's
`MLXLLM`) have no batching; the batched MLX servers (`vllm-mlx`, `omlx`) are Python and
can't be embedded. mlxforge sits in that gap: batched + language-neutral + in-process.

**Product hierarchy — read this before adding features.** The **library** (the
forthcoming `extern "C"` surface in `src/capi/mlxforge.h`, plus its bindings) is the
product. The **HTTP server** (`src/server/*`) and the **CLI** (`apps/mlxforge_cli.cpp`)
are **auxiliary QA harnesses** that exist to *exercise and prove the engine's
stability*: the server is the scheduler/batching concurrency & load harness, the CLI is
the golden-reference and weight-inspection smoke test. They are dev/QA instruments, not
user-facing deliverables. A change that only makes the server nicer is out of scope; a
change that hardens or validates the library is in scope. The released library artifact
builds with the harnesses **off** (a lean dylib, no httplib/curl).

The `doc/` folder is the design reference: `doc/embedding.md` (the library /
cross-language engine thesis + C-ABI quickstart — start here), `doc/architecture.md`
(engine internals + threading), `doc/llm-architecture.md` (the transformer forward
pass), `doc/supported-models.md`, `doc/applications.md` (the server/CLI harnesses), and
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
- **The tokenizers are our own**, behind the `EncoderBackend` interface and
  validated to byte-match the HF tokenizer against committed mlx-lm golden ids
  (`reference/fixtures*/tokenizer_corpus.json`, regenerated by `dump_ref.py`).
  Two backends: byte-level BPE (`tokenizer/bpe.{h,cpp}`, `BpeTokenizer`; Llama-3.2
  / Qwen) and SentencePiece-BPE (`tokenizer/spm.{h,cpp}`, `SpmBpeTokenizer`;
  metaspace + `byte_fallback`, e.g. Gemma — tokenizer-only, no Gemma model class).
  Both are pure/`const`/thread-safe (no mutex). `Tokenizer::from_file` throws on
  still-unimplemented families (e.g. Unigram/WordPiece).
- **Masks are additive fp16, never boolean** (avoids MLX bug #2894). See
  `DecoderModel::batch_mask`.
- **RoPE is llama3-scaled** — `compute_rope_freqs` mirrors `mlx_lm`'s
  `Llama3RoPE` exactly and feeds `fast::rope` via `freqs` (base disabled). It's
  validated against `reference/fixtures/rope_freqs.npy`; don't "simplify" the
  math.
- **Decode-with-cache vs full-recompute logits differ by fp16 accumulation
  order** — compare argmax / exact tokens, not raw logits at tight tolerance.
- **Quantized KV (kv_bits 8|4) mirrors mlx-lm's QuantizedKVCache** — triplet
  storage quantized at write time (`cache/kv_quant`), attention via the
  hand-rolled `quantized_sdpa` (`model/sdpa`, a port of mlx_lm base.py; MLX has
  no fused quantized SDPA kernel). Three traps: (1) the batched additive mask
  must be reshaped `(B,1,N,T)→(B,1,1,N,T)` under GQA, and masked columns must be
  **overridden** with `finfo(fp16).min`, never added — a fully-masked left-pad
  row makes NaN that `+(-inf)` cannot cancel; (2) quantized matmuls are
  **fusion-context-sensitive** (~1 logit shift between lazy and materialized
  inputs — mlx-lm disagrees with itself across graph contexts), so the golden
  gates are teacher-forced and margin-gated (`greedy_gaps_kvq*.npy`), never raw
  exact-stream asserts; (3) both caches deliberately share the block-grow +
  `slice_update` storage writer (`update_kv_components`) — buffer strides
  affect kernel accumulation order. Engine-wide setting, default off;
  vision/hybrid models are rejected at engine creation (no silent fp16
  fallback).
- **The prefix cache harvests PROMPT K/V only — never decode-produced K/V.**
  Decode-with-cache K/V differs from a recompute by fp16 accumulation order
  (the decode-vs-recompute gap below) and demonstrably flips later greedy
  choices; prefill-produced K/V is the proven exact-stable class, so pooling
  only it keeps the feature's gate (warm == cold, token-exact) sound.
  Multi-turn reuse still converges — the next turn's prompt contains the prior
  answer as text and pools after its own (seeded) prefill. The pool
  (`cache/block_pool`) stores immutable blocks keyed by a salted chain hash;
  matched blocks seed a batch-1 cache via `BatchKVCache::from_prefix`, written
  through the standard `update_kv_components` writer so buffer layout matches a
  cold prefill (strides are load-bearing, see kv-quant above). Harvest
  materializes copies (`mx::contiguous` + eval) — a lazy slice would pin the
  whole batch buffer. The SSD tier (`cache/block_store`) is byte-only across
  threads (worker does all array<->bytes conversion); its writer keeps the
  queue front visible to `get()`/`contains()` until the file lands, and its
  serialize order (per layer, K then V components) is gated by the exact-token
  spill test — an order mismatch produced silent garbage. Engine-wide opt-in;
  vision/hybrid models and spill-without-prefix-cache are rejected at engine
  creation. Multimodal rows are never harvested or matched.
- **Qwen3-VL interleaved M-RoPE can't use `fast::rope`** (it takes a 1D offset,
  not 3D `(t,h,w)` positions). `Qwen3VLModel` hand-rolls a half-split rotation
  with a per-frequency t/h/w selector; text tokens have `t==h==w` so it reduces to
  ordinary 1D RoPE (and a generated/decode token is a scalar position one past the
  prompt's max — it jumps over the image's spatial extent). Vision serving is
  **prefill-single, decode-batched** (like vLLM/omlx): the ViT + image-merge +
  3D-M-RoPE prefill runs single-stream (`runtime/multimodal_stream`), then the
  prompt's K/V is adopted into a batch-1 `BatchKVCache`
  (`BatchKVCache::from_single_sequence`) and **merged into the continuous-batching
  decode pool** (`Worker::admit_multimodal`) — a generated VL token is pure text
  (`t==h==w`), so it decodes through the ordinary batched forward alongside text
  rows. No per-row 3D positions are needed in the cache: the batched mask works in
  physical-slot space (`idx`/`left_padding`) while RoPE uses a *separate* per-row
  `offset`, so a VL row just carries `offset = max(3D position)+1` (well below its
  image-padded token count). The prefill itself stays single (the ViT can't batch
  ragged grids). Every vision stage is golden-gated against `mlx-vlm`
  (`reference/fixtures_qwen3_vl/`), and batched decode is gated equal to the
  single-stream stream (`tests/model/qwen3_vl_test.cpp`).

## Where things live

`src/{core,model,cache,sample,scheduler,runtime,server,tokenizer,capi,vision}`, apps
in `apps/` (`mlxforge` server, `mlxforge-cli`), tests mirror the module path under
`tests/`, shared test helpers in `tests/support/`. See the module table in
`README.md` and `doc/architecture.md` for the per-module responsibilities.

**Vision (Qwen3-VL):** `src/model/vision/vit.{h,cpp}` is the ViT encoder;
`src/model/qwen3_vl.{h,cpp}` is the fused model (image merge + interleaved M-RoPE +
DeepStack + cached decode); `src/vision/` does image decode (`stb_image`) +
preprocess (smart-resize/normalize/patchify); `runtime/multimodal_stream.{h,cpp}` is
the image→text path (single-stream prefill + batched-decode packaging). Selected by
`create_model` on a `vision_config`.

**Product-facing surface:** `src/capi/` is the stable `extern "C"` ABI
(`mlxforge.h`) wrapping `runtime/engine` — the public surface — and `bindings/`
holds the language bindings on top of it (`bindings/{node,swift,rust}`). These are
the primary entry points; `src/server` and `apps/` are harnesses behind them. The
ABI is append-only and guarded (`cmake/abi-baseline.txt`, `scripts/check-abi.sh`).
