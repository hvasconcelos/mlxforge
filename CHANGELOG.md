# Changelog

All notable changes to **mlxforge** are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **First-class Qwen3-Embedding support.** A bare `engine.embed(text)` now
  self-selects the checkpoint's embedding convention: the engine sniffs the
  sentence-transformers pooling sidecar (`1_Pooling/config.json`) and, for a
  `pooling_mode_lasttoken` model (Qwen3-Embedding), defaults to **last-token
  pooling** with a **trailing EOS** (`<|endoftext|>`); plain LLMs keep mean
  pooling. Retrieval queries take an instruction that renders as
  `Instruct: {instruction}\nQuery:{text}` (Qwen's `get_detailed_instruct`).
- **Backbone-root weight layout.** The loader normalizes checkpoints saved at the
  decoder root (`layers.N.*` / `embed_tokens` / `norm`, no `model.` prefix or
  `lm_head`) to the canonical `model.*` form, so the official
  `Qwen/Qwen3-Embedding-0.6B` repo loads directly.
- **C ABI v2** — `mlxforge_embed_ex` + `mlxforge_embed_opts` (pooling / add_eos /
  skip_normalize / instruction; `-1` defers to the model default). `mlxforge_embed`
  is unchanged. The Node, Swift, and Rust bindings expose the same options.
- **Embedding harnesses** — a `mlxforge-cli embed` subcommand and an
  OpenAI-compatible `POST /v1/embeddings` server endpoint (string or array input).
- **Golden gate** — `reference/dump_ref.py --model qwen3_embedding` dumps the real
  model's pooled query/document vectors + token ids; `tests/runtime/embedding_test.cpp`
  asserts the C++ pooled vector and tokenization (no-BOS + appended EOS) match.

## [0.1.0] - 2026-06-09

First release of **`libmlxforge`** — an embeddable, continuously batched LLM
inference engine for Apple Silicon, built from scratch in C++ on the Apple MLX
C++ core. It is the only MLX project that is a *complete, batched* engine
(scheduler + continuous batching + own tokenizer/GGUF/chat templates) designed to
be embedded **in-process** and driven from other languages through a stable C ABI.

### The library (the product)

- **Stable C ABI** (`src/capi/mlxforge.h`, ABI v1) — a small, append-only,
  versioned `extern "C"` surface that never throws across the boundary. Create an
  engine from a model spec, submit chat/text requests, and stream tokens:
  `mlxforge_engine_create` / `_ready` / `_model_name`, `mlxforge_submit_chat` /
  `_submit_text`, `mlxforge_request_next` (token streaming) / `_cancel` /
  `_finish_reason`, plus `mlxforge_embed` for embeddings. Runtime version and ABI
  introspection via `mlxforge_version` / `mlxforge_abi_version`.
- **Continuous batching scheduler** — many concurrent requests share one decode
  loop with a single `async_eval` per step over the whole batch, so throughput
  scales with load instead of one stream at a time.
- **Embeddings** with pooling, exposed directly through the C ABI.
- **Structured / constrained output** for reliable JSON and tool-call generation.
- **Language bindings** on top of the C ABI: **Node**, **Swift**, and **Rust**.
- **Lean dylib** — the released artifact builds with the HTTP server and CLI
  harnesses **off** (no httplib/curl), shipping just the engine. Versioned dylib
  (`VERSION 0.1.0`, `SOVERSION 0`) with distribution packaging, an ABI guard, and
  a conformance kit to keep bindings honest against the ABI.

### Models & formats

- **LLaMA-family decoder-only transformers**: Llama-3.2, Mistral, Qwen3 (dense),
  Qwen3 MoE (sparse mixture-of-experts), and Qwen3.5 hybrid (Gated-DeltaNet, text).
- **Weight formats**: HuggingFace safetensors (fp16 / 4-bit, mixed-bit) and
  **GGUF** (Q4_0/Q4_1/Q8_0, Q4_K/Q5_K/Q6_K) with per-weight quantization, plus
  automatic HuggingFace download and hub-cache resolution.
- **Own tokenizers** behind an `EncoderBackend` interface — byte-level BPE
  (Llama/Qwen) and SentencePiece-BPE (Gemma) — byte-validated against mlx-lm
  golden ids. Chat templates for Llama and ChatML (with Qwen `enable_thinking`).
- **Sampling**: temperature, top-p, min-p, and repetition / frequency / presence
  penalties.

### Correctness

- **Golden-reference gated.** The forward pass, KV cache, RoPE (llama3-scaled),
  and sampling are validated against `mlx-lm` `.npy` fixtures committed under
  `reference/`, guarding against the engine's defining failure mode — silent
  numerical garbage rather than a crash.

### Harnesses (dev/QA only)

- An HTTP server (OpenAI- and Anthropic-compatible endpoints, tool/function
  calling) and a CLI exist to exercise and prove engine stability. They are not
  part of the shipped library.

[0.1.0]: https://github.com/hvasconcelos/libmlxforge/releases/tag/0.1.0
