# xllm

A from-scratch LLaMA inference engine in **C++ on Apple MLX**, served behind an
**OpenAI-compatible HTTP API** with **continuous batching**.

xllm loads raw safetensors weights, runs a numerically-correct transformer
forward pass on the Metal GPU, and serves concurrent users through a vLLM-style
single-worker / three-queue scheduler. Every numerically-sensitive phase is
validated against an `mlx-lm` golden reference, because the failure mode here is
**silent garbage, not a crash**.

Target model: `mlx-community/Llama-3.2-1B-Instruct` (fp16 by default; optional
4-bit). 16 layers, hidden 2048, 32 query / 8 KV heads (GQA), head_dim 64,
RMSNorm, RoPE (llama3 scaling), SwiGLU, tied embeddings.

## Features

- **Numerically correct** — forward-pass logits and greedy tokens match `mlx-lm`
  (golden-reference `.npy` fixtures gate every step).
- **KV cache** — single-sequence and batched (`BatchKVCache`), left-padded,
  grown in 256-token blocks, with `filter` (eviction) / `merge` (admission).
- **Continuous batching** — one GPU worker thread owns all MLX state and is the
  only caller of `eval`/`async_eval`; exactly **one `async_eval` per decode
  step** over the whole batch; batch-size bucketing so the graph shape recurs.
- **Sampling as graph ops** — greedy, temperature, top-k, top-p (no host
  readback of logits).
- **C++ tokenizer** — HF `tokenizer.json` via `tokenizers-cpp`, the Llama-3.2
  chat template, and UTF-8-safe incremental detokenization.
- **OpenAI server** (cpp-httplib) — `/v1/chat/completions`, `/v1/completions`,
  `/v1/models`, `/health`; non-streaming and SSE streaming; cancellation on
  client disconnect; per-request metrics; OpenAI-shaped errors (400/429/503).
- **Optional 4-bit quantization** — `quantized_matmul` (group_size 64), ~0.65
  GiB resident vs ~2.3 GiB fp16.

## Requirements

- Apple Silicon (the MLX Metal backend) + the Xcode **Metal Toolchain**
  (`xcodebuild -downloadComponent MetalToolchain`).
- CMake ≥ 3.24, a C++17 compiler (Apple clang).
- `cargo` / Rust — `tokenizers-cpp` builds the Rust HF `tokenizers` crate.
- (Optional, for regenerating golden fixtures) Python 3.12 + `mlx-lm`.

All C++ dependencies (MLX, cpp-httplib, nlohmann/json, doctest, tokenizers-cpp)
are fetched and pinned by CMake — see `cmake/Dependencies.cmake`.

## Build

```sh
cmake -S . -B build
cmake --build build --parallel
```

The first build compiles MLX's Metal kernels and the Rust tokenizer crate, so it
takes a few minutes. Outputs:

- `build/xllm` — the OpenAI HTTP server
- `build/xllm-cli` — CLI (weight dump + single-stream generation)
- `build/tests/xllm_tests` — the doctest suite

## Get the model

```sh
# fp16 (full precision)
huggingface-cli download mlx-community/Llama-3.2-1B-Instruct-bf16
# or 4-bit
huggingface-cli download mlx-community/Llama-3.2-1B-Instruct-4bit
```

`MODEL_DIR` below is the resolved snapshot directory under
`~/.cache/huggingface/hub/.../snapshots/<rev>` (or any local dir containing
`config.json`, `tokenizer.json`, and `model.safetensors`).

## Run the server

```sh
./build/xllm "$MODEL_DIR" --port 8080 --max-ctx 8192 --max-waiting 256
```

Then use the official `openai` client:

```python
from openai import OpenAI
c = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="x")

# non-streaming
r = c.chat.completions.create(model="xllm",
    messages=[{"role": "user", "content": "What is the capital of France?"}],
    max_tokens=32)
print(r.choices[0].message.content)            # "The capital of France is Paris."

# streaming
for ev in c.chat.completions.create(model="xllm",
        messages=[{"role": "user", "content": "Tell me a joke."}],
        max_tokens=64, stream=True):
    print(ev.choices[0].delta.content or "", end="", flush=True)
```

Config knobs are also read from the environment (`XLLM_HOST`, `XLLM_PORT`,
`XLLM_MAX_CTX`, `XLLM_MAX_WAITING`, `XLLM_KV_BUDGET`). `SIGINT`/`SIGTERM`
trigger a graceful shutdown that drains in-flight requests.

## Run the CLI

```sh
# stream generated text from a chat prompt
./build/xllm-cli generate "$MODEL_DIR" "What is the capital of France?" 64

# generate from a pre-tokenized .npy prompt (ids printed/streamed)
./build/xllm-cli generate "$MODEL_DIR" reference/fixtures/prompt_0_ids.npy 20

# inspect weights: key -> shape -> dtype, assert fp16, report peak memory
./build/xllm-cli dump-weights "$MODEL_DIR"
```

## Tests

```sh
ctest --test-dir build --output-on-failure
```

Tests come in two tiers:

- **Unit tests** — pure logic, no GPU/weights (config parsing, sanitize,
  KV-cache bookkeeping, sampler math, request/response (de)serialization, SSE
  framing). Always run.
- **Golden-reference / integration tests** — numerically-sensitive stages
  validated against the `mlx-lm` reference dump (tensor-closeness / exact tokens)
  and the continuous-batching scheduler. These run only if the model is present
  locally; otherwise they self-skip (CMake globs the HF cache for the snapshot
  dir).

To (re)generate the golden fixtures (committed under `reference/fixtures/`):

```sh
python3.12 -m venv reference/.venv
reference/.venv/bin/pip install mlx-lm numpy
reference/.venv/bin/python reference/dump_ref.py
```

## Architecture

```
HTTP request ─▶ server/http_server (cpp-httplib)
                │  parse OpenAI JSON, apply chat template, tokenize
                ▼
            scheduler/  ── waiting queue (mutex + cv) ──▶ runtime/worker
                                                          (the ONE GPU thread)
                                                          │ owns weights,
                                                          │ BatchKVCache, sampler
                                                          ▼
   admit (prefill ─▶ merge) ─▶ decode step (1 async_eval/batch) ─▶ evict (filter)
                                                          │
   each row's tokens ─▶ per-request bounded TokenQueue ─▶ SSE / blocking response
```

Source layout (`src/`):

| Module | Responsibility |
|---|---|
| `core/config` | parse `config.json` into `ModelConfig` (incl. rope_scaling, quantization) |
| `core/weights` | load safetensors (single/sharded), sanitize keys, fp16-cast |
| `model/llama` | the transformer: embedding, RMSNorm, RoPE, GQA SDPA, SwiGLU, LM head; fp16 + quantized paths; single-stream and batched forward |
| `cache/kv_cache` | single-sequence KV cache |
| `cache/batch_kv_cache` | batched, left-padded KV cache: `update_and_fetch`, `filter`, `merge`, `pad_dummies` |
| `cache/kv_budget` | KV memory projection / admission gate |
| `sample/sampler` | greedy / temperature / top-k / top-p as MLX graph ops |
| `scheduler/request` | `Request` + bounded `TokenQueue` |
| `scheduler/scheduler` | the waiting queue + handoff |
| `runtime/worker` | the single GPU worker: admit / decode / evict loop |
| `runtime/batching` | prefill pass + batch-size bucketing |
| `runtime/single_stream` | the CLI's greedy generation loop |
| `tokenizer/tokenizer` | `tokenizers-cpp` wrapper, chat template, streaming detokenizer |
| `server/openai` | OpenAI request parse + response serialize (pure) |
| `server/http_server` | routes, blocking + SSE handlers, error shapes |
| `server/config` | CLI/env server configuration |

See `SPECIFICATION.md` for the full design and `STORIES.md` for the 25-story
implementation breakdown.

## License

Research/educational. Model weights are subject to their own licenses (Llama
Community License for the underlying Llama-3.2 weights).
