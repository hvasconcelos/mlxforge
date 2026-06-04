# mlxforge

A from-scratch Local inference engine in **C++ on Apple MLX**, served behind an
**OpenAI-compatible HTTP API** with **continuous batching**.

mlxforge loads raw safetensors weights, runs a numerically-correct transformer
forward pass on the Metal GPU, and serves concurrent users through a vLLM-style
single-worker / three-queue scheduler. Every numerically-sensitive phase is
validated against an `mlx-lm` golden reference, because the failure mode here is
**silent garbage, not a crash**.

Primary model: `mlx-community/Llama-3.2-1B-Instruct` (fp16 by default; optional
4-bit). 16 layers, hidden 2048, 32 query / 8 KV heads (GQA), head_dim 64,
RMSNorm, RoPE (llama3 scaling), SwiGLU, tied embeddings.

The forward pass is architecture-shared across the LLaMA family. It runs
Llama-3.2 and Qwen3 dense models today, from safetensors (fp16 / 4-bit) or a
single-file GGUF. See [Supported models](#supported-models).

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
- **C++ tokenizer** — a from-scratch byte-level BPE over HF `tokenizer.json`
  (no Rust), the Llama-3.2 chat template (selected from `config.json`'s
  `model_type`), and UTF-8-safe incremental detokenization.
- **OpenAI server** (cpp-httplib) — `/v1/chat/completions`, `/v1/completions`,
  `/v1/models`, `/health`; non-streaming and SSE streaming; tool / function
  calling (`tools` / `tool_choice` → `tool_calls`); cancellation on client
  disconnect; per-request metrics; OpenAI-shaped errors (400/429/503).
- **Optional 4-bit quantization** — `quantized_matmul` (group_size 64), ~0.65
  GiB resident vs ~2.3 GiB fp16.
- **Configurable logging** (spdlog) — `debug`/`info`/`warn`/`error` across the
  engine, with the level, output file, and format controlled by `MLXFORGE_LOG_*`
  environment variables.

## Requirements

- Apple Silicon (the MLX Metal backend) + the Xcode **Metal Toolchain**
  (`xcodebuild -downloadComponent MetalToolchain`).
- CMake ≥ 3.24, a C++17 compiler (Apple clang).
- System **libcurl** (shipped in the macOS SDK; found via `find_package(CURL)`) —
  used to download models from the HuggingFace Hub.
- (Optional, for regenerating golden fixtures) Python 3.12 + `mlx-lm`.

All C++ dependencies (MLX, cpp-httplib, nlohmann/json, doctest, spdlog) are
fetched and pinned by CMake — see `cmake/Dependencies.cmake`. The tokenizer is
our own C++ byte-level BPE, so there is no Rust/`cargo` requirement.

## Build

```sh
cmake -S . -B build
cmake --build build --parallel
```

The first build compiles MLX's Metal kernels, so it takes a few minutes.
Outputs:

- `build/mlxforge` — the OpenAI HTTP server
- `build/mlxforge-cli` — CLI (weight dump + single-stream generation)
- `build/tests/mlxforge_tests` — the doctest suite

## Get the model

The server and CLI accept a model **spec** that is either a **HuggingFace repo
id** or a **local directory** — pass it directly, llama.cpp-style:

```sh
# a repo id: downloaded on first use, then cached and reused
./build/mlxforge-cli generate mlx-community/Llama-3.2-1B-Instruct-4bit "Hi" 32

# a local model dir (any folder with config.json + tokenizer.json + safetensors)
./build/mlxforge-cli generate /path/to/model "Hi" 32
```

A repo-id spec is resolved in this order:

1. an existing local directory containing `config.json` → used as-is;
2. an existing HuggingFace parent dir (`…/models--org--name`) → its
   `snapshots/<rev>/` is auto-resolved (so the cache parent path "just works");
3. a repo already in the standard HF hub cache → that snapshot is reused;
4. a repo already downloaded by mlxforge → that download is reused;
5. otherwise it is **downloaded** (via libcurl) into mlxforge's own cache.

The download cache defaults to `~/.cache/mlxforge` and is overridable with
`MLXFORGE_CACHE`. The HF hub cache is honored too (`HF_HUB_CACHE` / `HF_HOME`),
and a `HF_TOKEN` (or `HUGGING_FACE_HUB_TOKEN`) is sent for gated/private repos.

`MODEL_DIR` in the examples below is any such spec — a repo id like
`mlx-community/Llama-3.2-1B-Instruct-4bit`, or a local model directory.
You can still pre-download with `huggingface-cli` if you prefer:

```sh
huggingface-cli download mlx-community/Llama-3.2-1B-Instruct-bf16   # fp16
huggingface-cli download mlx-community/Llama-3.2-1B-Instruct-4bit   # 4-bit
```

## Supported models

| Family | Example repo | Precision | Chat format |
| --- | --- | --- | --- |
| Llama-3.2 | `mlx-community/Llama-3.2-1B-Instruct-bf16` | fp16 (cast on load) | `<\|start_header_id\|>…` |
| Llama-3.2 | `mlx-community/Llama-3.2-1B-Instruct-4bit` | 4-bit (mixed bits ok) | `<\|start_header_id\|>…` |
| Llama-3.2 (GGUF) | `bartowski/Llama-3.2-1B-Instruct-GGUF` | Q4_0/Q4_1/Q8_0, Q4_K/Q5_K/Q6_K | `<\|start_header_id\|>…` |
| Qwen3 (dense) | `mlx-community/Qwen3-0.6B-bf16` | fp16 (cast on load) | ChatML (`<\|im_start\|>…`) |
| Qwen3 (dense) | `mlx-community/Qwen3-4B-4bit` | 4-bit (mixed bits ok) | ChatML (`<\|im_start\|>…`) |
| Qwen3 (GGUF) | `Qwen/Qwen3-0.6B-GGUF` | Q4_0/Q4_1/Q8_0, Q4_K/Q5_K/Q6_K | ChatML (`<\|im_start\|>…`) |
| Qwen3 (MoE) | `mlx-community/Qwen3-30B-A3B-4bit` | 4-bit / fp16 (mixed bits ok) | ChatML (`<\|im_start\|>…`) |

The transformer (the `DecoderModel` base in `model/`) is family-shared, and the chat template is
selected from `config.json`'s `model_type` with BOS / special-token handling
driven by `config.json` + `tokenizer.json` (no hard-coded ids). **Qwen3 dense**
models (0.6B–32B) run end-to-end: their three deltas over Llama-3.2 — per-head
**QK-Norm**, the **ChatML** template (with an `enable_thinking` toggle), and
single-digit number pre-tokenization — are all handled automatically. Qwen3 has
no BOS token. **Qwen3 MoE** models (e.g. 30B-A3B) also run: on the MoE layers the
dense SwiGLU is replaced by a routed top-k mixture of experts (gather matmul, dense
or quantized). The 11B/90B Llama vision variants are **not** supported. Loading is
org-agnostic — any
HuggingFace repo with safetensors weights, the canonical HF key layout, and a
byte-level BPE `tokenizer.json` runs, so the 3B Llama-3.2 siblings and other
quant formats work too. See [`doc/supported-models.md`](./doc/supported-models.md)
for the full compatibility matrix.

The model is gated against its `mlx-lm` golden reference. Regenerate the
fixtures (rarely needed) with:

```sh
reference/.venv/bin/python reference/dump_ref.py --model llama     # -> reference/fixtures/
reference/.venv/bin/python reference/dump_ref.py --model qwen3     # -> reference/fixtures_qwen3/
```

## Run the server

```sh
./build/mlxforge -m "$MODEL_DIR" --port 8080 --max-ctx 8192 --max-waiting 256
```

Then use the official `openai` client:

```python
from openai import OpenAI
c = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="x")

# non-streaming
r = c.chat.completions.create(model="mlxforge",
    messages=[{"role": "user", "content": "What is the capital of France?"}],
    max_tokens=32)
print(r.choices[0].message.content)            # "The capital of France is Paris."

# streaming
for ev in c.chat.completions.create(model="mlxforge",
        messages=[{"role": "user", "content": "Tell me a joke."}],
        max_tokens=64, stream=True):
    print(ev.choices[0].delta.content or "", end="", flush=True)
```

Config knobs are also read from the environment (`MLXFORGE_HOST`, `MLXFORGE_PORT`,
`MLXFORGE_MAX_CTX`, `MLXFORGE_MAX_WAITING`, `MLXFORGE_KV_BUDGET`). `SIGINT`/`SIGTERM`
trigger a graceful shutdown that drains in-flight requests.

## Run the CLI

```sh
# stream generated text from a chat prompt
./build/mlxforge-cli generate "$MODEL_DIR" "What is the capital of France?" 64

# generate from a pre-tokenized .npy prompt (ids printed/streamed)
./build/mlxforge-cli generate "$MODEL_DIR" reference/fixtures/prompt_0_ids.npy 20

# inspect weights: key -> shape -> dtype, assert fp16, report peak memory
./build/mlxforge-cli dump-weights "$MODEL_DIR"
```

## Logging

Both binaries log through [spdlog](https://github.com/gabime/spdlog) to
**stderr** (so stdout stays clean for the CLI's generated text / weight dumps).
Verbosity and output are controlled by environment variables; an unset variable
uses the default shown:

| Env | Default | Meaning |
| --- | --- | --- |
| `MLXFORGE_LOG_LEVEL` | `info` | `trace` \| `debug` \| `info` \| `warn` \| `error` \| `critical` \| `off`. An unrecognized value falls back to `info`. |
| `MLXFORGE_LOG_FILE` | _(unset)_ | If set, logs are **appended** to this file in addition to the console. |
| `MLXFORGE_LOG_PATTERN` | `[%H:%M:%S.%e] [%^%l%$] %v` | spdlog [format pattern](https://github.com/gabime/spdlog/wiki/Custom-formatting). |

```sh
# verbose decode-loop / scheduler / request tracing
MLXFORGE_LOG_LEVEL=debug ./build/mlxforge -m "$MODEL_DIR"

# errors only, also tee'd to a file
MLXFORGE_LOG_LEVEL=error MLXFORGE_LOG_FILE=/var/log/mlxforge.log ./build/mlxforge -m "$MODEL_DIR"
```

`info` includes lifecycle events (model load, server listen/stop) and the
per-request metrics (TTFT, tokens/s, batch size, queue depth). See
[`.env.example`](./.env.example) for all runtime environment variables.

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
| `core/weights` | load safetensors (single/sharded), sanitize keys, fp16-cast, tag per-weight quant params |
| `core/gguf` | load a self-contained `.gguf` (config + tokenizer + weights from metadata); own file parser (no `mx::load_gguf`), ggml→HF name remap, q/k un-permute, own Q4_0/Q4_1/Q8_0 + Q4_K/Q5_K/Q6_K readers |
| `core/model_source` | resolve a model spec (local dir, `.gguf` file, or HF repo id) to a path; HF-cache reuse + download |
| `core/hf_download` | download HF repos via libcurl (list, filter, fetch through CDN redirect, atomic rename) |
| `core/env` | `env_or`/`env_long` environment-variable helpers |
| `model/` | the transformer: `DecoderModel` base (embedding, RMSNorm, RoPE, GQA SDPA, SwiGLU, LM head; fp16 + quantized paths; single-stream and batched forward) + `LlamaModel`/`Qwen3Model`/`Qwen3MoeModel` subclasses + `create_model` factory |
| `cache/kv_cache` | single-sequence KV cache |
| `cache/batch_kv_cache` | batched, left-padded KV cache: `update_and_fetch`, `filter`, `merge`, `pad_dummies` |
| `cache/kv_budget` | KV memory projection / admission gate |
| `sample/sampler` | greedy / temperature / top-k / top-p as MLX graph ops |
| `scheduler/request` | `Request` + bounded `TokenQueue` |
| `scheduler/scheduler` | the waiting queue + handoff |
| `runtime/worker` | the single GPU worker: admit / decode / evict loop |
| `runtime/batching` | prefill pass + batch-size bucketing |
| `runtime/single_stream` | the CLI's greedy generation loop |
| `tokenizer/tokenizer` | tokenizer facade: chat template, streaming detokenizer |
| `tokenizer/bpe` | from-scratch byte-level BPE (encode/decode) over `tokenizer.json` |
| `server/openai` | OpenAI request parse + response serialize (pure) |
| `server/http_server` | routes, blocking + SSE handlers, error shapes |
| `server/config` | CLI/env server configuration |

For the full design see the [`doc/`](./doc) folder:

- [`doc/architecture.md`](./doc/architecture.md) — engine architecture, the
  single-GPU-thread model, request lifecycle, continuous batching.
- [`doc/llm-architecture.md`](./doc/llm-architecture.md) — the transformer
  forward pass (embedding, RMSNorm, RoPE, GQA attention, SwiGLU, KV cache,
  sampling, quantization).
- [`doc/supported-models.md`](./doc/supported-models.md) — model families,
  adding a new one, the golden-reference discipline.
- [`doc/applications.md`](./doc/applications.md) — the server and CLI binaries
  and the OpenAI API surface.
- [`doc/contributing.md`](./doc/contributing.md) — build/test workflow,
  conventions, and the hard-won numerical gotchas.

## License

mlxforge is released under the [MIT License](./LICENSE).

Model weights are **not** covered by this license and remain subject to their
own terms (e.g. the Llama Community License for the Llama-3.2 weights).
