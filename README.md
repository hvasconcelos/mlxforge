# libmlxforge

**An embeddable, batched MLX LLM engine you bind from any language.**
A from-scratch local inference engine in **C++ on Apple MLX** with **continuous
batching**, exposed through a stable C ABI so Node (first), then Swift and Rust, can
embed the *same batched engine* in-process — not just array ops, and not one stream at a
time.

mlxforge loads raw safetensors / GGUF weights, runs a numerically-correct transformer
forward pass on the Metal GPU, and serves many concurrent requests through a vLLM-style
single-worker / three-queue scheduler. Every numerically-sensitive phase is validated
against an `mlx-lm` golden reference, because the failure mode here is **silent garbage,
not a crash** — the guarantee an embedded engine has to make.

> **The library is the product.** The OpenAI-compatible HTTP server and the CLI in this
> repo are **QA harnesses** that exercise and prove the engine's stability (the server
> load-tests the scheduler/batching; the CLI runs the golden-reference and weight
> smoke tests). They are dev/QA tools, not the deliverable — see
> [Validating the engine](#validating-the-engine-harnesses).

## Why mlxforge (vs the rest of the MLX ecosystem)

Apple Silicon already has plenty of ways to run a local model — but each existing option
is either an *array framework* (you build the engine yourself), a *single-stream* LLM
lib, or a *Python server* you can't embed. None is a complete, batched engine bindable
from another language. That's the gap mlxforge fills:

| Option | What it is | In-process | MLX-native | Batched / concurrent |
| --- | --- | :---: | :---: | :---: |
| `mlx-c` / `mlx-rs` / `mlx-rust` | array-framework bindings | ✅ | ✅ | build it yourself |
| **`node-llama-cpp`** | Node LLM lib (llama.cpp/GGUF) | ✅ | ❌ | ⚠️ weak |
| **`node-mlx`** | Node LLM lib (MLX) | ✅ | ✅ | ❌ single-stream |
| Apple **`MLXLLM`** (mlx-swift) | Swift LLM lib | ✅ | ✅ | ❌ single-stream |
| **`vllm-mlx`** / **`omlx`** | Python MLX servers | ❌ (HTTP) | ✅ | ✅ |
| Ollama (`+ ollama` npm) | Go sidecar (MLX since 0.19) | ❌ (HTTP) | ✅ | ✅ |
| **mlxforge** | **batched engine + C ABI** | ✅ | ✅ | ✅ |

mlxforge is the only row that is **in-process + MLX-native + batched** — and the only
batched MLX engine you can bind from Node/Swift/Rust instead of running as a separate
Python/Go process. See [`doc/embedding.md`](./doc/embedding.md) for the full thesis and
the C-ABI / Node quickstart.

Primary model: `mlx-community/Llama-3.2-1B-Instruct` (fp16 by default; optional
4-bit). 16 layers, hidden 2048, 32 query / 8 KV heads (GQA), head_dim 64,
RMSNorm, RoPE (llama3 scaling), SwiGLU, tied embeddings.

The forward pass is architecture-shared across the LLaMA family. It runs
Llama-3.2, Qwen3 (dense / MoE), and Qwen3.5 hybrid models today, from safetensors
(fp16 / 4-bit) or a single-file GGUF. See [Supported models](#supported-models).

## Features

- **Embeddable batched engine** — `runtime/engine` is an HTTP-free, in-process engine
  (one GPU worker + scheduler + batched KV cache) designed to be wrapped by a stable
  `extern "C"` ABI (`src/capi/mlxforge.h`, *in progress*) and bound from other
  languages. Concurrent requests share one batched engine. This is the product.
- **Numerically correct** — forward-pass logits and greedy tokens match `mlx-lm`
  (golden-reference `.npy` fixtures gate every step). The guarantee an engine embedded
  in someone else's app has to make.
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
- **OpenAI server harness** (cpp-httplib) — a concurrency/load harness that drives the
  engine over HTTP: `/v1/chat/completions`, `/v1/completions`, `/v1/models`, `/health`;
  non-streaming and SSE streaming; tool / function calling (`tools` / `tool_choice` →
  `tool_calls`); cancellation on client disconnect; per-request metrics; OpenAI-shaped
  errors (400/429/503). Built only when `MLXFORGE_BUILD_SERVER=ON`; the released library
  ships without it.
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

- `build/libmlxforge.dylib` — **the product**: the engine behind the C ABI
  (`src/capi/mlxforge.h`); link this to embed mlxforge (see
  [`doc/embedding.md`](./doc/embedding.md))
- `build/mlxforge` — the OpenAI HTTP server *harness*
- `build/mlxforge-cli` — CLI *harness* (weight dump + single-stream generation)
- `build/tests/mlxforge_tests` — the doctest suite

Build options (the harnesses are on by default for development; the released
library is built with them off):

| Option | Default | Effect |
| --- | --- | --- |
| `MLXFORGE_BUILD_SHARED` | `ON` | build `libmlxforge.dylib` (the C-ABI product) |
| `MLXFORGE_BUILD_SERVER` | `ON` | build the HTTP server harness (pulls `cpp-httplib`) |
| `MLXFORGE_BUILD_CLI` | `ON` | build the CLI harness |
| `MLXFORGE_ENABLE_HF_DOWNLOAD` | `ON` | HuggingFace download (pulls `libcurl`) |

A lean library build — `-DMLXFORGE_BUILD_SERVER=OFF -DMLXFORGE_BUILD_CLI=OFF
-DMLXFORGE_ENABLE_HF_DOWNLOAD=OFF` — produces a `libmlxforge.dylib` with neither
`cpp-httplib` nor `libcurl` linked.

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
| Qwen3.5 (hybrid) | `mlx-community/Qwen3.5-0.8B-4bit` | 4-bit (mixed bits ok) | ChatML (`<\|im_start\|>…`) |

The transformer (the `DecoderModel` base in `model/`) is family-shared, and the chat template is
selected from `config.json`'s `model_type` with BOS / special-token handling
driven by `config.json` + `tokenizer.json` (no hard-coded ids). **Qwen3 dense**
models (0.6B–32B) run end-to-end: their three deltas over Llama-3.2 — per-head
**QK-Norm**, the **ChatML** template (with an `enable_thinking` toggle), and
single-digit number pre-tokenization — are all handled automatically. Qwen3 has
no BOS token. **Qwen3 MoE** models (e.g. 30B-A3B) also run: on the MoE layers the
dense SwiGLU is replaced by a routed top-k mixture of experts (gather matmul, dense
or quantized). **Qwen3.5 hybrid** models (e.g. 0.8B) run their text tower: the vision
tower is dropped and the Qwen3-Next-style decoder interleaves gated full-attention
layers (output gate + partial RoPE) with Gated-DeltaNet linear-attention layers
(`config.full_attention_interval`), served through a hybrid KV + recurrent-state cache.
The 11B/90B Llama vision variants are **not** supported. Loading is
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
reference/.venv/bin/python reference/dump_ref.py --model qwen3_5   # -> reference/fixtures_qwen3_5/
```

## Validating the engine (harnesses)

The server and CLI below are **not the product** — they are how we drive and validate
`libmlxforge`. The server is the scheduler/batching concurrency & load harness; the CLI
is the golden-reference and weight-inspection smoke test. The shippable library is the C
ABI in [`doc/embedding.md`](./doc/embedding.md); these binaries are built only with
`MLXFORGE_BUILD_SERVER` / `MLXFORGE_BUILD_CLI` on (the default for dev/CI).

### The server harness

```sh
./build/mlxforge -m "$MODEL_DIR" --port 8080 --max-ctx 8192 --max-waiting 256
```

It speaks the OpenAI API so existing clients and load tools can hammer the scheduler.
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

### The CLI harness

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

The request lifecycle below is drawn through the **server harness** (the most familiar
entry point); an embedder reaches the same `scheduler → worker` core directly through
the C ABI over `runtime/engine`, with no HTTP involved.

```
HTTP request ─▶ server/http_server (cpp-httplib)   ← harness; C-ABI callers skip this
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
| `model/` | the transformer: `DecoderModel` base (embedding, RMSNorm, RoPE, GQA SDPA, SwiGLU, LM head; fp16 + quantized paths; single-stream and batched forward) + `LlamaModel`/`Qwen3Model`/`Qwen3MoeModel`/`Qwen35Model` subclasses + `create_model` factory |
| `cache/kv_cache` | single-sequence KV cache |
| `cache/batch_kv_cache` | batched, left-padded KV cache: `update_and_fetch`, `filter`, `merge`, `pad_dummies` |
| `cache/kv_budget` | KV memory projection / admission gate |
| `sample/sampler` | greedy / temperature / top-k / top-p as MLX graph ops |
| `scheduler/request` | `Request` + bounded `TokenQueue` |
| `scheduler/scheduler` | the waiting queue + handoff |
| `runtime/worker` | the single GPU worker: admit / decode / evict loop |
| `runtime/batching` | prefill pass + batch-size bucketing |
| `runtime/single_stream` | the CLI harness's greedy generation loop |
| `runtime/engine` | **the embeddable engine** — HTTP-free object that resolves a spec, loads config/tokenizer, and boots the worker; what the C ABI wraps |
| `tokenizer/tokenizer` | tokenizer facade: chat template, streaming detokenizer |
| `tokenizer/bpe` | from-scratch byte-level BPE (encode/decode) over `tokenizer.json` |
| **`capi/` (planned)** | **the product** — stable `extern "C"` ABI (`mlxforge.h`) wrapping `runtime/engine`; the public surface every binding links |
| **`bindings/` (planned)** | language bindings over the C ABI (`bindings/node` first; then Swift, Rust) |
| `server/openai` *(harness)* | OpenAI request parse + response serialize (pure) |
| `server/http_server` *(harness)* | routes, blocking + SSE handlers, error shapes |
| `server/config` *(harness)* | CLI/env server configuration |

For the full design see the [`doc/`](./doc) folder:

- [`doc/embedding.md`](./doc/embedding.md) — **start here**: the library / cross-language
  batched-engine thesis, the competitive landscape, and the C-ABI + Node quickstart.
- [`doc/architecture.md`](./doc/architecture.md) — engine architecture, the
  single-GPU-thread model, request lifecycle, continuous batching.
- [`doc/llm-architecture.md`](./doc/llm-architecture.md) — the transformer
  forward pass (embedding, RMSNorm, RoPE, GQA attention, SwiGLU, KV cache,
  sampling, quantization).
- [`doc/supported-models.md`](./doc/supported-models.md) — model families,
  adding a new one, the golden-reference discipline.
- [`doc/applications.md`](./doc/applications.md) — the server and CLI **harnesses**
  (how the engine is driven and validated) and the OpenAI API surface they expose.
- [`doc/distribution.md`](./doc/distribution.md) — release artifacts and how
  Homebrew, SwiftPM, npm, and the Rust crate consume them.
- [`doc/abi-stability.md`](./doc/abi-stability.md) — the C-ABI versioning policy
  and the automated symbol/leak guard (`scripts/check-abi.sh`).
- [`doc/conformance.md`](./doc/conformance.md) — validate a build against the
  `mlx-lm` golden reference (`scripts/conformance.sh`).
- [`doc/contributing.md`](./doc/contributing.md) — build/test workflow,
  conventions, and the hard-won numerical gotchas.

## License

mlxforge is released under the [MIT License](./LICENSE).

Model weights are **not** covered by this license and remain subject to their
own terms (e.g. the Llama Community License for the Llama-3.2 weights).
