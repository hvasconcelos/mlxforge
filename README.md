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
> repo are **QA harnesses** that exercise and prove the engine's stability — they are
> dev/QA tools, not the deliverable. See [`doc/applications.md`](./doc/applications.md).

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

## Features

- **Embeddable batched engine** — `runtime/engine` is an HTTP-free, in-process engine
  (one GPU worker + scheduler + batched KV cache) wrapped by a stable `extern "C"` ABI
  (`src/capi/mlxforge.h`, *in progress*) and bound from other languages. This is the
  product.
- **Numerically correct** — forward-pass logits and greedy tokens match `mlx-lm`
  (golden-reference `.npy` fixtures gate every step).
- **Continuous batching** — one GPU worker thread owns all MLX state and is the only
  caller of `eval`/`async_eval`; exactly **one `async_eval` per decode step** over the
  whole batch, with batch-size bucketing.
- **KV cache** — single-sequence and batched (`BatchKVCache`), left-padded, grown in
  256-token blocks, with `filter` (eviction) / `merge` (admission). Optional
  **KV-cache quantization** (`--kv-bits 8|4`, default fp16): mlx-lm-matching quantized
  storage + attention for ~1.9×/~3.6× less cache memory — including the active
  continuous-decode batch, which no other MLX server quantizes.
- **Sampling as graph ops** — greedy, temperature, top-k, top-p (no host readback).
- **Embeddings** — `engine.embed` runs the decoder to its final hidden states, pools
  (mean or last-token) and L2-normalizes. **Qwen3-Embedding** is first-class. Exposed
  through the C ABI, the bindings, the CLI `embed` command, and `POST /v1/embeddings`.
- **C++ tokenizer** — from-scratch byte-level BPE and SentencePiece-BPE over HF
  `tokenizer.json` (no Rust), with chat templates and UTF-8-safe streaming detokenization.
- **OpenAI server harness** — drives the engine over HTTP (`/v1/chat/completions`,
  `/v1/completions`, `/v1/embeddings`, `/v1/models`, `/health`), streaming + tool calling.
  Built only when `MLXFORGE_BUILD_SERVER=ON`; the released library ships without it.
- **Optional 4-bit quantization** — `quantized_matmul` (group_size 64), ~0.65 GiB
  resident vs ~2.3 GiB fp16.
- **Configurable logging** (spdlog) — level / file / format via `MLXFORGE_LOG_*`.

## Supported models

The forward pass (the `DecoderModel` base in `model/`) is family-shared and runs
Llama-3.2, Qwen3 (dense / MoE), and Qwen3.5 hybrid models from safetensors (fp16 /
4-bit) or a single-file GGUF, plus **Qwen3-VL** vision-language (image → text). The
chat template and special-token handling are selected from `config.json` — loading
is org-agnostic.

| Family | Example repo | Precision |
| --- | --- | --- |
| Llama-3.2 | `mlx-community/Llama-3.2-1B-Instruct-bf16` / `-4bit` | fp16 / 4-bit |
| Llama-3.2 (GGUF) | `bartowski/Llama-3.2-1B-Instruct-GGUF` | Q4_0/Q4_1/Q8_0, Q4_K/Q5_K/Q6_K |
| Qwen3 (dense) | `mlx-community/Qwen3-0.6B-bf16` / `Qwen3-4B-4bit` | fp16 / 4-bit |
| Qwen3 (GGUF) | `Qwen/Qwen3-0.6B-GGUF` | Q4_0/Q4_1/Q8_0, Q4_K/Q5_K/Q6_K |
| Qwen3 (MoE) | `mlx-community/Qwen3-30B-A3B-4bit` | 4-bit / fp16 |
| Qwen3.5 (hybrid) | `mlx-community/Qwen3.5-0.8B-4bit` | 4-bit (text tower) |
| Qwen3-VL (vision) | `mlx-community/Qwen3-VL-4B-Instruct-4bit` | 4-bit (image → text) |

See [`doc/supported-models.md`](./doc/supported-models.md) for the full compatibility
matrix, the per-family deltas, and how to add a new family.

## Requirements

- Apple Silicon (the MLX Metal backend) + the Xcode **Metal Toolchain**
  (`xcodebuild -downloadComponent MetalToolchain`).
- CMake ≥ 3.24, a C++17 compiler (Apple clang).
- System **libcurl** (in the macOS SDK; found via `find_package(CURL)`) — for
  HuggingFace downloads.
- (Optional, for regenerating golden fixtures) Python 3.12 + `mlx-lm`.

All C++ dependencies (MLX, cpp-httplib, nlohmann/json, doctest, spdlog) are fetched and
pinned by CMake (`cmake/Dependencies.cmake`). The tokenizer is our own, so there is no
Rust/`cargo` requirement.

## Build

```sh
cmake -S . -B build
cmake --build build --parallel
```

The first build compiles MLX's Metal kernels, so it takes a few minutes. Outputs:

- `build/libmlxforge.dylib` — **the product**: the engine behind the C ABI
  (`src/capi/mlxforge.h`); link this to embed mlxforge (see
  [`doc/embedding.md`](./doc/embedding.md)).
- `build/mlxforge` — the OpenAI HTTP server *harness*.
- `build/mlxforge-cli` — CLI *harness* (weight dump + single-stream generation).
- `build/tests/mlxforge_tests` — the doctest suite.

The harnesses are on by default for development; the released library is built with them
off (a lean dylib, no httplib/curl). For the build-option matrix see
[`doc/applications.md`](./doc/applications.md).

## Quickstart

The server and CLI accept a model **spec** — a HuggingFace repo id (downloaded and
cached on first use) or a local model directory — passed directly, llama.cpp-style:

```sh
# CLI: stream generated text from a chat prompt (downloads the model if needed)
./build/mlxforge-cli generate mlx-community/Llama-3.2-1B-Instruct-4bit \
  "What is the capital of France?" 64

# Server: speak the OpenAI API so existing clients can hammer the scheduler
./build/mlxforge -m mlx-community/Llama-3.2-1B-Instruct-4bit --port 8080
```

```python
from openai import OpenAI
c = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="x")
r = c.chat.completions.create(model="mlxforge",
    messages=[{"role": "user", "content": "What is the capital of France?"}],
    max_tokens=32)
print(r.choices[0].message.content)            # "The capital of France is Paris."
```

How specs are resolved (HF cache reuse, download cache, gated repos) is documented in
[`doc/supported-models.md`](./doc/supported-models.md); the harness flags, endpoints,
streaming, tool calling, and `embed` are in [`doc/applications.md`](./doc/applications.md).

## Documentation

The [`doc/`](./doc) folder is the design reference:

- [`doc/embedding.md`](./doc/embedding.md) — **start here**: the library / cross-language
  batched-engine thesis, the competitive landscape, and the C-ABI + Node quickstart.
- [`doc/architecture.md`](./doc/architecture.md) — engine architecture, the
  single-GPU-thread model, request lifecycle, continuous batching, and the module map.
- [`doc/llm-architecture.md`](./doc/llm-architecture.md) — the transformer forward pass
  (embedding, RMSNorm, RoPE, GQA attention, SwiGLU, KV cache, sampling, quantization).
- [`doc/supported-models.md`](./doc/supported-models.md) — model families, getting the
  weights, spec resolution, and adding a new family.
- [`doc/tokenizer.md`](./doc/tokenizer.md) — the from-scratch byte-level BPE / SPM
  tokenizers and their byte-exact validation against mlx-lm.
- [`doc/applications.md`](./doc/applications.md) — the server and CLI **harnesses**,
  the build options, the OpenAI API surface, and logging.
- [`doc/distribution.md`](./doc/distribution.md) — release artifacts and how Homebrew,
  SwiftPM, npm, and the Rust crate consume them.
- [`doc/abi-stability.md`](./doc/abi-stability.md) — the C-ABI versioning policy and the
  automated symbol/leak guard.
- [`doc/conformance.md`](./doc/conformance.md) — validate a build against the `mlx-lm`
  golden reference.
- [`doc/contributing.md`](./doc/contributing.md) — build/test workflow, conventions, the
  golden-reference discipline, and the hard-won numerical gotchas.

## License

mlxforge is released under the [MIT License](./LICENSE).

Model weights are **not** covered by this license and remain subject to their own terms
(e.g. the Llama Community License for the Llama-3.2 weights).
