# Embedding mlxforge (`libmlxforge`)

This is the document to read first. It explains **what mlxforge is for**, why it
occupies a niche nothing else in the MLX ecosystem does, and how you will embed it.

> **Status.** The engine (scheduler, continuous batching, batched KV cache, tokenizer,
> GGUF, chat templates, sampling) exists and is golden-reference-gated today. The
> public **C ABI** (`src/capi/mlxforge.h`) and the language **bindings**
> (`bindings/node` first) are the active work — the API shown below is the **design
> target** that the roadmap is building toward, not yet a shipped interface. It is
> documented here so the shape is settled before the code lands.

## The product is the library

`libmlxforge` — the engine as an embeddable library, exposed through a stable
`extern "C"` ABI — **is the product**. The OpenAI-compatible HTTP server and the CLI in
this repo are **QA harnesses**: the server load-tests the scheduler and batching, the
CLI runs the golden-reference and weight smoke tests. They prove the engine is stable;
they are not the deliverable. See [`applications.md`](./applications.md) for what the
harnesses do.

Concretely, that means: the released library artifact builds with the harnesses **off**
(a lean dylib, no `cpp-httplib`, no `libcurl`), and the only header an embedder needs is
`mlxforge.h`.

## The niche: a batched MLX engine you can bind from any language

Apple Silicon is not short of ways to run a local model. But sort the options by what
they actually are, and a gap appears:

| Option | What it is | In-process | MLX-native | Batched / concurrent |
| --- | --- | :---: | :---: | :---: |
| `mlx-c`, `mlx-rs`, `mlx-rust`, `mlx-node` | array-framework bindings | ✅ | ✅ | you build the engine yourself |
| `node-llama-cpp` | Node LLM lib (llama.cpp / GGUF) | ✅ | ❌ | ⚠️ weak |
| `node-mlx` | Node LLM lib (MLX) | ✅ | ✅ | ❌ single-stream |
| Apple `MLXLLM` (mlx-swift) | Swift LLM lib | ✅ | ✅ | ❌ single-stream |
| `vllm-mlx`, `omlx`, `mlx-serve` | full MLX servers | ❌ (HTTP) | ✅ | ✅ |
| Ollama (`+ ollama` npm) | Go sidecar (MLX since 0.19) | ❌ (HTTP) | ✅ | ✅ |
| **mlxforge** | **batched engine + C ABI** | ✅ | ✅ | ✅ |

The array-framework bindings hand you tensor ops and leave the transformer, tokenizer,
KV cache, and batching to you. The single-language LLM libs (`node-mlx`, `MLXLLM`) are
**single-stream** — fine for one chat, but they cannot batch concurrent requests, which
is exactly what agent loops with parallel tool calls and multi-user backends need. The
batched servers (`vllm-mlx`, `omlx`, Ollama) **have** continuous batching but run as a
**separate process** you talk to over HTTP — you can't link them into your app, and the
MLX ones are Python.

mlxforge is the only option that is **in-process + MLX-native + batched** — and the only
batched MLX engine you bind from Node / Swift / Rust instead of running as a sidecar.

> **Why this matters.** `node-mlx`'s own headline ("2× node-llama-cpp on Apple
> Silicon") shows the demand for MLX inference outside Python is real, and that it is
> served today only by single-stream, single-language libraries. mlxforge brings the
> batched, scheduler-backed engine to that audience through one C ABI.

## Why a binding, not "just run the server"

The server already speaks OpenAI, so any app can hit it over HTTP. A *binding* earns its
place where that is not enough:

- **In-process, no sidecar** — one artifact, no port, no subprocess lifecycle to
  manage, lower latency, and the big model lives in your process's unified memory.
- **Desktop / Electron** — ship the engine *inside* the app; no bundled server binary to
  spawn and supervise. `npm install` pulls a prebuilt binary; there is no build step.
- **Tighter control** — direct streaming, cancellation, and backpressure without an HTTP
  hop.

If you only need a stateless backend on a box you control, the HTTP server may be all
you want. The binding's sweet spot is embedded and in-process use.

## The C ABI (design target)

One header, opaque handles, no C++ types across the boundary. Every fallible call
reports through a `char** err`; C++ exceptions are never allowed to cross `extern "C"`.
Strings are UTF-8 and freed by the caller via `mlxforge_string_free`. The ABI is
append-only and versioned (`mlxforge_abi_version()`).

```c
#include "mlxforge.h"

typedef struct mlxforge_engine  mlxforge_engine;
typedef struct mlxforge_request mlxforge_request;

// Create one engine; it owns the GPU worker thread and the batching scheduler.
mlxforge_engine* eng = mlxforge_engine_create("mlx-community/Llama-3.2-1B-Instruct-4bit",
                                              /*opts=*/NULL, &err);
while (!mlxforge_engine_ready(eng)) { /* model still loading on the worker thread */ }

// Submit many requests concurrently — they share one batched engine. This is the
// differentiator over the single-stream libs.
mlxforge_msg msgs[] = {{"user", "What is the capital of France?"}};
mlxforge_request* req = mlxforge_submit_chat(eng, msgs, 1, /*sampling=*/NULL, &err);

// Pull decoded UTF-8 text incrementally (the detokenizer lives behind the ABI).
char* piece = NULL;
while (mlxforge_request_next(req, &piece) == 0) {   // 0 = more, 1 = done
  fputs(piece, stdout);
  mlxforge_string_free(piece);
}
mlxforge_request_free(req);
mlxforge_engine_free(eng);
```

Threading: all MLX work stays on the engine's single worker thread (MLX arrays are
thread-bound); callers only ever touch their own `mlxforge_request`. The binding layer
follows the same rule — see [`architecture.md`](./architecture.md).

## Node binding (design target)

The first binding. Published as `@mlxforge/node` with **prebuilt arm64 binaries**, so
`npm install` needs no compiler. The surface mirrors `node-llama-cpp` to make migration
near-zero-diff, while adding what it can't do on MLX: **batched concurrency**.

```js
import { Engine } from "@mlxforge/node";

const engine = await Engine.load("mlx-community/Llama-3.2-1B-Instruct-4bit");

// Streaming chat
for await (const chunk of engine.chat([{ role: "user", content: "Tell me a joke." }]))
  process.stdout.write(chunk);

// Concurrency is the point: these run on ONE batched engine, not queued one-by-one.
const answers = await Promise.all(
  questions.map((q) => engine.complete([{ role: "user", content: q }])),
);
```

Planned surface, in order: chat + streaming + tool calling (engine has these today) →
JSON-schema / structured output (new constrained decoding in `sample/sampler`) →
embeddings (new encoder model family). Until structured output and embeddings land, they
are documented as gaps rather than implied — see the roadmap.

## What is real today vs. design target

- **Real now:** the batched engine (`runtime/engine`, `scheduler/`, `cache/`), the
  tokenizer + chat templates, GGUF + safetensors loading, greedy / temperature / top-k /
  top-p sampling, tool/function-calling plumbing, and the golden-reference gate. You can
  drive all of it through the server/CLI harnesses today.
- **Design target (in progress):** `src/capi/mlxforge.h`, the `bindings/node` package,
  JSON-schema-constrained decoding, and an embeddings path. The APIs above are the
  contract those will implement.

## See also

- [`architecture.md`](./architecture.md) — the single-GPU-thread model and request
  lifecycle the C ABI sits on top of.
- [`applications.md`](./applications.md) — the server and CLI harnesses.
- [`supported-models.md`](./supported-models.md) — model families and the
  golden-reference discipline every binding is validated against.
