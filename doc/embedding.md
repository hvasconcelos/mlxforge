# Embedding mlxforge (`libmlxforge`)

This is the document to read first. It explains **what mlxforge is for**, why it
occupies a niche nothing else in the MLX ecosystem does, and how you will embed it.

> **Status.** Real today: the engine, the **C ABI** (`src/capi/mlxforge.h` →
> `libmlxforge`, gated by `tests/capi`), the **Node** and **Swift** bindings,
> **JSON-constrained structured output** (`json_schema`), and **embeddings**
> (`mlxforge_embed` / `engine.embed`). Every example below compiles and runs.

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

## The C ABI

One header, opaque handles, no C++ types across the boundary. Every fallible call
reports through a `char** err`; C++ exceptions are never allowed to cross `extern "C"`.
Strings are UTF-8 and freed by the caller via `mlxforge_string_free`. The ABI is
append-only and versioned (`mlxforge_abi_version()`).

Build it (the released library excludes the server/CLI harnesses, so the dylib carries
no `cpp-httplib`; pass `-DMLXFORGE_ENABLE_HF_DOWNLOAD=OFF` to also drop `libcurl`):

```sh
cmake -S . -B build -DMLXFORGE_BUILD_SERVER=OFF -DMLXFORGE_BUILD_CLI=OFF
cmake --build build --target mlxforge_shared      # -> build/libmlxforge.dylib + src/capi/mlxforge.h
```

```c
#include "mlxforge.h"

typedef struct mlxforge_engine  mlxforge_engine;
typedef struct mlxforge_request mlxforge_request;

// Create one engine; it owns the GPU worker thread and the batching scheduler.
mlxforge_engine* eng = mlxforge_engine_create("mlx-community/Llama-3.2-1B-Instruct-4bit",
                                              /*opts=*/NULL, &err);
// Or with extended options (ABI v6+): a quantized KV cache cuts the dominant
// growing allocation ~1.9x (8-bit, near-lossless) or ~3.6x (4-bit), and the
// prefix cache (v7+) skips re-prefilling shared prompt prefixes — ~20x lower
// warm TTFT on a 2k-token system prompt, same greedy tokens. kv_spill_dir adds
// an SSD tier that survives engine restarts.
//   mlxforge_engine_opts2 opts = { .struct_size = sizeof(opts), .kv_bits = 8,
//                                  .prefix_cache = 1 };
//   eng = mlxforge_engine_create2(spec, &opts, &err);
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

## Node binding (`bindings/node`)

`@mlxforge/node` — chat + streaming + batched concurrency, today. The surface mirrors
`node-llama-cpp` to make migration near-zero-diff, while adding what it can't do on MLX:
**batched concurrency**. (Source builds against the local dylib; a published package
would ship a prebuilt `.node` + dylib so `npm install` needs no compiler.)

```js
const { Engine } = require("@mlxforge/node");

const engine = await Engine.load("mlx-community/Llama-3.2-1B-Instruct-4bit");

// Streaming chat
for await (const chunk of engine.chat([{ role: "user", content: "Tell me a joke." }]))
  process.stdout.write(chunk);

// Concurrency is the point: these run on ONE batched engine, not queued one-by-one.
const answers = await Promise.all(
  questions.map((q) => engine.complete([{ role: "user", content: q }])),
);
```

## Swift package (`bindings/swift`)

`MLXForge` — the same engine for native macOS/iOS apps, `async`/`AsyncSequence`-based.
Apple's `MLXLLM` runs one stream; this serves many, batched, in one process.

```swift
import MLXForge

let engine = try await Engine.load("mlx-community/Llama-3.2-1B-Instruct-4bit")
for try await chunk in try engine.chat([.init(role: "user", content: "Tell me a joke.")]) {
  print(chunk, terminator: "")
}
async let a = engine.complete([.init(role: "user", content: "Name a color.")])
async let b = engine.complete([.init(role: "user", content: "Name a fruit.")])
print(try await a, try await b)   // one batched engine
```

## Structured output (constrained decoding)

Pass a `json_schema` sampling option and the engine masks each decode step so the
output can only be **well-formed JSON** — the guarantee holds regardless of model size,
because invalid tokens are removed before sampling. Two forms:

- `"json"` — any valid JSON value (OpenAI's `response_format: json_object`).
- a JSON-Schema string — supported subset: a **top-level object** with ordered,
  **required**, scalar-typed properties (`string` / `number` / `integer` / `boolean`).
  Keys are forced in schema order; `integer` forbids a decimal point. Nested-object and
  array property schemas fall back to a free JSON value. (Generation uses a compact
  mode — no inter-token whitespace — so greedy decoding cannot stall.)

```js
// Node: force the shape { city: string, population: integer }
const schema = JSON.stringify({
  type: "object",
  properties: { city: { type: "string" }, population: { type: "integer" } },
});
const out = await engine.complete([{ role: "user", content: "Tell me about Paris." }],
                                  { jsonSchema: schema, maxTokens: 128 });
JSON.parse(out); // always succeeds -> { city: "Paris", population: 2 }
```

```swift
var s = Sampling.greedy
s.jsonSchema = #"{"type":"object","properties":{"city":{"type":"string"}}}"#
let out = try await engine.complete([.init(role: "user", content: "Tell me about Paris.")],
                                    sampling: s)
```

The grammar engine is `src/sample/json_grammar.{h,cpp}` (pure, unit-tested in
`tests/sample/json_grammar_test.cpp`); the worker masks logits per row in
`runtime/worker`. It is opt-in and costs a per-step host-side vocab scan, so leave it
off for unconstrained generation.

## Embeddings

`engine.embed(text)` returns a unit-normalized vector. Rather than a separate BERT/
WordPiece stack, embeddings reuse the existing decoder and BPE tokenizer: the model runs
to its final hidden states (no LM head — `DecoderModel::forward_hidden`), which are
pooled over the sequence and L2-normalized. So *any* LLaMA/Qwen checkpoint yields an
embedding, and an embedding-tuned checkpoint yields a good one. Embedding requests flow
through the same scheduler as generation (a one-shot forward; no tokens streamed).

**Qwen3-Embedding works out of the box.** A bare `embed(text)` self-selects the model's
convention: the engine sniffs the sentence-transformers pooling sidecar
(`1_Pooling/config.json`) in the model dir, and a `pooling_mode_lasttoken` checkpoint —
i.e. Qwen3-Embedding — defaults to **last-token pooling** with a **trailing EOS**
(`<|endoftext|>`), exactly as the reference recipe requires. A plain LLM keeps mean
pooling and no EOS. The canonical `Qwen/Qwen3-Embedding-0.6B` repo stores its decoder at
the *root* (no `model.` prefix, no `lm_head`); the loader normalizes that backbone-root
layout to the engine's `model.*` form, so the official repo loads directly.

```js
// Documents: just embed them.
const doc = await engine.embed("The capital of China is Beijing.");
// Retrieval queries: pass the task instruction (Qwen3-Embedding convention,
// rendered as "Instruct: {instruction}\nQuery:{text}").
const q = await engine.embed("What is the capital of China?", {
  instruction: "Given a web search query, retrieve relevant passages that answer the query",
});
const cos = q.reduce((s, x, i) => s + x * doc[i], 0);        // high for a relevant doc
```

The second argument accepts an options object — `{ pooling, addEos, instruction, normalize }`
(or a legacy pooling number) — and the C ABI mirrors it with `mlxforge_embed_ex`
(`mlxforge_embed_opts`); `pooling`/`add_eos` left at `-1` defer to the detected default.
The Swift and Rust bindings expose the same options (`embed(_:pooling:addEos:instruction:normalize:)`,
`Engine::embed_with`).

```swift
let v = try await engine.embed("What is the capital of China?",
                               instruction: "Given a web search query, retrieve relevant passages")
```

**Harnesses.** The CLI embeds and prints a vector
(`mlxforge-cli embed <model> <text> [--last|--mean] [--eos] [--instruct "..."] [--no-normalize]`),
and the server exposes an OpenAI-compatible `POST /v1/embeddings` (string or array `input`,
returning the `{object:"list", data:[{embedding,...}], usage}` shape).

Gated by a golden fixture dumped from the real Qwen3-Embedding model
(`reference/dump_ref.py --model qwen3_embedding`): `tests/runtime/embedding_test.cpp`
asserts the pooled+normalized query/document vectors match the reference **and** that the
engine's tokenization (no BOS + appended EOS) is byte-identical — the anti-"silent
garbage" discipline applied to the embedding path, not just eyeballed. (`tests/capi` also
covers unit norm, determinism, and semantic ordering.) A cross-tool fixture vs
`sentence-transformers` fp32 remains a future hardening step.

## What is real today vs. design target

- **Real now:** the batched engine (`runtime/engine`, `scheduler/`, `cache/`), the
  tokenizer + chat templates, GGUF + safetensors loading, greedy / temperature / top-k /
  top-p sampling, JSON-constrained structured output, **embeddings**, tool/function-calling
  plumbing, the golden-reference gate, the C ABI (`src/capi/mlxforge.h` → `libmlxforge`),
  and the Node and Swift bindings. Every example above compiles and runs today.
- **Future hardening:** a *cross-tool* embedding fixture vs `sentence-transformers` fp32
  (a self-consistent mlx-lm golden already gates the Qwen3-Embedding pooled vector +
  tokenization); full JSON-Schema beyond the current object/scalar subset; prebuilt binary
  distribution (roadmap M4).

## See also

- [`architecture.md`](./architecture.md) — the single-GPU-thread model and request
  lifecycle the C ABI sits on top of.
- [`applications.md`](./applications.md) — the server and CLI harnesses.
- [`supported-models.md`](./supported-models.md) — model families and the
  golden-reference discipline every binding is validated against.
