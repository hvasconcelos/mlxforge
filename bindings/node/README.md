# @mlxforge/node

Node.js bindings for **libmlxforge** — an embeddable, MLX-native, **continuously
batched** LLM engine for Apple Silicon. Unlike single-stream Node LLM libraries,
many `chat()` calls run concurrently on **one** in-process engine and share its
batching scheduler.

> Apple Silicon (arm64 macOS) only. This package binds the C ABI in
> [`src/capi/mlxforge.h`](../../src/capi/mlxforge.h); see
> [`doc/embedding.md`](../../doc/embedding.md) for the design.

## Build (from this repo)

The addon links the `libmlxforge.dylib` produced by the top-level CMake build:

```sh
# 1) build the library (from the repo root)
cmake -S . -B build && cmake --build build --target mlxforge_shared

# 2) build the Node addon
cd bindings/node
npm install          # installs node-addon-api and compiles src/addon.cc
npm test             # set MLXFORGE_MODEL_DIR to exercise a real model
```

A published package would ship a prebuilt `.node` + dylib so `npm install` needs
no toolchain; here the addon is built from source against the local `build/` dir.

## Usage

```js
const { Engine } = require('@mlxforge/node');

const engine = await Engine.load('mlx-community/Llama-3.2-1B-Instruct-4bit');

// Streaming
for await (const chunk of engine.chat([{ role: 'user', content: 'Tell me a joke.' }])) {
  process.stdout.write(chunk);
}

// Concurrency is the point: these share ONE batched engine, not a queue.
const answers = await Promise.all(
  questions.map((q) => engine.complete([{ role: 'user', content: q }])),
);

engine.dispose();
```

### Notes

- `chat(messages, sampling?)` / `text(prompt, sampling?)` return an async-iterable
  of text chunks; `complete(...)` collects to a string. Sampling options:
  `temperature`, `topK`, `topP`, `minP`, `repetitionPenalty`, `frequencyPenalty`,
  `presencePenalty`, `seed`, `maxTokens`.
- Concurrency uses the libuv thread pool (one blocking poll per in-flight
  request). For many simultaneous streams raise it, e.g. `UV_THREADPOOL_SIZE=16`.
- Keep the `Engine` alive while requests are streaming; `dispose()` after.

### Structured output

Pass `jsonSchema` to force valid JSON (the output is masked so it can only be
well-formed). `"json"` allows any JSON value; a JSON-Schema string constrains a
top-level object with ordered, required, scalar-typed properties:

```js
const schema = JSON.stringify({
  type: "object",
  properties: { city: { type: "string" }, population: { type: "integer" } },
});
const out = await engine.complete([{ role: "user", content: "Tell me about Paris." }],
                                  { jsonSchema: schema });
JSON.parse(out); // always valid
```

### Embeddings

```js
const a = await engine.embed("The cat sat on the mat."); // Float32Array, unit-normalized
const b = await engine.embed("A kitten rests on a rug.");
const cosine = a.reduce((s, x, i) => s + x * b[i], 0);   // ~0.88
```

Any LLaMA/Qwen model works (decoder hidden states, mean-pooled + normalized); an
embedding-tuned model produces a better vector.

## Status

Streaming, batched concurrency, JSON-constrained structured output, and
embeddings are implemented. Prebuilt binary distribution is on the roadmap.
