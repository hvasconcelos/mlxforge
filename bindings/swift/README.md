# MLXForge (Swift)

Swift bindings for **libmlxforge** — an embeddable, MLX-native, **continuously
batched** LLM engine. Positioned against Apple's `MLXLLM`: that runs one stream;
`MLXForge` serves many, batched, on one in-process engine.

> Apple Silicon / macOS 13+. Binds the C ABI in
> [`src/capi/mlxforge.h`](../../src/capi/mlxforge.h) via a Clang module; see
> [`doc/embedding.md`](../../doc/embedding.md).

## Build & test (from this repo)

The package links the `libmlxforge.dylib` from the top-level CMake build:

```sh
# 1) build the library (repo root)
cmake -S . -B build && cmake --build build --target mlxforge_shared

# 2) build / test / run the Swift package
cd bindings/swift
swift build
swift test                                   # ABI test always; model test if MLXFORGE_MODEL_DIR set
swift run mlxforge-swift-example mlx-community/Llama-3.2-1B-Instruct-4bit "Tell me a joke."
```

## Usage

```swift
import MLXForge

let engine = try await Engine.load("mlx-community/Llama-3.2-1B-Instruct-4bit")

// Streaming
for try await chunk in try engine.chat([.init(role: "user", content: "Tell me a joke.")]) {
  print(chunk, terminator: "")
}

// Concurrency: these share ONE batched engine.
async let a = engine.complete([.init(role: "user", content: "Name a color.")])
async let b = engine.complete([.init(role: "user", content: "Name a fruit.")])
print(try await a, try await b)
```

`chat`/`text` return `AsyncThrowingStream<String, Error>`; `complete` collects to
a `String`. `Sampling` exposes `temperature`, `topK`, `topP`, `minP`,
`repetitionPenalty`, `frequencyPenalty`, `presencePenalty`, `seed`, `maxTokens`.

## Distribution

`scripts/make_xcframework.sh` packages the dylib + header into
`MLXForge.xcframework`, which a published package would consume as a binary
target (so adopters need no CMake/MLX build). The framework is gitignored;
regenerate it from a fresh `mlxforge_shared` build.

## SwiftUI

The `Engine` API is `async`/`AsyncSequence`-based and drives cleanly from a
SwiftUI view model (await `chat(...)` and append chunks to `@Published` state). A
full sample app target is a follow-up; the `Example` executable shows the calls.
