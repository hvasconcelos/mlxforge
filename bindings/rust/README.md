# mlxforge (Rust)

Rust bindings for **libmlxforge** — an embeddable, MLX-native, continuously
batched LLM engine for Apple Silicon. A thin, safe wrapper over the same C ABI
(`src/capi/mlxforge.h`) that backs the Node and Swift bindings, proving the
engine is language-neutral.

> Apple Silicon / macOS. Links the `libmlxforge.dylib` from the repo's CMake build.

## Build & test (from this repo)

```sh
# 1) build the library (repo root)
cmake -S . -B build && cmake --build build --target mlxforge_shared

# 2) build / test / run the crate
cd bindings/rust
cargo build
cargo test                                  # ABI test always; model tests if MLXFORGE_MODEL_DIR set
cargo run --example chat -- mlx-community/Llama-3.2-1B-Instruct-4bit "Tell me a joke."
```

## Usage

```rust
use mlxforge::{Engine, Sampling};

let engine = Engine::load("mlx-community/Llama-3.2-1B-Instruct-4bit")?;

// Chat (collected to a String)
let reply = engine.chat(&[("user", "Tell me a joke.")], &Sampling::greedy())?;

// Constrained JSON
let mut s = Sampling::greedy();
s.json_schema = Some("json".into());
let json = engine.chat(&[("user", "Describe a city.")], &s)?;

// Embeddings (unit-normalized)
let v = engine.embed("The cat sat on the mat.", 0)?;
```

`Engine` is `Send + Sync`, so multiple threads can drive one engine concurrently;
requests share its continuous-batching scheduler.

## Status

Reference binding: chat/text completion (collected), constrained JSON, and
embeddings. Token-streaming and a prebuilt-dylib `build.rs` (rather than the local
CMake build) are follow-ups.
