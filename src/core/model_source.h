#pragma once

// Turn a user-supplied model spec into a concrete local directory holding
// config.json / tokenizer.json / safetensors. A spec is either:
//   * a local directory (used as-is), or
//   * a HuggingFace repo id like "mlx-community/Llama-3.2-1B-Instruct-4bit".
//
// This is the single entry point the CLI and server call before loading a model,
// giving llama.cpp-style ergonomics: pass a folder or a repo id and it just
// works. Repo ids are resolved against the existing HuggingFace cache first
// (so models you already downloaded are reused) and only fetched over the
// network as a last resort.

#include <string>

namespace mlxforge {

// Resolve `spec` to a local model directory, downloading from the Hub if needed.
// Resolution order:
//   1. an existing directory containing config.json            -> use as-is
//   2. an existing HF-style parent dir (.../models--org--name) -> its snapshot
//   3. a repo id already in the HF hub cache                   -> that snapshot
//   4. a repo id already in the mlxforge cache                 -> that download
//   5. a repo id not cached anywhere                           -> download it
// Throws std::runtime_error if `spec` is neither a usable directory nor a
// well-formed repo id, or on download failure.
std::string resolve_model_dir(const std::string& spec);

// ---- helpers (exposed for unit testing) ------------------------------------

// True if `spec` looks like an HF repo id: not an existing path, exactly one
// '/', both halves non-empty, all chars in [A-Za-z0-9._-].
bool looks_like_repo_id(const std::string& spec);

// True if `dir` exists and contains config.json.
bool is_model_dir(const std::string& dir);

// Given the HF hub cache root and an "org/name" repo id, resolve the snapshot
// dir via <root>/models--org--name/refs/main (the commit hash) ->
// <root>/models--org--name/snapshots/<hash>. Returns "" if absent/unusable.
std::string hf_cache_snapshot_dir(const std::string& hub_cache_root, const std::string& repo_id);

// The HF hub cache root: $HF_HUB_CACHE, else $HF_HOME/hub, else
// ~/.cache/huggingface/hub.
std::string hf_hub_cache_root();

// mlxforge's own download cache root: $MLXFORGE_CACHE, else ~/.cache/mlxforge.
std::string mlxforge_cache_root();

}  // namespace mlxforge
