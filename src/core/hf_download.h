#pragma once

// Download a model's files from the HuggingFace Hub into a local directory,
// using system libcurl. This is the only place in the engine that speaks HTTP as
// a client; keep curl symbols confined to hf_download.cpp.
//
// The flow: list the repo's files via the HF model-info API, filter to the
// inference-relevant set (config + tokenizer + safetensors), then fetch each
// over HTTPS (following the cross-host redirect to the LFS CDN). Files land in a
// sibling ".incomplete-<pid>" directory that is renamed into place only once all
// downloads succeed, so a half-finished pull never looks like a usable model.

#include <string>
#include <vector>

namespace mlxforge {

// Download `repo_id` (e.g. "mlx-community/Llama-3.2-1B-Instruct-4bit") at
// `revision` into `dest_dir`, creating it. Returns `dest_dir`. Throws
// std::runtime_error on a network error, an HTTP error (with a targeted message
// for gated/missing repos), or if the repo has no loadable safetensors weights.
std::string hf_download_repo(const std::string& repo_id, const std::string& dest_dir,
                             const std::string& revision = "main");

// Parse an HF model-info JSON body (from /api/models/<repo>) into the list of
// `siblings[].rfilename` entries. Pure; no network. Exposed for unit testing.
std::vector<std::string> parse_repo_siblings(const std::string& model_info_json);

// From a repo's full file list, pick the files this engine needs to load a model
// (config/tokenizer JSON + *.safetensors), excluding PyTorch/GGUF/ONNX weights,
// READMEs, and images. Pure. Throws if the result lacks a config.json or any
// *.safetensors (a repo this engine cannot load). Exposed for unit testing.
std::vector<std::string> select_files_to_download(const std::vector<std::string>& siblings);

// Download a single self-contained `.gguf` file from `repo_id` (a GGUF repo
// typically ships many quant variants) at `revision` into `dest_dir`, picking the
// one whose name contains `tag` (case-insensitive, e.g. "Q4_0"). Returns the path
// to the downloaded `.gguf`. Reuses an already-present file without re-fetching.
// Throws on network/HTTP error, or if `tag` matches zero or more than one `.gguf`.
std::string hf_download_gguf(const std::string& repo_id, const std::string& dest_dir,
                             const std::string& tag, const std::string& revision = "main");

// From a repo's file list, pick the single `.gguf` whose name contains `tag`
// (case-insensitive substring). Pure. Throws — listing the candidates — if the
// repo has no `.gguf` files, or if `tag` matches zero or more than one. Exposed
// for unit testing.
std::string select_gguf_file(const std::vector<std::string>& siblings, const std::string& tag);

}  // namespace mlxforge
