// XLLM-004: load, sanitize and fp16-cast model weights from safetensors.
//
// Handles both single-file (model.safetensors) and sharded
// (model-0000N-of-*.safetensors + index JSON) layouts. Every tensor is cast to
// fp16 on load (unified memory: no host/device copy). Key names are run through
// sanitize_key, which canonicalizes a few known aliases and drops non-weight
// buffers (e.g. rotary inv_freq).
#pragma once

#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include <nlohmann/json.hpp>

#include "mlx/array.h"

namespace xllm {

// Canonicalize a raw safetensors key. Returns the canonical key, or nullopt to
// drop the tensor (buffers we never use). Pure string logic — unit tested.
std::optional<std::string> sanitize_key(const std::string& raw);

// Parse a safetensors index JSON's "weight_map" (tensor key -> shard filename).
std::unordered_map<std::string, std::string> parse_shard_index(const nlohmann::json& index_json);

// The unique shard filenames referenced by a weight_map, sorted for determinism.
std::vector<std::string> shard_files(
    const std::unordered_map<std::string, std::string>& weight_map);

struct Weights {
  std::unordered_map<std::string, mlx::core::array> tensors;

  bool has(const std::string& key) const { return tensors.count(key) > 0; }
  // Throws std::runtime_error naming the key if absent.
  const mlx::core::array& at(const std::string& key) const;
  size_t size() const { return tensors.size(); }

  // Human-readable "key  [shape]  dtype" dump, sorted by key.
  std::string summary() const;
};

// Load every weight tensor from a model directory, applying sanitize and casting
// to fp16. Throws if neither an index JSON nor model.safetensors is found.
Weights load_weights(const std::string& model_dir);

}  // namespace xllm
