// Load, sanitize and fp16-cast model weights from safetensors.
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

#include "core/config.h"
#include "mlx/array.h"

namespace mlxforge {

// Canonicalize a raw safetensors key. Returns the canonical key, or nullopt to
// drop the tensor (buffers we never use). The language tower is canonicalized to
// "model.*" across both wrapper layouts (Qwen3.5 "language_model.model.*" and
// Qwen3-VL "model.language_model.*"). The ViT tower is dropped unless
// `keep_vision` (VLM loads), where it canonicalizes to a leading "visual.*".
// Pure string logic — unit tested.
std::optional<std::string> sanitize_key(const std::string& raw, bool keep_vision = false);

// Parse a safetensors index JSON's "weight_map" (tensor key -> shard filename).
std::unordered_map<std::string, std::string> parse_shard_index(const nlohmann::json& index_json);

// The unique shard filenames referenced by a weight_map, sorted for determinism.
std::vector<std::string> shard_files(
    const std::unordered_map<std::string, std::string>& weight_map);

struct Weights {
  std::unordered_map<std::string, mlx::core::array> tensors;
  // Quant params for each quantized weight, keyed by the weight base (the key
  // without the trailing ".weight"). A weight is quantized iff "<base>.scales"
  // is present in `tensors`; this map records its group_size/bits. Dense
  // weights have no entry here.
  std::unordered_map<std::string, QuantParams> quant;

  bool has(const std::string& key) const { return tensors.count(key) > 0; }
  // Throws std::runtime_error naming the key if absent.
  const mlx::core::array& at(const std::string& key) const;
  size_t size() const { return tensors.size(); }

  // Whether `weight_key` (ending in ".weight") is an integer-quantized weight,
  // i.e. it has a sibling "<base>.scales" tensor. On true, fills `out` with the
  // weight's quant params (from `quant`, falling back to QuantParams defaults).
  bool is_quantized(const std::string& weight_key, QuantParams& out) const;

  // Human-readable "key  [shape]  dtype" dump, sorted by key.
  std::string summary() const;
};

// Load every weight tensor from a model directory, applying sanitize and casting
// to fp16. `cfg` supplies the quant params (defaults + per-module overrides) used
// to populate Weights::quant for each quantized tensor. Throws if neither an
// index JSON nor model.safetensors is found.
Weights load_weights(const std::string& model_dir, const ModelConfig& cfg);

}  // namespace mlxforge
