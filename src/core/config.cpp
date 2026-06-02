#include "core/config.h"

#include <fstream>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace mlxforge {

namespace {

// Helper function: fetch a required field from a JSON object.
// Throws a clear runtime_error with the missing key name if absent,
// or with specific type info if the type is incorrect.
template <typename T>
T require(const nlohmann::json& j, const char* key) {
  auto it = j.find(key);
  if (it == j.end()) {
    throw std::runtime_error(std::string("config.json: missing required field '") + key + "'");
  }
  try {
    return it->get<T>();
  } catch (const nlohmann::json::exception& e) {
    throw std::runtime_error(
        std::string("config.json: field '") + key + "' has wrong type: " + e.what());
  }
}

// eos_token_id in HuggingFace config can be an int or an array of ints.
// Normalize to a vector of ints for internal use.
// Returns empty vector if not present.
std::vector<int> parse_eos_ids(const nlohmann::json& j) {
  auto it = j.find("eos_token_id");
  if (it == j.end()) return {};
  if (it->is_array()) return it->get<std::vector<int>>();
  return {it->get<int>()};
}

// Attempt to parse the optional "rope_scaling" sub-object, if present and is an object.
// Returns std::nullopt if absent or of incorrect type.
// Otherwise, fills out RopeScaling struct with available fields.
std::optional<RopeScaling> parse_rope_scaling(const nlohmann::json& j) {
  auto it = j.find("rope_scaling");
  if (it == j.end() || !it->is_object()) return std::nullopt;
  RopeScaling rs;
  rs.rope_type = it->value("rope_type", std::string{});
  rs.factor = it->value("factor", 1.0f);
  rs.high_freq_factor = it->value("high_freq_factor", 1.0f);
  rs.low_freq_factor = it->value("low_freq_factor", 1.0f);
  rs.original_max_position_embeddings = it->value("original_max_position_embeddings", 0);
  return rs;
}

}  // namespace

// Constructs and returns a ModelConfig from a nlohmann::json config (typically read from HuggingFace .json).
// Throws if any required fields are missing or wrong type.
// Uses sensible defaults for certain optional fields.
ModelConfig ModelConfig::from_json(const nlohmann::json& j) {
  ModelConfig c;
  // model_type is optional; empty string if not present.
  c.model_type = j.value("model_type", std::string{});

  // These are required for all transformer configs.
  c.n_layers = require<int>(j, "num_hidden_layers");
  c.hidden = require<int>(j, "hidden_size");
  c.n_heads = require<int>(j, "num_attention_heads");
  c.n_kv_heads = require<int>(j, "num_key_value_heads");
  c.vocab = require<int>(j, "vocab_size");
  c.intermediate_size = require<int>(j, "intermediate_size");
  c.rope_theta = require<float>(j, "rope_theta");
  c.rms_eps = require<float>(j, "rms_norm_eps");

  // head_dim is sometimes omitted; default to hidden/n_heads if so.
  c.head_dim = j.value("head_dim", c.hidden / c.n_heads);
  // Optional fields; fallback to reasonable defaults if not present.
  c.max_position_embeddings = j.value("max_position_embeddings", 0);
  c.tie_word_embeddings = j.value("tie_word_embeddings", true);
  c.bos_token_id = j.value("bos_token_id", -1);

  // Parse eos_token_id (as int or vector<int>) and rope_scaling (optional sub-object).
  c.eos_token_ids = parse_eos_ids(j);
  c.rope_scaling = parse_rope_scaling(j);

  // If a quantization sub-object exists, extract its config fields as well.
  // The block carries top-level defaults (group_size/bits) plus, for
  // mixed-precision checkpoints, per-module overrides keyed by the module path
  // (the weight base, e.g. "model.layers.0.mlp.down_proj"). A per-module value
  // is an object with its own bits/group_size; a bare `false` means the module
  // is left dense (no override needed — it simply has no ".scales" tensor).
  if (auto it = j.find("quantization"); it != j.end() && it->is_object()) {
    c.quantized = true;
    c.quant_group_size = it->value("group_size", c.quant_group_size);
    c.quant_bits = it->value("bits", c.quant_bits);
    for (const auto& [key, val] : it->items()) {
      if (key == "group_size" || key == "bits") continue;  // top-level defaults
      if (val.is_object() && val.contains("bits")) {
        QuantParams qp;
        qp.group_size = val.value("group_size", c.quant_group_size);
        qp.bits = val.value("bits", c.quant_bits);
        c.quant_overrides.emplace(key, qp);
      }
    }
  }
  return c;
}

// Loads a ModelConfig from a JSON file at the specified path.
// Throws with clear errors if the file is missing or invalid.
ModelConfig ModelConfig::from_file(const std::string& path) {
  std::ifstream f(path);
  if (!f) {
    throw std::runtime_error("config.json: cannot open file '" + path + "'");
  }
  nlohmann::json j;
  try {
    f >> j;
  } catch (const nlohmann::json::exception& e) {
    throw std::runtime_error("config.json: parse error in '" + path + "': " + e.what());
  }
  return from_json(j);
}

}  // namespace mlxforge
