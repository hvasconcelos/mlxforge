#include "core/config.h"

#include <fstream>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace xllm {

namespace {

// Fetch a required field, throwing a clear error (with the key name) if absent.
template <typename T>
T require(const nlohmann::json& j, const char* key) {
  auto it = j.find(key);
  if (it == j.end()) {
    throw std::runtime_error(std::string("config.json: missing required field '") + key + "'");
  }
  try {
    return it->get<T>();
  } catch (const nlohmann::json::exception& e) {
    throw std::runtime_error(std::string("config.json: field '") + key + "' has wrong type: " +
                             e.what());
  }
}

// eos_token_id may be a single int or a list of ints; normalize to a vector.
std::vector<int> parse_eos_ids(const nlohmann::json& j) {
  auto it = j.find("eos_token_id");
  if (it == j.end()) return {};
  if (it->is_array()) return it->get<std::vector<int>>();
  return {it->get<int>()};
}

// rope_scaling is an optional sub-object; absent or non-object yields no value.
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

ModelConfig ModelConfig::from_json(const nlohmann::json& j) {
  ModelConfig c;
  c.n_layers = require<int>(j, "num_hidden_layers");
  c.hidden = require<int>(j, "hidden_size");
  c.n_heads = require<int>(j, "num_attention_heads");
  c.n_kv_heads = require<int>(j, "num_key_value_heads");
  c.vocab = require<int>(j, "vocab_size");
  c.intermediate_size = require<int>(j, "intermediate_size");
  c.rope_theta = require<float>(j, "rope_theta");
  c.rms_eps = require<float>(j, "rms_norm_eps");

  // head_dim is optional: default to hidden / n_heads when absent.
  c.head_dim = j.value("head_dim", c.hidden / c.n_heads);
  c.max_position_embeddings = j.value("max_position_embeddings", 0);
  c.tie_word_embeddings = j.value("tie_word_embeddings", true);
  c.bos_token_id = j.value("bos_token_id", -1);
  c.eos_token_ids = parse_eos_ids(j);
  c.rope_scaling = parse_rope_scaling(j);
  return c;
}

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

}  // namespace xllm
