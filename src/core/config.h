// MLXFORGE-003: ModelConfig — the model's architecture, parsed from config.json.
//
// Only the fields the engine actually consumes are exposed. Required fields
// (those without a sensible default) raise a clear error when missing; unknown
// keys in the JSON are ignored.
#pragma once

#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace mlxforge {

// Llama-3 RoPE frequency rescaling (config.json "rope_scaling"). Llama-3.2 sets
// rope_type "llama3"; honoured by the RoPE stage in MLXFORGE-006.
struct RopeScaling {
  std::string rope_type;
  float factor = 1.0f;
  float high_freq_factor = 1.0f;
  float low_freq_factor = 1.0f;
  int original_max_position_embeddings = 0;
};

struct ModelConfig {
  int n_layers = 0;            // num_hidden_layers
  int hidden = 0;             // hidden_size
  int n_heads = 0;            // num_attention_heads
  int n_kv_heads = 0;         // num_key_value_heads (GQA)
  int head_dim = 0;           // head_dim (defaults to hidden / n_heads)
  int vocab = 0;              // vocab_size
  int intermediate_size = 0;  // SwiGLU MLP width
  float rope_theta = 0.0f;    // rope base frequency
  float rms_eps = 0.0f;       // rms_norm_eps
  int max_position_embeddings = 0;
  bool tie_word_embeddings = true;
  // 4-bit quantization (config.json "quantization"), absent for fp16 models.
  bool quantized = false;
  int quant_group_size = 64;
  int quant_bits = 4;
  std::optional<RopeScaling> rope_scaling;
  std::vector<int> eos_token_ids;
  int bos_token_id = -1;

  // Parse from an already-loaded JSON object. Throws std::runtime_error naming
  // the first missing required field.
  static ModelConfig from_json(const nlohmann::json& j);

  // Read and parse a config.json file. Throws if the file cannot be opened or
  // parsed, or if a required field is missing.
  static ModelConfig from_file(const std::string& path);
};

}  // namespace mlxforge
