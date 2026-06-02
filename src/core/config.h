#pragma once

#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace mlxforge {

/// @brief Specifies frequency scaling parameters for RoPE, as given
///        in config.json under "rope_scaling".
/// e.g., Llama-3.2 models use rope_type "llama3".
/// Used by the RoPE stage to adjust rotational frequencies and handle
/// position extrapolation for extended context.
struct RopeScaling {
  std::string rope_type;             ///< Type of RoPE scaling ("llama3", etc.)
  float factor = 1.0f;               ///< Primary scaling factor.
  float high_freq_factor = 1.0f;     ///< Scaling factor for high frequency components.
  float low_freq_factor = 1.0f;      ///< Scaling factor for low frequency components.
  int original_max_position_embeddings = 0; ///< Original context length before scaling.
};

/// @brief Represents the required architecture and config parameters
///        for a model, as loaded from config.json. Only fields needed
///        by the engine are exposed. If a required field is missing,
///        a clear error is thrown at parse time (see from_json()).
///
///        Unknown keys in the JSON are silently ignored.
///
struct ModelConfig {
  // ----- Core model hyperparameters -----
  std::string model_type;       ///< Model type ("llama", ...), determines chat template/format.
  int n_layers = 0;             ///< Number of transformer layers (num_hidden_layers).
  int hidden = 0;               ///< Hidden size of transformer blocks.
  int n_heads = 0;              ///< Number of attention heads.
  int n_kv_heads = 0;           ///< Number of key/value heads (for GQA).
  int head_dim = 0;             ///< Size of each attention head (defaults to hidden/n_heads if not set).
  int vocab = 0;                ///< Vocabulary size.
  int intermediate_size = 0;    ///< Width of the intermediate (SwiGLU) MLP.
  float rope_theta = 0.0f;      ///< RoPE base frequency.
  float rms_eps = 0.0f;         ///< Epsilon for RMSNorm.
  int max_position_embeddings = 0; ///< Max context length supported by positional embedding.
  bool tie_word_embeddings = true; ///< Whether to tie input/output word embeddings.

  // ----- Quantization parameters -----
  /// 4-bit quantization settings (present in quantized models, absent for fp16).
  bool quantized = false;       ///< Model uses integer quantized weights.
  int quant_group_size = 64;    ///< Group size for weight quantization (tokens per quant group).
  int quant_bits = 4;           ///< Quantization bit-width (typically 4).

  // ----- RoPE scaling -----
  std::optional<RopeScaling> rope_scaling; ///< RoPE scaling config (if present in JSON).

  // ----- Tokenization -----
  std::vector<int> eos_token_ids; ///< IDs for end-of-sequence tokens.
  int bos_token_id = -1;          ///< ID for beginning-of-sequence token.

  /// @brief Parse a ModelConfig from an already-loaded JSON object.
  ///        Throws std::runtime_error naming the first missing required field.
  /// @param j The parsed JSON object from config.json.
  /// @return ModelConfig populated from JSON.
  static ModelConfig from_json(const nlohmann::json& j);

  /// @brief Read and parse a config.json file.
  ///        Throws if the file cannot be opened, parsed, or if a required field is missing.
  /// @param path Filesystem path to the JSON config.
  /// @return ModelConfig instance.
  static ModelConfig from_file(const std::string& path);
};

}  // namespace mlxforge
