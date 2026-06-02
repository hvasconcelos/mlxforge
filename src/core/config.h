#pragma once

#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include <nlohmann/json.hpp>

namespace mlxforge {

/// @brief Per-weight integer-quantization parameters (group size + bit-width).
/// A weight is quantized iff a "<base>.scales" tensor accompanies it; the
/// matching params come from the model's quantization config (MLX) or are
/// fixed by the source format (GGUF emits group_size 32). See Weights::quant.
struct QuantParams {
  int group_size = 64;
  int bits = 4;
};

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
  /// Quantization settings (present in quantized models, absent for fp16).
  /// `quantized` is informational only: the forward pass detects quantization
  /// per-weight (presence of "<base>.scales"), since a single checkpoint may
  /// mix quantized and dense tensors (GGUF) or vary bit-width per layer
  /// (mixed-precision MLX repos).
  bool quantized = false;       ///< Any weight is integer-quantized (logging/info).
  int quant_group_size = 64;    ///< Default group size for weight quantization.
  int quant_bits = 4;           ///< Default quantization bit-width (typically 4).
  /// Per-module quant overrides, keyed by weight base (the key without the
  /// trailing ".weight", e.g. "model.layers.0.mlp.down_proj"). Empty => every
  /// quantized weight uses the defaults above.
  std::unordered_map<std::string, QuantParams> quant_overrides;

  /// Resolve the quant params for a weight base: the per-module override if one
  /// exists, otherwise the model defaults.
  QuantParams quant_for(const std::string& base) const {
    auto it = quant_overrides.find(base);
    if (it != quant_overrides.end()) return it->second;
    return {quant_group_size, quant_bits};
  }

  // ----- RoPE scaling -----
  std::optional<RopeScaling> rope_scaling; ///< RoPE scaling config (if present in JSON).
  /// Precomputed per-dimension RoPE frequency factors (head_dim/2 values).
  /// GGUF checkpoints bake the llama3 rope rescaling into a `rope_freqs.weight`
  /// tensor instead of exposing the scaling params; when present, the final
  /// frequencies are `base**(2i/d) * rope_freq_factors[i]` and the param-based
  /// llama3 path is skipped. Absent for safetensors models.
  std::optional<std::vector<float>> rope_freq_factors;

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
