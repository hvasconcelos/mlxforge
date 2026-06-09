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

/// @brief Vision-tower (ViT) hyperparameters for a multimodal checkpoint.
/// Present only for VLMs (e.g. Qwen3-VL), which nest these under "vision_config".
/// Text-only models leave ModelConfig::vision empty and never touch this. Field
/// names mirror config.json's vision_config so the parse stays a direct read.
struct VisionConfig {
  int depth = 0;                ///< Number of ViT transformer blocks.
  int hidden = 0;               ///< ViT hidden size.
  int intermediate_size = 0;    ///< ViT MLP width.
  int num_heads = 0;            ///< ViT attention heads (head_dim = hidden / num_heads).
  int in_channels = 3;          ///< Image input channels (RGB).
  int patch_size = 0;           ///< Spatial patch edge, in pixels.
  int temporal_patch_size = 1;  ///< Frames grouped into one temporal patch.
  int spatial_merge_size = 1;   ///< Patch-merger edge: spatial_merge_size^2 patches -> 1 token.
  int out_hidden_size = 0;      ///< Merger output dim (equals the LLM hidden size).
  int num_position_embeddings = 0; ///< Learned position-embedding table size (a square grid).
  /// ViT block indices whose hidden states are fed through the DeepStack mergers
  /// and added into the first decoder layers of the language model.
  std::vector<int> deepstack_visual_indexes;

  /// Patches merged into one LLM token. The placeholder-token count for an image
  /// is prod(grid_thw) / merge_unit().
  int merge_unit() const { return spatial_merge_size * spatial_merge_size; }
  /// Per-head ViT attention dimension.
  int head_dim() const { return num_heads > 0 ? hidden / num_heads : 0; }
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

  // ----- Mixture-of-Experts (MoE) parameters -----
  /// Sparse-MoE models (e.g. Qwen3-30B-A3B) replace the dense SwiGLU MLP, on the
  /// MoE layers, with a router that selects `num_experts_per_tok` of `num_experts`
  /// per token. `num_experts == 0` means a dense model (every field below unused),
  /// keeping the dense path byte-for-byte unchanged.
  int num_experts = 0;            ///< Total routed experts per MoE layer (0 => dense model).
  int num_experts_per_tok = 0;    ///< Experts selected per token (router top-k).
  int moe_intermediate_size = 0;  ///< Expert MLP width (defaults to intermediate_size).
  int decoder_sparse_step = 1;    ///< A layer is MoE iff (i+1) % decoder_sparse_step == 0 ...
  std::vector<int> mlp_only_layers; ///< ... and i is not in this dense-only list.
  bool norm_topk_prob = false;    ///< Renormalize the top-k routing scores to sum to 1.

  // ----- Hybrid linear-attention (Qwen3.5 / Qwen3-Next) parameters -----
  /// Qwen3.5 interleaves Gated-DeltaNet linear-attention layers with periodic
  /// full (softmax) attention. A layer is full-attention iff (i+1) is a multiple
  /// of `full_attention_interval`; the rest are linear. `full_attention_interval
  /// == 0` means a non-hybrid model (every field below unused), leaving the plain
  /// full-attention path unchanged. The full-attention layers add a sigmoid output
  /// gate (`attn_output_gate`) and use partial RoPE (`partial_rotary_factor`).
  int full_attention_interval = 0;   ///< (i+1) % interval == 0 => full attention (0 => not hybrid).
  bool attn_output_gate = false;     ///< Full-attn output gate (q_proj is 2x width: queries||gate).
  /// Gated-DeltaNet (linear-attention) dimensions. Mirror config.json's linear_*.
  int linear_num_key_heads = 0;      ///< Q/K heads in the linear-attention layers.
  int linear_num_value_heads = 0;    ///< V heads (>= key heads; GVA repeat = v/k).
  int linear_key_head_dim = 0;       ///< Per-head Q/K width.
  int linear_value_head_dim = 0;     ///< Per-head V width.
  int linear_conv_kernel_dim = 0;    ///< Causal depthwise conv kernel size over the q/k/v stream.

  /// Fraction of head_dim rotated by RoPE (1.0 => full rotary, the default for
  /// Llama/Qwen3). Qwen3.5 rotates only the leading `partial_rotary_factor *
  /// head_dim` dimensions; the tail is passed through un-rotated.
  float partial_rotary_factor = 1.0f;

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

  /// True if decoder layer `i` uses the sparse-MoE block rather than the dense
  /// SwiGLU MLP. Mirrors mlx_lm's Qwen3MoeDecoderLayer selection rule.
  bool is_moe_layer(int i) const {
    if (num_experts <= 0) return false;
    for (int only : mlp_only_layers)
      if (only == i) return false;
    return (i + 1) % decoder_sparse_step == 0;
  }

  /// True if decoder layer `i` is a Gated-DeltaNet linear-attention layer rather
  /// than a full (softmax) attention layer. Mirrors mlx_lm's Qwen3.5 DecoderLayer
  /// rule: every `full_attention_interval`-th layer is full attention, the rest
  /// linear. Non-hybrid models (interval 0) have no linear layers.
  bool is_linear_layer(int i) const {
    if (full_attention_interval <= 0) return false;
    return (i + 1) % full_attention_interval != 0;
  }

  // ----- RoPE scaling -----
  std::optional<RopeScaling> rope_scaling; ///< RoPE scaling config (if present in JSON).
  /// Precomputed per-dimension RoPE frequency factors (head_dim/2 values).
  /// GGUF checkpoints bake the llama3 rope rescaling into a `rope_freqs.weight`
  /// tensor instead of exposing the scaling params; when present, the final
  /// frequencies are `base**(2i/d) * rope_freq_factors[i]` and the param-based
  /// llama3 path is skipped. Absent for safetensors models.
  std::optional<std::vector<float>> rope_freq_factors;

  // ----- Multimodal vision (VLMs, e.g. Qwen3-VL) -----
  /// ViT config, present iff the checkpoint is a vision-language model. When set,
  /// the engine keeps + runs the vision tower; when empty the model is text-only
  /// and every field below is unused, leaving the text path unchanged.
  std::optional<VisionConfig> vision;
  bool has_vision_tower() const { return vision.has_value(); }
  int image_token_id = -1;         ///< Placeholder expanded per merged image patch (<|image_pad|>).
  int video_token_id = -1;         ///< Placeholder expanded per merged video patch (<|video_pad|>).
  int vision_start_token_id = -1;  ///< Marks the start of a vision span (<|vision_start|>).
  int vision_end_token_id = -1;    ///< Marks the end of a vision span (<|vision_end|>).

  // ----- Interleaved multimodal RoPE (M-RoPE) -----
  /// Qwen3-VL gives image tokens 3D (temporal, height, width) rotary positions
  /// with an *interleaved* frequency layout. `mrope_section` (entries summing to
  /// head_dim/2) allocates frequency bands to the t/h/w axes. Empty => plain 1D
  /// RoPE (text-only models). For text tokens t==h==w, so interleaved M-RoPE
  /// reduces exactly to standard 1D RoPE at that position.
  std::vector<int> mrope_section;
  bool mrope_interleaved = false;

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
