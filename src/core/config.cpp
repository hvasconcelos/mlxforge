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

// RoPE base frequency. Most configs expose a top-level "rope_theta"; Qwen3.5 nests
// it (with partial_rotary_factor) under a "rope_parameters" sub-object instead.
// Prefer the sub-object when present, else fall back to the required top-level key.
float parse_rope_theta(const nlohmann::json& j) {
  if (auto rp = j.find("rope_parameters"); rp != j.end() && rp->is_object()) {
    if (rp->contains("rope_theta")) return rp->at("rope_theta").get<float>();
  }
  return require<float>(j, "rope_theta");
}

// Parse the optional ViT sub-object of a multimodal checkpoint. Returns
// std::nullopt for text-only models (no "vision_config" key). Mirrors Qwen3-VL's
// vision_config field names; missing optional fields fall back to defaults.
std::optional<VisionConfig> parse_vision_config(const nlohmann::json& j_top) {
  auto it = j_top.find("vision_config");
  if (it == j_top.end() || !it->is_object()) return std::nullopt;
  const nlohmann::json& v = *it;
  VisionConfig vc;
  vc.depth = v.value("depth", 0);
  vc.hidden = v.value("hidden_size", 0);
  vc.intermediate_size = v.value("intermediate_size", 0);
  vc.num_heads = v.value("num_heads", 0);
  vc.in_channels = v.value("in_channels", 3);
  vc.patch_size = v.value("patch_size", 0);
  vc.temporal_patch_size = v.value("temporal_patch_size", 1);
  vc.spatial_merge_size = v.value("spatial_merge_size", 1);
  vc.out_hidden_size = v.value("out_hidden_size", 0);
  vc.num_position_embeddings = v.value("num_position_embeddings", 0);
  if (auto di = v.find("deepstack_visual_indexes"); di != v.end() && di->is_array()) {
    vc.deepstack_visual_indexes = di->get<std::vector<int>>();
  }
  return vc;
}

// Interleaved M-RoPE lives under either "rope_scaling" (Qwen3-VL) or
// "rope_parameters" (Qwen3.5); both carry mrope_section/mrope_interleaved. Fill
// the out-params from whichever sub-object provides them; leave them at their
// text-only defaults (empty section, not interleaved) otherwise.
void parse_mrope(const nlohmann::json& j, std::vector<int>& section, bool& interleaved) {
  for (const char* key : {"rope_scaling", "rope_parameters"}) {
    auto it = j.find(key);
    if (it == j.end() || !it->is_object()) continue;
    if (auto ms = it->find("mrope_section"); ms != it->end() && ms->is_array()) {
      section = ms->get<std::vector<int>>();
    }
    interleaved = it->value("mrope_interleaved", interleaved);
  }
}

}  // namespace

// Constructs and returns a ModelConfig from a nlohmann::json config (typically read from HuggingFace .json).
// Throws if any required fields are missing or wrong type.
// Uses sensible defaults for certain optional fields.
ModelConfig ModelConfig::from_json(const nlohmann::json& j_top) {
  ModelConfig c;
  // Multimodal wrappers (Qwen3.5 "Qwen3_5ForConditionalGeneration") nest the
  // text decoder's hyperparameters under "text_config"; the architecture id and
  // quantization block stay at the top level. Read text hyperparameters from the
  // nested object when present, everything else from the top level.
  const nlohmann::json& j = j_top.contains("text_config") && j_top.at("text_config").is_object()
                                ? j_top.at("text_config")
                                : j_top;

  // model_type is optional; empty string if not present. Prefer the top-level id
  // (e.g. "qwen3_5") over the nested "*_text" variant — it selects the chat format.
  c.model_type = j_top.value("model_type", j.value("model_type", std::string{}));

  // These are required for all transformer configs.
  c.n_layers = require<int>(j, "num_hidden_layers");
  c.hidden = require<int>(j, "hidden_size");
  c.n_heads = require<int>(j, "num_attention_heads");
  c.n_kv_heads = require<int>(j, "num_key_value_heads");
  c.vocab = require<int>(j, "vocab_size");
  c.intermediate_size = require<int>(j, "intermediate_size");
  c.rope_theta = parse_rope_theta(j);  // top-level "rope_theta" or nested rope_parameters
  c.rms_eps = require<float>(j, "rms_norm_eps");

  // head_dim is sometimes omitted; default to hidden/n_heads if so.
  c.head_dim = j.value("head_dim", c.hidden / c.n_heads);
  // Optional fields; fallback to reasonable defaults if not present.
  c.max_position_embeddings = j.value("max_position_embeddings", 0);
  c.tie_word_embeddings = j.value("tie_word_embeddings", true);
  c.bos_token_id = j_top.value("bos_token_id", -1);

  // Parse eos_token_id (as int or vector<int>) and rope_scaling (optional sub-object).
  c.eos_token_ids = parse_eos_ids(j);
  c.rope_scaling = parse_rope_scaling(j);

  // Hybrid linear-attention (Qwen3.5 / Qwen3-Next). Absent in non-hybrid configs:
  // full_attention_interval stays 0, leaving every layer full attention and the
  // linear_* fields unused. partial_rotary_factor defaults to 1.0 (full rotary).
  c.full_attention_interval = j.value("full_attention_interval", 0);
  c.attn_output_gate = j.value("attn_output_gate", false);
  c.linear_num_key_heads = j.value("linear_num_key_heads", 0);
  c.linear_num_value_heads = j.value("linear_num_value_heads", 0);
  c.linear_key_head_dim = j.value("linear_key_head_dim", 0);
  c.linear_value_head_dim = j.value("linear_value_head_dim", 0);
  c.linear_conv_kernel_dim = j.value("linear_conv_kernel_dim", 0);
  if (auto rp = j.find("rope_parameters"); rp != j.end() && rp->is_object()) {
    c.partial_rotary_factor = rp->value("partial_rotary_factor", 1.0f);
  }

  // MoE fields are optional: absent in dense configs (num_experts stays 0, which
  // leaves the dense SwiGLU path unchanged). moe_intermediate_size defaults to the
  // dense intermediate_size; decoder_sparse_step defaults to 1 (every layer MoE).
  c.num_experts = j.value("num_experts", 0);
  c.num_experts_per_tok = j.value("num_experts_per_tok", 0);
  c.moe_intermediate_size = j.value("moe_intermediate_size", c.intermediate_size);
  c.decoder_sparse_step = j.value("decoder_sparse_step", 1);
  c.norm_topk_prob = j.value("norm_topk_prob", false);
  if (auto it = j.find("mlp_only_layers"); it != j.end() && it->is_array()) {
    c.mlp_only_layers = it->get<std::vector<int>>();
  }

  // If a quantization sub-object exists, extract its config fields as well.
  // The block carries top-level defaults (group_size/bits) plus, for
  // mixed-precision checkpoints, per-module overrides keyed by the module path
  // (the weight base, e.g. "model.layers.0.mlp.down_proj"). A per-module value
  // is an object with its own bits/group_size; a bare `false` means the module
  // is left dense (no override needed — it simply has no ".scales" tensor).
  if (auto it = j_top.find("quantization"); it != j_top.end() && it->is_object()) {
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
  // Multimodal vision tower (VLMs, e.g. Qwen3-VL). The ViT config and the vision
  // special-token ids live at the top level; M-RoPE parameters nest with the text
  // rope config (read from `j`). All absent for text-only models, which then keep
  // every vision/M-RoPE field at its inert default.
  c.vision = parse_vision_config(j_top);
  c.image_token_id = j_top.value("image_token_id", -1);
  c.video_token_id = j_top.value("video_token_id", -1);
  c.vision_start_token_id = j_top.value("vision_start_token_id", -1);
  c.vision_end_token_id = j_top.value("vision_end_token_id", -1);
  parse_mrope(j, c.mrope_section, c.mrope_interleaved);

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
