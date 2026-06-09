#include "core/weights.h"

#include <algorithm>
#include <fstream>
#include <set>
#include <sstream>
#include <stdexcept>

#include "core/logging.h"
#include "mlx/io.h"
#include "mlx/ops.h"

namespace mx = mlx::core;

namespace mlxforge {

namespace {
constexpr const char* kLanguageModelPrefix = "language_model.";

bool ends_with(const std::string& s, const std::string& suffix) {
  return s.size() >= suffix.size() && s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

const char* dtype_name(mx::Dtype dt) {
  if (dt == mx::float16) return "float16";
  if (dt == mx::bfloat16) return "bfloat16";
  if (dt == mx::float32) return "float32";
  if (dt == mx::int32) return "int32";
  if (dt == mx::uint32) return "uint32";
  return "other";
}
}  // namespace

std::optional<std::string> sanitize_key(const std::string& raw) {
  // Rotary position buffers are recomputed, never loaded as weights.
  if (ends_with(raw, ".inv_freq") || raw.find("rotary_emb.inv_freq") != std::string::npos) {
    return std::nullopt;
  }
  // Drop the vision tower of a multimodal checkpoint (the engine is text-only):
  // Qwen-VL / Qwen3.5 ship a ViT under "vision_tower.*" (and some wrappers under
  // "(model.)visual.*") that the language model never reads.
  for (const char* vp : {"vision_tower.", "model.visual.", "visual."}) {
    if (raw.rfind(vp, 0) == 0) return std::nullopt;
  }
  // Some wrapped/VLM checkpoints prefix the language tower; strip it so keys
  // match the canonical "model.*" / "lm_head.*" forms (e.g. Qwen3.5's
  // "language_model.model.embed_tokens.weight" -> "model.embed_tokens.weight").
  const std::string prefix = kLanguageModelPrefix;
  if (raw.rfind(prefix, 0) == 0) {
    return raw.substr(prefix.size());
  }
  return raw;  // already canonical
}

std::unordered_map<std::string, std::string> parse_shard_index(const nlohmann::json& index_json) {
  auto it = index_json.find("weight_map");
  if (it == index_json.end() || !it->is_object()) {
    throw std::runtime_error("safetensors index: missing 'weight_map' object");
  }
  return it->get<std::unordered_map<std::string, std::string>>();
}

std::vector<std::string> shard_files(
    const std::unordered_map<std::string, std::string>& weight_map) {
  std::set<std::string> uniq;
  for (const auto& [key, file] : weight_map) uniq.insert(file);
  return {uniq.begin(), uniq.end()};
}

const mx::array& Weights::at(const std::string& key) const {
  auto it = tensors.find(key);
  if (it == tensors.end()) {
    throw std::runtime_error("weights: missing tensor '" + key + "'");
  }
  return it->second;
}

bool Weights::is_quantized(const std::string& weight_key, QuantParams& out) const {
  static const std::string kWeightSuffix = ".weight";
  if (!ends_with(weight_key, kWeightSuffix)) return false;
  const std::string base = weight_key.substr(0, weight_key.size() - kWeightSuffix.size());
  if (!tensors.count(base + ".scales")) return false;
  auto it = quant.find(base);
  out = (it != quant.end()) ? it->second : QuantParams{};
  return true;
}

std::string Weights::summary() const {
  std::vector<std::string> keys;
  keys.reserve(tensors.size());
  for (const auto& [k, _] : tensors) keys.push_back(k);
  std::sort(keys.begin(), keys.end());

  std::ostringstream os;
  for (const auto& k : keys) {
    const auto& a = tensors.at(k);
    const auto& shape = a.shape();
    os << k << "  [";
    for (size_t i = 0; i < shape.size(); ++i) {
      os << shape[i] << (i + 1 < shape.size() ? ", " : "");
    }
    os << "]  " << dtype_name(a.dtype()) << "\n";
  }
  return os.str();
}

namespace {
// Merge one shard's tensors into `out`, applying sanitize + fp16 cast.
void absorb(std::unordered_map<std::string, mx::array>& out,
            const std::unordered_map<std::string, mx::array>& shard) {
  for (const auto& [raw, arr] : shard) {
    auto canon = sanitize_key(raw);
    if (!canon) continue;  // dropped buffer
    // Cast only floating tensors to fp16; packed 4-bit weights (uint32) and
    // other integer tensors are kept as-is. Exception: Gated-DeltaNet's `A_log`
    // (the per-head decay log-rate) is kept in its source fp32 — the recurrence
    // exponentiates it twice (exp(-exp(A_log))) and fp16 there visibly drifts the
    // decay. Mirrors mlx_lm's cast_predicate, which excludes A_log from the cast.
    const bool keep_dtype = ends_with(*canon, ".A_log");
    mx::array value = (!keep_dtype && mx::issubdtype(arr.dtype(), mx::floating))
                          ? mx::astype(arr, mx::float16)
                          : arr;
    out.emplace(*canon, value);
  }
}
}  // namespace

namespace {
// Some embedding checkpoints export the decoder backbone at the *root* — keys
// like "layers.N.*", "embed_tokens.weight", "norm.weight" with no "model."
// prefix and no "lm_head" (Qwen3-Embedding, saved via AutoModel rather than
// AutoModelForCausalLM). Normalize to the canonical "model.*" layout the engine
// expects; the tied-embedding head path supplies the (absent) lm_head. Detected
// by an unprefixed "embed_tokens.weight" with no "model."-prefixed sibling, so
// canonical checkpoints are left untouched.
void normalize_backbone_root_keys(Weights& w) {
  if (w.tensors.count("model.embed_tokens.weight") || !w.tensors.count("embed_tokens.weight"))
    return;
  std::unordered_map<std::string, mx::array> remapped;
  remapped.reserve(w.tensors.size());
  for (auto& [key, arr] : w.tensors) {
    const bool canonical = key.rfind("model.", 0) == 0 || key.rfind("lm_head", 0) == 0;
    remapped.emplace(canonical ? key : "model." + key, arr);
  }
  w.tensors = std::move(remapped);
  log::info("weights: normalized backbone-root keys to model.* (embedding checkpoint)");
}

// Mixture-of-experts checkpoints store the per-layer experts one of two ways:
//   - raw HF: per-expert tensors "model.layers.{l}.mlp.experts.{e}.{proj}.weight"
//   - mlx (pre-stacked): "model.layers.{l}.mlp.switch_mlp.{proj}.weight" of shape
//     (num_experts, out, in) — what mlx_lm's Model.sanitize produces at convert.
// The forward pass (gather_mm / gather_qmm) consumes the stacked form, so for the
// raw layout we stack experts 0..num_experts-1 here, mirroring that sanitize step.
// Quantized experts also carry ".scales"/".biases" siblings, which are stacked the
// same way. Pre-stacked checkpoints have no "experts.0.*" key and are left as-is.
void stack_moe_experts(Weights& w, const ModelConfig& cfg) {
  if (cfg.num_experts <= 0) return;
  static const char* kProjs[] = {"gate_proj", "up_proj", "down_proj"};
  static const char* kSuffixes[] = {".weight", ".scales", ".biases"};
  for (int l = 0; l < cfg.n_layers; ++l) {
    const std::string base = "model.layers." + std::to_string(l) + ".mlp.";
    for (const char* proj : kProjs) {
      // Only act on the raw per-expert layout (expert 0 present).
      if (!w.tensors.count(base + "experts.0." + proj + ".weight")) continue;
      for (const char* suffix : kSuffixes) {
        const std::string e0 = base + "experts.0." + proj + suffix;
        if (!w.tensors.count(e0)) continue;  // e.g. no .biases for an fp16 expert
        std::vector<mx::array> stack;
        stack.reserve(cfg.num_experts);
        for (int e = 0; e < cfg.num_experts; ++e) {
          const std::string key = base + "experts." + std::to_string(e) + "." + proj + suffix;
          stack.push_back(w.tensors.at(key));
          w.tensors.erase(key);
        }
        w.tensors.emplace(base + "switch_mlp." + proj + suffix, mx::stack(stack));
      }
    }
  }
}

// Record quant params for every quantized weight: any base with a ".scales"
// sibling is quantized; its group_size/bits come from the model config (per-
// module override if present, else the model defaults).
void index_quantized(Weights& w, const ModelConfig& cfg) {
  static const std::string kScales = ".scales";
  for (const auto& [key, _] : w.tensors) {
    if (!ends_with(key, kScales)) continue;
    const std::string base = key.substr(0, key.size() - kScales.size());
    w.quant.emplace(base, cfg.quant_for(base));
  }
}
}  // namespace

Weights load_weights(const std::string& model_dir, const ModelConfig& cfg) {
  const std::string index_path = model_dir + "/model.safetensors.index.json";
  Weights w;

  std::ifstream index_file(index_path);
  if (index_file) {
    nlohmann::json index_json;
    index_file >> index_json;
    auto weight_map = parse_shard_index(index_json);
    const auto files = shard_files(weight_map);
    log::debug("weights: sharded checkpoint, {} files", files.size());
    for (const auto& file : files) {
      log::debug("weights: loading shard {}", file);
      absorb(w.tensors, mx::load_safetensors(model_dir + "/" + file).first);
    }
  } else {
    const std::string single = model_dir + "/model.safetensors";
    if (!std::ifstream(single)) {
      throw std::runtime_error("weights: no model.safetensors[.index.json] in '" + model_dir + "'");
    }
    log::debug("weights: single-file checkpoint");
    absorb(w.tensors, mx::load_safetensors(single).first);
  }

  std::size_t non_fp16 = 0;
  for (const auto& [_, a] : w.tensors)
    if (a.dtype() != mx::float16) ++non_fp16;
  log::info("weights: loaded {} tensors from '{}' ({} non-fp16)", w.tensors.size(), model_dir,
            non_fp16);
  if (non_fp16 > 0)
    log::warn("weights: {} tensors are not fp16 (expected for quantized models)", non_fp16);

  normalize_backbone_root_keys(w);  // embedding checkpoints: backbone-root -> model.*
  stack_moe_experts(w, cfg);  // raw per-expert MoE tensors -> stacked switch_mlp
  index_quantized(w, cfg);
  if (!w.quant.empty())
    log::info("weights: {} quantized weights detected", w.quant.size());
  return w;
}

}  // namespace mlxforge
