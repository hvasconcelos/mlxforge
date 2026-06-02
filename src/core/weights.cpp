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
  // Some wrapped/VLM checkpoints prefix the language tower; strip it so keys
  // match the canonical "model.*" / "lm_head.*" forms.
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
    // other integer tensors are kept as-is.
    mx::array value = mx::issubdtype(arr.dtype(), mx::floating) ? mx::astype(arr, mx::float16) : arr;
    out.emplace(*canon, value);
  }
}
}  // namespace

Weights load_weights(const std::string& model_dir) {
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
  return w;
}

}  // namespace mlxforge
