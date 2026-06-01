#include "core/weights.h"

#include <algorithm>
#include <fstream>
#include <set>
#include <sstream>
#include <stdexcept>

#include "mlx/io.h"
#include "mlx/ops.h"

namespace mx = mlx::core;

namespace xllm {

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
    out.emplace(*canon, mx::astype(arr, mx::float16));
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
    for (const auto& file : shard_files(weight_map)) {
      absorb(w.tensors, mx::load_safetensors(model_dir + "/" + file).first);
    }
  } else {
    const std::string single = model_dir + "/model.safetensors";
    if (!std::ifstream(single)) {
      throw std::runtime_error("weights: no model.safetensors[.index.json] in '" + model_dir + "'");
    }
    absorb(w.tensors, mx::load_safetensors(single).first);
  }
  return w;
}

}  // namespace xllm
