// Shared, lazily-loaded LlamaModel for integration tests (loads the real model
// from MLXFORGE_MODEL_DIR once). Numerically-sensitive stories validate against the
// golden reference using this fixture; if the model dir is absent the test
// short-circuits as a pass with a message.
#pragma once

#include <fstream>
#include <string>
#include <utility>

#include "core/config.h"
#include "core/weights.h"
#include "model/llama.h"
#include "model/qwen3.h"
#include "model/qwen3_moe.h"

namespace mlxforge::test {

inline std::string model_dir() { return MLXFORGE_MODEL_DIR; }

inline bool model_available() {
  const std::string d = model_dir();
  return !d.empty() && std::ifstream(d + "/config.json").good();
}

// Loaded once on first use and reused across test cases.
inline LlamaModel& shared_model() {
  static LlamaModel model = [] {
    ModelConfig cfg = ModelConfig::from_file(model_dir() + "/config.json");
    Weights w = load_weights(model_dir(), cfg);
    return LlamaModel(std::move(cfg), std::move(w));
  }();
  return model;
}

// Same, for the Qwen3 dense model (QK-Norm + ChatML integration tests).
inline std::string qwen3_model_dir() { return MLXFORGE_MODEL_DIR_QWEN3; }

inline bool qwen3_model_available() {
  const std::string d = qwen3_model_dir();
  return !d.empty() && std::ifstream(d + "/config.json").good();
}

inline Qwen3Model& shared_qwen3_model() {
  static Qwen3Model model = [] {
    ModelConfig cfg = ModelConfig::from_file(qwen3_model_dir() + "/config.json");
    Weights w = load_weights(qwen3_model_dir(), cfg);
    return Qwen3Model(std::move(cfg), std::move(w));
  }();
  return model;
}

// Same, for the Qwen3 MoE model (sparse expert-routing integration tests).
inline std::string qwen3_moe_model_dir() { return MLXFORGE_MODEL_DIR_QWEN3_MOE; }

inline bool qwen3_moe_model_available() {
  const std::string d = qwen3_moe_model_dir();
  return !d.empty() && std::ifstream(d + "/config.json").good();
}

inline Qwen3MoeModel& shared_qwen3_moe_model() {
  static Qwen3MoeModel model = [] {
    ModelConfig cfg = ModelConfig::from_file(qwen3_moe_model_dir() + "/config.json");
    Weights w = load_weights(qwen3_moe_model_dir(), cfg);
    return Qwen3MoeModel(std::move(cfg), std::move(w));
  }();
  return model;
}

}  // namespace mlxforge::test
