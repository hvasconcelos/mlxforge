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
#include "model/qwen3_5.h"
#include "model/qwen3_moe.h"
#include "model/qwen3_vl.h"
#include "model/vision/vit.h"

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

// Same, for the Qwen3.5 hybrid model (Gated-DeltaNet + gated full-attention).
inline std::string qwen3_5_model_dir() { return MLXFORGE_MODEL_DIR_QWEN3_5; }

inline bool qwen3_5_model_available() {
  const std::string d = qwen3_5_model_dir();
  return !d.empty() && std::ifstream(d + "/config.json").good();
}

inline Qwen35Model& shared_qwen3_5_model() {
  static Qwen35Model model = [] {
    ModelConfig cfg = ModelConfig::from_file(qwen3_5_model_dir() + "/config.json");
    Weights w = load_weights(qwen3_5_model_dir(), cfg);
    return Qwen35Model(std::move(cfg), std::move(w));
  }();
  return model;
}

// Qwen3-VL vision-language model (ViT + multimodal integration tests). The
// config and the (keep_vision) Weights are loaded once and shared: the ViT
// encoder borrows the Weights, the language model takes a (cheap, handle-only)
// copy. Function-local statics, so the shared Weights outlives the borrowers.
inline std::string qwen3_vl_model_dir() { return MLXFORGE_MODEL_DIR_QWEN3_VL; }

inline bool qwen3_vl_model_available() {
  const std::string d = qwen3_vl_model_dir();
  return !d.empty() && std::ifstream(d + "/config.json").good();
}

inline const ModelConfig& qwen3_vl_config() {
  static ModelConfig cfg = ModelConfig::from_file(qwen3_vl_model_dir() + "/config.json");
  return cfg;
}

inline const Weights& qwen3_vl_weights() {
  static Weights weights = load_weights(qwen3_vl_model_dir(), qwen3_vl_config());
  return weights;
}

inline VitEncoder& shared_qwen3_vl_vit() {
  static VitEncoder vit(*qwen3_vl_config().vision, qwen3_vl_weights());
  return vit;
}

inline Qwen3VLModel& shared_qwen3_vl_model() {
  static Qwen3VLModel model(qwen3_vl_config(), qwen3_vl_weights());
  return model;
}

}  // namespace mlxforge::test
