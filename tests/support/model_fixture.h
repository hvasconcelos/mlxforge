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
    Weights w = load_weights(model_dir());
    return LlamaModel(std::move(cfg), std::move(w));
  }();
  return model;
}

}  // namespace mlxforge::test
