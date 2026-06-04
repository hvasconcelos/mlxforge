// LLaMA-family decoder model: the plain DecoderModel with no per-architecture
// overrides (no QK-Norm, dense SwiGLU MLP). Serves Llama-3.2 and other
// GGUF/safetensors Llama-style checkpoints. Selected by create_model() when a
// checkpoint has neither q_norm weights nor MoE experts.
#pragma once

#include <utility>

#include "model/decoder_model.h"

namespace mlxforge {

class LlamaModel : public DecoderModel {
 public:
  LlamaModel(ModelConfig config, Weights weights)
      : DecoderModel(std::move(config), std::move(weights)) {}
};

}  // namespace mlxforge
