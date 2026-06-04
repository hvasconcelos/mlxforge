#include "model/model_factory.h"

#include <utility>

#include "model/llama.h"
#include "model/qwen3.h"
#include "model/qwen3_moe.h"

namespace mlxforge {

std::unique_ptr<DecoderModel> create_model(ModelConfig config, Weights weights) {
  // MoE is checked first: a Qwen3 MoE checkpoint also carries q_norm weights, and
  // Qwen3MoeModel inherits QK-Norm from Qwen3Model.
  if (config.num_experts > 0) {
    return std::make_unique<Qwen3MoeModel>(std::move(config), std::move(weights));
  }
  if (weights.has("model.layers.0.self_attn.q_norm.weight")) {
    return std::make_unique<Qwen3Model>(std::move(config), std::move(weights));
  }
  return std::make_unique<LlamaModel>(std::move(config), std::move(weights));
}

}  // namespace mlxforge
