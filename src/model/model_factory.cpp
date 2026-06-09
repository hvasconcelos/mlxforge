#include "model/model_factory.h"

#include <utility>

#include "model/llama.h"
#include "model/qwen3.h"
#include "model/qwen3_5.h"
#include "model/qwen3_moe.h"
#include "model/qwen3_vl.h"

namespace mlxforge {

std::unique_ptr<DecoderModel> create_model(ModelConfig config, Weights weights) {
  // Qwen3.5 is checked first: its hybrid checkpoint also carries q_norm weights
  // (on the full-attention layers), but the linear layers and per-layer dispatch
  // need Qwen35Model. full_attention_interval > 0 marks the hybrid layout.
  if (config.full_attention_interval > 0) {
    return std::make_unique<Qwen35Model>(std::move(config), std::move(weights));
  }
  // MoE next: a Qwen3 MoE checkpoint also carries q_norm weights, and
  // Qwen3MoeModel inherits QK-Norm from Qwen3Model.
  if (config.num_experts > 0) {
    return std::make_unique<Qwen3MoeModel>(std::move(config), std::move(weights));
  }
  // Dense Qwen3-VL: a Qwen3 backbone (q_norm present) plus a ViT tower and
  // interleaved M-RoPE. (The MoE-VL variant is not yet a distinct class.)
  if (config.has_vision_tower()) {
    return std::make_unique<Qwen3VLModel>(std::move(config), std::move(weights));
  }
  if (weights.has("model.layers.0.self_attn.q_norm.weight")) {
    return std::make_unique<Qwen3Model>(std::move(config), std::move(weights));
  }
  return std::make_unique<LlamaModel>(std::move(config), std::move(weights));
}

}  // namespace mlxforge
