// Qwen3 dense decoder model. The single delta from Llama is QK-Norm: an extra
// RMSNorm applied to each Q/K head (over head_dim) before RoPE, using the
// per-layer self_attn.{q,k}_norm.weight tensors. The MLP is still dense SwiGLU,
// so feed_forward() is inherited. Selected by create_model() when a checkpoint
// carries q_norm weights but no MoE experts.
#pragma once

#include <utility>

#include "model/decoder_model.h"

namespace mlxforge {

class Qwen3Model : public DecoderModel {
 public:
  Qwen3Model(ModelConfig config, Weights weights)
      : DecoderModel(std::move(config), std::move(weights)) {}

 protected:
  mx::array norm_qk_head(const mx::array& h, int layer, bool is_query) const override;
};

}  // namespace mlxforge
