// Qwen3 MoE (sparse mixture-of-experts) decoder model. Extends Qwen3Model (so it
// inherits QK-Norm) and overrides feed_forward() to route each token through its
// top-k experts on the sparse layers (config().is_moe_layer(layer)), falling back
// to the dense SwiGLU mlp() on the remaining layers. Selected by create_model()
// when a checkpoint declares experts (num_experts > 0).
#pragma once

#include <utility>

#include "model/qwen3.h"

namespace mlxforge {

class Qwen3MoeModel : public Qwen3Model {
 public:
  Qwen3MoeModel(ModelConfig config, Weights weights)
      : Qwen3Model(std::move(config), std::move(weights)) {}

  // Sparse mixture-of-experts MLP sublayer. Routes each token to its top-k
  // experts (softmax over the router, top-k via argpartition), runs a per-expert
  // SwiGLU, and sums the expert outputs weighted by the routing scores. Same
  // residual-stream shape (B, L, hidden) as mlp(). Public: a golden-reference
  // test exercises it in isolation.
  mx::array moe_mlp(const mx::array& x, int layer) const;

 protected:
  // Sparse layers route through moe_mlp(); dense layers use the inherited mlp().
  mx::array feed_forward(const mx::array& x, int layer) const override;

 private:
  // Per-expert gather matmul against a stacked SwitchLinear weight
  // "model.layers.{layer}.mlp.switch_mlp.{proj}.weight" (num_experts, out, in),
  // gathering the expert rows named by `inds`. Quantization-aware (gather_qmm
  // when the weight has a ".scales" sibling, else gather_mm), mirroring linear().
  mx::array switch_linear(const mx::array& x, int layer, const char* proj,
                          const mx::array& inds) const;
};

}  // namespace mlxforge
