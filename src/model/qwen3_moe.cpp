#include "model/qwen3_moe.h"

#include <string>
#include <string_view>

#include "mlx/ops.h"

namespace mlxforge {

mx::array Qwen3MoeModel::switch_linear(const mx::array& x, int layer, const char* proj,
                                       const mx::array& inds) const {
  // Stacked SwitchLinear weight (num_experts, out, in); gather the per-token
  // expert rows named by `inds`. Quantization is detected the same way as the
  // dense linear() (a "<base>.scales" sibling), so a mixed checkpoint just works.
  const std::string wkey = layer_key(layer, std::string("mlp.switch_mlp.") + proj + ".weight");
  QuantParams qp;
  if (w_.is_quantized(wkey, qp)) {
    static constexpr std::string_view kWeightSuffix = ".weight";
    const std::string base = wkey.substr(0, wkey.size() - kWeightSuffix.size());
    return mx::gather_qmm(x, w_.at(wkey), w_.at(base + ".scales"), w_.at(base + ".biases"),
                          /*lhs_indices=*/std::nullopt, /*rhs_indices=*/inds,
                          /*transpose=*/true, qp.group_size, qp.bits);
  }
  // Dense: x @ W^T per expert. W is (E, out, in); swap to (E, in, out) for gather_mm.
  return mx::gather_mm(x, mx::swapaxes(w_.at(wkey), -1, -2),
                       /*lhs_indices=*/std::nullopt, /*rhs_indices=*/inds);
}

mx::array Qwen3MoeModel::moe_mlp(const mx::array& x, int layer) const {
  // Router logits -> probabilities. softmax is computed in fp32 (precise) to
  // match mlx_lm's softmax(..., precise=True); the gate may be 8-bit quantized,
  // which linear() handles transparently.
  mx::array gates = mx::softmax(linear(x, layer_key(layer, "mlp.gate.weight")),
                                /*axis=*/-1, /*precise=*/true);  // (B, L, n_experts)
  const int B = x.shape()[0];
  const int L = x.shape()[1];
  const int n_experts = cfg_.num_experts;
  const int k = cfg_.num_experts_per_tok;

  // Top-k experts: argpartition leaves the k largest in the last k positions; the
  // exact intra-k order is irrelevant (the outputs are summed). Cast to uint32 for
  // the gather indices.
  mx::array part = mx::argpartition(gates, /*kth=*/n_experts - k, /*axis=*/-1);
  mx::array inds =
      mx::astype(mx::slice(part, {0, 0, n_experts - k}, {B, L, n_experts}), mx::uint32);
  mx::array scores = mx::take_along_axis(gates, inds, /*axis=*/-1);  // (B, L, k)
  if (cfg_.norm_topk_prob) {
    scores = mx::divide(scores, mx::sum(scores, /*axis=*/-1, /*keepdims=*/true));
  }

  // Per-expert SwiGLU (SwitchGLU): broadcast x over the expert axis, gather each
  // token's chosen experts, then collapse the matmul singleton dim. Shapes:
  //   xe (B,L,1,1,hidden) -> up/gate (B,L,k,1,moe_inter) -> y (B,L,k,1,hidden).
  mx::array xe = mx::expand_dims(x, {-2, -3});
  mx::array up = switch_linear(xe, layer, "up_proj", inds);
  mx::array gate = switch_linear(xe, layer, "gate_proj", inds);
  mx::array silu = mx::multiply(gate, mx::sigmoid(gate));
  mx::array y = switch_linear(mx::multiply(silu, up), layer, "down_proj", inds);
  y = mx::squeeze(y, /*axis=*/-2);  // (B, L, k, hidden)

  // Weighted sum over the k selected experts.
  mx::array weighted = mx::multiply(y, mx::expand_dims(scores, -1));  // (B, L, k, hidden)
  return mx::sum(weighted, /*axis=*/-2);  // (B, L, hidden)
}

mx::array Qwen3MoeModel::feed_forward(const mx::array& x, int layer) const {
  return cfg_.is_moe_layer(layer) ? moe_mlp(x, layer) : mlp(x, layer);
}

}  // namespace mlxforge
