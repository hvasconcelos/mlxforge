#include "model/qwen3.h"

namespace mlxforge {

mx::array Qwen3Model::norm_qk_head(const mx::array& h, int layer, bool is_query) const {
  // RMSNorm over head_dim, per Q/K head, before RoPE (the Qwen3 attention delta).
  const char* suffix = is_query ? "self_attn.q_norm.weight" : "self_attn.k_norm.weight";
  return rms_norm(h, layer_w(layer, suffix));
}

}  // namespace mlxforge
