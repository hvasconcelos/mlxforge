#include "cache/kv_budget.h"

#include <algorithm>

namespace mlxforge {

namespace {
constexpr std::size_t kFp16Bytes = 2;
constexpr std::size_t kFp32Bytes = 4;

// Count the layers that grow a KV cache per token. For a hybrid model (Qwen3.5)
// only the full-attention layers do; the Gated-DeltaNet linear layers carry a
// fixed-size recurrent state counted separately. Non-hybrid models: every layer.
std::size_t kv_layer_count(const ModelConfig& cfg) {
  if (cfg.full_attention_interval <= 0) return cfg.n_layers;
  std::size_t n = 0;
  for (int i = 0; i < cfg.n_layers; ++i)
    if (!cfg.is_linear_layer(i)) ++n;
  return n;
}

// Fixed per-sequence bytes of the linear layers' streaming state: a conv buffer
// (K-1, conv_dim) in fp16 and a delta-rule state (Hv, Dv, Dk) in fp32, per linear
// layer. Zero for non-hybrid models. Unlike KV this does not grow with length.
std::size_t compute_linear_state_bytes(const ModelConfig& cfg) {
  if (cfg.full_attention_interval <= 0) return 0;
  std::size_t linear_layers = 0;
  for (int i = 0; i < cfg.n_layers; ++i)
    if (cfg.is_linear_layer(i)) ++linear_layers;
  const std::size_t key_dim = static_cast<std::size_t>(cfg.linear_num_key_heads) *
                              cfg.linear_key_head_dim;
  const std::size_t value_dim = static_cast<std::size_t>(cfg.linear_num_value_heads) *
                                cfg.linear_value_head_dim;
  const std::size_t conv_dim = 2 * key_dim + value_dim;
  const std::size_t conv_bytes =
      static_cast<std::size_t>(std::max(cfg.linear_conv_kernel_dim - 1, 0)) * conv_dim * kFp16Bytes;
  const std::size_t recur_bytes = value_dim * cfg.linear_key_head_dim * kFp32Bytes;
  return linear_layers * (conv_bytes + recur_bytes);
}
// One K-or-V row of head_dim values: fp16 when dense; packed bits plus the
// per-group fp16 scale and bias when quantized (e.g. D=64/g=64: 128 B fp16,
// 68 B at 8-bit, 36 B at 4-bit).
std::size_t per_head_row_bytes(const ModelConfig& cfg, KVQuantConfig q) {
  const std::size_t d = static_cast<std::size_t>(cfg.head_dim);
  if (!q.enabled()) return d * kFp16Bytes;
  return d * q.bits / 8 + (d / q.group_size) * 2 /*scale+bias*/ * kFp16Bytes;
}
}  // namespace

KVBudget::KVBudget(const ModelConfig& cfg, std::size_t budget_bytes, KVQuantConfig kv_quant)
    : bytes_per_token_(2 /*K and V*/ * kv_layer_count(cfg) * cfg.n_kv_heads *
                       per_head_row_bytes(cfg, kv_quant)),
      linear_state_bytes_(compute_linear_state_bytes(cfg)),
      budget_bytes_(budget_bytes) {}

std::size_t KVBudget::project_bytes(int max_len, int max_new, int batch) const {
  return static_cast<std::size_t>(batch) *
         (linear_state_bytes_ +
          static_cast<std::size_t>(max_len + max_new) * bytes_per_token_);
}

bool KVBudget::can_admit(int max_len, int max_new, int batch) const {
  if (budget_bytes_ == 0) return true;  // unbounded
  return project_bytes(max_len, max_new, batch) <= budget_bytes_;
}

}  // namespace mlxforge
