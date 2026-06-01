#include "cache/kv_budget.h"

namespace mlxforge {

namespace {
constexpr std::size_t kFp16Bytes = 2;
}

KVBudget::KVBudget(const ModelConfig& cfg, std::size_t budget_bytes)
    : bytes_per_token_(2 /*K and V*/ * static_cast<std::size_t>(cfg.n_layers) * cfg.n_kv_heads *
                       cfg.head_dim * kFp16Bytes),
      budget_bytes_(budget_bytes) {}

std::size_t KVBudget::project_bytes(int max_len, int max_new, int batch) const {
  return static_cast<std::size_t>(max_len + max_new) * batch * bytes_per_token_;
}

bool KVBudget::can_admit(int max_len, int max_new, int batch) const {
  if (budget_bytes_ == 0) return true;  // unbounded
  return project_bytes(max_len, max_new, batch) <= budget_bytes_;
}

}  // namespace mlxforge
