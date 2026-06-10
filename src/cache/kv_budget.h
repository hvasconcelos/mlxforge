// KV memory admission gate (the real OOM guard).
//
// MLX allocates until Metal fails, so before admitting a batch we project its
// peak KV footprint and refuse/queue if it would exceed a configured budget.
// Per token the cache holds K and V for every layer:
//   bytes/token = 2 (K+V) * n_layers * n_kv_heads * head_dim * sizeof(fp16)
// For Llama-3.2-1B that is 2*16*8*64*2 = 32 KiB/token.
#pragma once

#include <cstddef>

#include "cache/kv_quant.h"
#include "core/config.h"

namespace mlxforge {

class KVBudget {
 public:
  // budget_bytes == 0 means "unbounded" (admission always allowed). With a
  // KVQuantConfig the per-token figure accounts for the quantized triplet
  // storage (packed words + per-group fp16 scales and biases) instead of fp16.
  KVBudget(const ModelConfig& cfg, std::size_t budget_bytes, KVQuantConfig kv_quant = {});

  // KV bytes consumed by one token across all layers (the 32 KiB/token figure).
  std::size_t bytes_per_token() const { return bytes_per_token_; }

  // Projected peak KV bytes for a batch of `batch` sequences each reaching up to
  // (max_len + max_new) tokens.
  std::size_t project_bytes(int max_len, int max_new, int batch) const;

  // Whether such a batch fits within the budget.
  bool can_admit(int max_len, int max_new, int batch) const;

  std::size_t budget_bytes() const { return budget_bytes_; }

  // Fixed per-sequence bytes of a hybrid model's linear-attention streaming state
  // (0 for non-hybrid models). Does not grow with sequence length.
  std::size_t linear_state_bytes() const { return linear_state_bytes_; }

 private:
  std::size_t bytes_per_token_;
  std::size_t linear_state_bytes_ = 0;
  std::size_t budget_bytes_;
};

}  // namespace mlxforge
