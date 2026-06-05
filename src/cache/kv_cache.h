// Single-sequence KV cache. The simplest form of the prefill/decode split that
// underlies the batched cache (BatchKVCache) the server needs: prefill fills the
// cache once; each decode step appends one token's K/V and attends over the
// cached history.
#pragma once

#include <optional>
#include <vector>

#include "mlx/array.h"

namespace mlxforge {

namespace mx = mlx::core;

class KVCache {
 public:
  explicit KVCache(int n_layers)
      : keys_(n_layers), values_(n_layers), conv_state_(n_layers), recur_state_(n_layers) {}

  // Tokens written so far (== sequence length). Used as the RoPE position
  // offset for the next chunk. Stays fixed across a single token's layer sweep;
  // call advance() once after all layers are processed.
  int offset() const { return offset_; }
  void advance(int n_tokens) { offset_ += n_tokens; }

  // Append this layer's K/V (each (1, n_kv_heads, L, head_dim)) along the
  // sequence axis and return the full cached (keys, values) to attend over.
  std::pair<mx::array, mx::array> update_and_fetch(int layer, const mx::array& k,
                                                   const mx::array& v);

  // Gated-DeltaNet recurrent state for hybrid models (Qwen3.5): the linear
  // layers carry a fixed conv buffer (1, K-1, conv_dim) and a delta-rule state
  // (1, Hv, Dv, Dk) instead of a growing KV. Lazily set by the model on the first
  // write; pure-attention models never touch these slots.
  bool has_linear_state(int layer) const { return conv_state_[layer].has_value(); }
  std::pair<mx::array, mx::array> linear_state(int layer) const {
    return {*conv_state_[layer], *recur_state_[layer]};
  }
  void set_linear_state(int layer, const mx::array& conv_state, const mx::array& recur_state) {
    conv_state_[layer] = conv_state;
    recur_state_[layer] = recur_state;
  }

 private:
  std::vector<std::optional<mx::array>> keys_;
  std::vector<std::optional<mx::array>> values_;
  std::vector<std::optional<mx::array>> conv_state_;
  std::vector<std::optional<mx::array>> recur_state_;
  int offset_ = 0;
};

}  // namespace mlxforge
