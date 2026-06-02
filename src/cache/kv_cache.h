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
  explicit KVCache(int n_layers) : keys_(n_layers), values_(n_layers) {}

  // Tokens written so far (== sequence length). Used as the RoPE position
  // offset for the next chunk. Stays fixed across a single token's layer sweep;
  // call advance() once after all layers are processed.
  int offset() const { return offset_; }
  void advance(int n_tokens) { offset_ += n_tokens; }

  // Append this layer's K/V (each (1, n_kv_heads, L, head_dim)) along the
  // sequence axis and return the full cached (keys, values) to attend over.
  std::pair<mx::array, mx::array> update_and_fetch(int layer, const mx::array& k,
                                                   const mx::array& v);

 private:
  std::vector<std::optional<mx::array>> keys_;
  std::vector<std::optional<mx::array>> values_;
  int offset_ = 0;
};

}  // namespace mlxforge
