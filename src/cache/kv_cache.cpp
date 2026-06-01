#include "cache/kv_cache.h"

#include "mlx/ops.h"

namespace mlxforge {

std::pair<mx::array, mx::array> KVCache::update_and_fetch(int layer, const mx::array& k,
                                                          const mx::array& v) {
  if (!keys_[layer]) {
    keys_[layer] = k;
    values_[layer] = v;
  } else {
    // Sequence axis is 2: (1, n_kv_heads, S, head_dim).
    keys_[layer] = mx::concatenate({*keys_[layer], k}, /*axis=*/2);
    values_[layer] = mx::concatenate({*values_[layer], v}, /*axis=*/2);
  }
  return {*keys_[layer], *values_[layer]};
}

}  // namespace mlxforge
