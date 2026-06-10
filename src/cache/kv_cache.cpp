#include "cache/kv_cache.h"

#include <stdexcept>

#include "mlx/ops.h"

namespace mlxforge {

std::pair<mx::array, mx::array> KVCache::fetch(int layer) const {
  if (quantized()) throw std::logic_error("KVCache::fetch: cache is quantized");
  return {keys_[layer][0], values_[layer][0]};
}

std::pair<mx::array, mx::array> KVCache::update_and_fetch(int layer, const mx::array& k,
                                                          const mx::array& v) {
  if (quantized()) throw std::logic_error("KVCache::update_and_fetch: cache is quantized");
  if (keys_[layer].empty()) {
    keys_[layer] = {k};
    values_[layer] = {v};
  } else {
    // Sequence axis is 2: (1, n_kv_heads, S, head_dim).
    keys_[layer][0] = mx::concatenate({keys_[layer][0], k}, /*axis=*/2);
    values_[layer][0] = mx::concatenate({values_[layer][0], v}, /*axis=*/2);
  }
  return {keys_[layer][0], values_[layer][0]};
}

QuantizedKVSlice KVCache::update_and_fetch_quantized(int layer, const mx::array& k,
                                                     const mx::array& v) {
  if (!quantized())
    throw std::logic_error("KVCache::update_and_fetch_quantized: cache is dense");
  // Block-grown storage written at offset() (the prefill/decode protocol bumps
  // it via advance() once per token sweep), mirroring mlx-lm's QuantizedKVCache
  // exactly — the golden gates depend on the matching buffer strides.
  QuantizedKV ks =
      quantize_and_update(keys_[layer], k, qcfg_.group_size, qcfg_.bits, offset_, kStep);
  QuantizedKV vs =
      quantize_and_update(values_[layer], v, qcfg_.group_size, qcfg_.bits, offset_, kStep);
  return {ks, vs};
}

}  // namespace mlxforge
