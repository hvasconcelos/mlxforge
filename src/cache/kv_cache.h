// Single-sequence KV cache. The simplest form of the prefill/decode split that
// underlies the batched cache (BatchKVCache) the server needs: prefill fills the
// cache once; each decode step appends one token's K/V and attends over the
// cached history.
//
// Storage is per layer a small component vector: empty until written, one array
// when dense ((1, n_kv_heads, S, head_dim) fp16), three when quantized (the
// mx::quantize triplet, see cache/kv_quant.h). The dense pair API and the
// quantized API are mutually exclusive, selected by the KVQuantConfig at
// construction; the model dispatches on quantized() (model/sdpa.h).
#pragma once

#include <optional>
#include <utility>
#include <vector>

#include "cache/kv_quant.h"

#include "mlx/array.h"

namespace mlxforge {

namespace mx = mlx::core;

class KVCache {
 public:
  static constexpr int kStep = 256;  // quantized-storage block growth size

  explicit KVCache(int n_layers, KVQuantConfig qcfg = {})
      : keys_(n_layers), values_(n_layers), conv_state_(n_layers), recur_state_(n_layers),
        qcfg_(qcfg) {}

  // Tokens written so far (== sequence length). Used as the RoPE position
  // offset for the next chunk. Stays fixed across a single token's layer sweep;
  // call advance() once after all layers are processed.
  int offset() const { return offset_; }
  void advance(int n_tokens) { offset_ += n_tokens; }

  int n_layers() const { return static_cast<int>(keys_.size()); }

  bool quantized() const { return qcfg_.enabled(); }
  const KVQuantConfig& quant_config() const { return qcfg_; }

  // Stored K/V for a layer (each (1, n_kv_heads, offset, head_dim), no capacity
  // padding). Valid only after the layer has been written; dense caches only.
  // Lets a prefilled single sequence be handed to the batched cache for
  // continuous-batching decode (BatchKVCache::from_single_sequence).
  std::pair<mx::array, mx::array> fetch(int layer) const;

  // Append this layer's K/V (each (1, n_kv_heads, L, head_dim)) along the
  // sequence axis and return the full cached (keys, values) to attend over.
  // Dense caches only (throws on a quantized cache).
  std::pair<mx::array, mx::array> update_and_fetch(int layer, const mx::array& k,
                                                   const mx::array& v);

  // Quantized counterpart: quantize the incoming K/V (mx::quantize, per
  // position), append each triplet component along the sequence axis, and return
  // the full cached triplets for quantized_sdpa. Quantized caches only.
  QuantizedKVSlice update_and_fetch_quantized(int layer, const mx::array& k, const mx::array& v);

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
  // Per layer: empty until written; 1 component dense, 3 quantized.
  std::vector<std::vector<mx::array>> keys_;
  std::vector<std::vector<mx::array>> values_;
  std::vector<std::optional<mx::array>> conv_state_;
  std::vector<std::optional<mx::array>> recur_state_;
  KVQuantConfig qcfg_;
  int offset_ = 0;
};

}  // namespace mlxforge
