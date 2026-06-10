// Shared types for quantized KV-cache storage.
//
// Mirrors mlx_lm/models/cache.py::QuantizedKVCache's layout: each cached K or V
// tensor is the 3-tuple mx::quantize() produces — packed uint32 words plus
// per-group fp16 scales and biases — quantized at write time, per position, so
// prefill chunking can never change the stored values. The attention math over
// this storage lives in model/sdpa.h (quantized_sdpa), ported from
// mlx_lm/models/base.py.
#pragma once

#include <vector>

#include "mlx/array.h"

namespace mlxforge {

namespace mx = mlx::core;

// Engine-wide KV-cache quantization setting. bits == 0 keeps the cache dense
// fp16 (the default); 8 or 4 enable quantized storage. group_size must divide
// head_dim (64 and 128 both work with the default 64).
struct KVQuantConfig {
  int bits = 0;
  int group_size = 64;
  bool enabled() const { return bits > 0; }
  bool operator==(const KVQuantConfig& o) const {
    return bits == o.bits && group_size == o.group_size;
  }
  bool operator!=(const KVQuantConfig& o) const { return !(*this == o); }
};

// One quantized tensor: w (..., S, head_dim*bits/32) uint32, scales/biases
// (..., S, head_dim/group_size) in the source dtype (fp16).
struct QuantizedKV {
  mx::array w;
  mx::array scales;
  mx::array biases;
};

// Slice [start, stop) along the sequence axis (2), keeping axes 0/1/3 full.
// Shared by both caches' growth/eviction surgery.
mx::array slice_seq(const mx::array& a, int start, int stop);

// The populated K/V slice of one layer, ready for quantized_sdpa.
struct QuantizedKVSlice {
  QuantizedKV k;
  QuantizedKV v;
};

// Write `in`'s components (1 dense, 3 quantized) at sequence positions
// [prev, prev + L) of `store`, growing capacity in `step` blocks (zeros +
// concatenate, trimming any unused tail of the last block first), and return
// each component's populated slice [..., :prev+L, :]. This block-grow +
// slice_update + slice-view strategy deliberately mirrors mlx-lm's caches
// bit-for-bit — the returned views' buffer shapes/strides affect kernel
// accumulation order, and the exact-token golden gates depend on it.
std::vector<mx::array> update_kv_components(std::vector<mx::array>& store,
                                            const std::vector<mx::array>& in, int prev, int step);

// Quantize `x` (mx::quantize, per position) and write the resulting triplet into
// `store` at sequence position `pos` via update_kv_components, returning the
// populated triplet for quantized_sdpa. Shared by KVCache and BatchKVCache so
// the quantize-then-update op order is identical on both paths.
QuantizedKV quantize_and_update(std::vector<mx::array>& store, const mx::array& x, int group_size,
                                int bits, int pos, int step);

}  // namespace mlxforge
