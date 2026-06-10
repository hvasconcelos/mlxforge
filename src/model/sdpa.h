// Cache-aware scaled-dot-product attention dispatch.
//
// sdpa_with_cache() is the one seam the models call after projecting Q/K/V: it
// appends K/V to the cache and attends over the stored history, picking the
// dense fast::scaled_dot_product_attention kernel or the hand-rolled quantized
// path by the cache's KVQuantConfig — the C++ twin of mlx_lm/models/base.py::
// scaled_dot_product_attention's hasattr(cache, "bits") dispatch.
//
// quantized_sdpa() is ported op-for-op from mlx_lm/models/base.py::
// quantized_scaled_dot_product_attention (MLX has no fused quantized SDPA
// kernel): fp16 q*scale, GQA via a (B, n_kv_heads, n_repeats, L, D) reshape,
// mx::quantized_matmul for both the scores and the output, and a precise
// softmax. Deviating from any of those breaks the exact-token golden gate.
#pragma once

#include <optional>
#include <string>

#include "cache/batch_kv_cache.h"
#include "cache/kv_cache.h"
#include "cache/kv_quant.h"

#include "mlx/array.h"

namespace mlxforge {

namespace mx = mlx::core;

// Attention over quantized K/V triplets. `mask_mode` is "causal" or "" (mirrors
// fast::scaled_dot_product_attention); `mask` is the batched additive fp16
// (B, 1, N, T_kv) mask, exclusive with mask_mode.
mx::array quantized_sdpa(const mx::array& q, const QuantizedKV& keys, const QuantizedKV& values,
                         float scale, const std::string& mask_mode,
                         const std::optional<mx::array>& mask, int group_size, int bits);

// Single-stream path (prefill is causal, decode unmasked). `cache` may be null
// (full recompute): plain fast SDPA with no cache write.
mx::array sdpa_with_cache(const mx::array& q, const mx::array& k, const mx::array& v,
                          KVCache* cache, int layer, float scale, const std::string& mask_mode);

// Continuous-batching path: per-row additive fp16 mask from batch_mask().
mx::array sdpa_with_cache(const mx::array& q, const mx::array& k, const mx::array& v,
                          BatchKVCache& cache, int layer, float scale, const mx::array& mask);

}  // namespace mlxforge
