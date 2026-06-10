#include "model/sdpa.h"

#include "mlx/fast.h"
#include "mlx/ops.h"
#include "mlx/utils.h"

namespace mlxforge {

mx::array quantized_sdpa(const mx::array& q, const QuantizedKV& keys, const QuantizedKV& values,
                         float scale, const std::string& mask_mode,
                         const std::optional<mx::array>& mask, int group_size, int bits) {
  const int B = q.shape()[0];
  const int n_q_heads = q.shape()[1];
  const int L = q.shape()[2];
  const int D = q.shape()[3];
  const int n_kv_heads = keys.w.shape()[1];
  const int n_repeats = n_q_heads / n_kv_heads;

  mx::array queries = mx::multiply(q, mx::array(scale, q.dtype()));

  QuantizedKV k = keys;
  QuantizedKV v = values;
  std::optional<mx::array> m = mask;
  if (n_repeats > 1) {
    // GQA: group the query heads under their kv head; the kv triplets gain a
    // broadcast axis. quantized_matmul has no native GQA handling.
    queries = mx::reshape(queries, {B, n_kv_heads, n_repeats, L, D});
    k = {mx::expand_dims(k.w, -3), mx::expand_dims(k.scales, -3), mx::expand_dims(k.biases, -3)};
    v = {mx::expand_dims(v.w, -3), mx::expand_dims(v.scales, -3), mx::expand_dims(v.biases, -3)};
    // Scores are now 5-D (B, n_kv, n_rep, N, T_kv); the (B, 1, N, T_kv) mask
    // must gain the n_rep axis too, or trailing-axis broadcasting would align
    // B against n_rep — silently wrong attention.
    if (m) m = mx::expand_dims(*m, 1);
  }

  mx::array scores =
      mx::quantized_matmul(queries, k.w, k.scales, k.biases, /*transpose=*/true, group_size, bits);

  if (mask_mode == "causal") {
    const int kL = scores.shape().back();
    const int qL = L;
    // Bool is safe here: these are plain ops on the scores, not the fast::SDPA
    // kernel mask (#2894). finfo(...).min (not -inf) matches mlx-lm exactly.
    mx::array q_idx = mx::reshape(mx::arange(kL - qL, kL, 1, mx::int32), {qL, 1});
    mx::array k_idx = mx::reshape(mx::arange(0, kL, 1, mx::int32), {1, kL});
    mx::array causal = mx::greater_equal(q_idx, k_idx);
    const float lowest = static_cast<float>(mx::finfo(scores.dtype()).min);
    scores = mx::where(causal, scores, mx::array(lowest, scores.dtype()));
  } else if (m) {
    // The mask must OVERRIDE masked columns, not add: a fully-masked left-pad
    // query row yields NaN attention output, which poisons that position's K/V
    // in later layers, and NaN + (-inf) stays NaN (the fused dense kernel
    // handles this internally; this path must do it explicitly). Real columns
    // are unchanged (mask 0), and exp(lowest - max) underflows to exactly 0 in
    // the precise softmax, so real rows match the plain additive form.
    const float lowest = static_cast<float>(mx::finfo(scores.dtype()).min);
    scores = mx::where(mx::isneginf(*m), mx::array(lowest, scores.dtype()),
                       mx::add(scores, *m));
  }

  scores = mx::softmax(scores, /*axis=*/-1, /*precise=*/true);
  mx::array out =
      mx::quantized_matmul(scores, v.w, v.scales, v.biases, /*transpose=*/false, group_size, bits);

  if (n_repeats > 1) {
    out = mx::reshape(out, {B, n_q_heads, L, out.shape().back()});
  }
  return out;
}

mx::array sdpa_with_cache(const mx::array& q, const mx::array& k, const mx::array& v,
                          KVCache* cache, int layer, float scale, const std::string& mask_mode) {
  if (cache && cache->quantized()) {
    QuantizedKVSlice s = cache->update_and_fetch_quantized(layer, k, v);
    const KVQuantConfig& qc = cache->quant_config();
    return quantized_sdpa(q, s.k, s.v, scale, mask_mode, std::nullopt, qc.group_size, qc.bits);
  }
  if (cache) {
    auto kv = cache->update_and_fetch(layer, k, v);
    return mx::fast::scaled_dot_product_attention(q, kv.first, kv.second, scale, mask_mode);
  }
  return mx::fast::scaled_dot_product_attention(q, k, v, scale, mask_mode);
}

mx::array sdpa_with_cache(const mx::array& q, const mx::array& k, const mx::array& v,
                          BatchKVCache& cache, int layer, float scale, const mx::array& mask) {
  if (cache.quantized()) {
    QuantizedKVSlice s = cache.update_and_fetch_quantized(layer, k, v);
    const KVQuantConfig& qc = cache.quant_config();
    return quantized_sdpa(q, s.k, s.v, scale, /*mask_mode=*/"", mask, qc.group_size, qc.bits);
  }
  auto kv = cache.update_and_fetch(layer, k, v);
  return mx::fast::scaled_dot_product_attention(q, kv.first, kv.second, scale,
                                                /*mask_mode=*/"", mask);
}

}  // namespace mlxforge
