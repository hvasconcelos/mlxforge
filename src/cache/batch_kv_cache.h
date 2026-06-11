// Batched KV cache — the layout the continuous-batching server needs.
//
// Ported from mlx_lm/models/cache.py::BatchKVCache. Per layer the cache stores
// K/V as (B, n_kv_heads, S_cap, head_dim), contiguous and LEFT-padded, grown in
// blocks of `step` (256) via zeros + concatenate(axis=2). Two per-row arrays
// drive correct batched attention:
//   offset       initialized to -left_padding, += written_len each token sweep
//                (so each row's offset = its real-token count / RoPE position).
//   left_padding the number of pad tokens prepended to each row.
//
// All layers share idx/offset/left_padding (they process the same tokens):
// update_and_fetch(layer, k, v) writes one layer's slice; advance(n) bumps the
// shared bookkeeping once per token sweep. filter()/merge() do the batch-axis
// surgery (eviction/admission) the scheduler needs.
//
// With a KVQuantConfig (kv_bits > 0) each layer's K/V is instead the
// mx::quantize triplet (cache/kv_quant.h), quantized at write time exactly like
// mlx_lm's QuantizedKVCache. Every axis-0/2 operation (grow, write, filter,
// merge, pad) is last-axis-agnostic, so the same surgery runs per component;
// zero-filled pad regions dequantize to exactly 0 and are masked anyway.
#pragma once

#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "cache/kv_quant.h"

#include "mlx/array.h"

namespace mlxforge {

namespace mx = mlx::core;

struct KVBlock;  // cache/block_pool.h

class BatchKVCache {
 public:
  static constexpr int kStep = 256;

  // One cache for all layers; `left_padding[i]` is the pad count of batch row i.
  BatchKVCache(int n_layers, const std::vector<int>& left_padding, KVQuantConfig qcfg = {});

  // Build a batch-1 cache from one already-prefilled sequence's per-layer K/V
  // (each (1, n_kv_heads, seq, head_dim)). `decode_offset` seeds the per-row RoPE
  // position for the next (decode) token. For an interleaved-M-RoPE prompt
  // (Qwen3-VL) this is one past the prompt's max 3D position, which sits BELOW its
  // token count (the image collapses many tokens into few positions) — so offset
  // is passed explicitly rather than derived from seq, and the standard batched
  // decode (whose mask works in physical-slot space while RoPE uses offset) is
  // numerically correct unchanged. left_padding is 0 (a fresh single-row prefill
  // has no padding). Used to admit a vision prompt into the decode pool (merge()).
  // Always dense (the vision path is rejected when KV quantization is on).
  static BatchKVCache from_single_sequence(
      std::vector<std::pair<mx::array, mx::array>> kv_per_layer, int seq, int decode_offset);

  // Build a batch-1 cache whose first `len` positions are the given prefix-pool
  // blocks' K/V (consecutive from position 0; `len` may stop short of the last
  // block's end — the prompt's final token is always recomputed). The blocks
  // are written through the standard block-grow writer (update_kv_components)
  // so the buffer layout matches a normal prefill chunk's; RoPE offset = len,
  // no left padding. The suffix prefill then appends at idx == len.
  static BatchKVCache from_prefix(int n_layers,
                                  const std::vector<std::shared_ptr<const KVBlock>>& blocks,
                                  int len, KVQuantConfig qcfg = {});

  int batch_size() const { return batch_; }
  int idx() const { return idx_; }  // populated sequence length (_idx)
  // Allocated capacity along the sequence axis (0 before the first write).
  int s_cap() const;
  const mx::array& offset() const { return offset_; }              // (B,) int32
  const mx::array& left_padding() const { return left_padding_; }  // (B,) int32

  bool quantized() const { return qcfg_.enabled(); }
  const KVQuantConfig& quant_config() const { return qcfg_; }

  // Append layer `layer`'s K/V (each (B, n_kv_heads, L, head_dim)) at the
  // current write position, growing capacity in `step` blocks if needed, and
  // return the populated slice [..., :idx+L, :] to attend over. Dense caches
  // only (throws on a quantized cache).
  std::pair<mx::array, mx::array> update_and_fetch(int layer, const mx::array& k,
                                                   const mx::array& v);

  // Quantized counterpart: quantize the incoming K/V (mx::quantize, per
  // position), write each triplet component at the current position with the
  // same block growth, and return the populated triplets for quantized_sdpa.
  QuantizedKVSlice update_and_fetch_quantized(int layer, const mx::array& k, const mx::array& v);

  // Advance the shared offset/idx by `n_tokens` (call once after all layers).
  void advance(int n_tokens);

  // Materialize the whole cache (all layers' K/V + offset/left_padding). Called
  // at chunked-prefill boundaries to bound graph/memory growth.
  void eval_state();

  // Populated K/V slice [..., :idx, :] for a layer (for inspection/tests).
  // Dense caches only.
  std::pair<mx::array, mx::array> fetch(int layer) const;

  // Populated K/V slice of a quantized layer, dequantized back to fp16 (for
  // inspection/tests — the model attends over the triplets directly).
  std::pair<mx::array, mx::array> fetch_dequantized(int layer) const;

  // One row's populated K/V component vectors for a layer, as
  // (1, n_kv_heads, len, comp_dim) views over physical slots
  // [left_pad, left_pad + len). For harvesting a finished row into the prefix
  // pool (which materializes its own copies — these are lazy views).
  std::pair<std::vector<mx::array>, std::vector<mx::array>> fetch_row_components(
      int layer, int row, int left_pad, int len) const;

  // Host copy of the per-row left padding (small, already materialized).
  std::vector<int> left_padding_host() const;

  // Eviction: keep only the given batch rows (take on axis 0) across every
  // layer's K/V plus offset/left_padding, then shift off any common left
  // padding. `keep` indexes the current batch rows.
  void filter(const std::vector<int>& keep);

  // Admission: pad this cache and `other` to a common S_cap (right-justified by
  // write index) and concatenate on the batch axis. Used to admit a freshly
  // prefilled batch into the decode cache. Both caches must share a
  // KVQuantConfig.
  void merge(BatchKVCache& other);

  // Append `extra` masked dummy rows on the batch axis to reach a decode
  // bucket size. Dummy rows attend only to their own position (left_padding =
  // idx) so they never produce NaNs and — being independent batch rows — cannot
  // affect the real rows. They are trimmed back with filter() after the step.
  void pad_dummies(int extra);

  // Gated-DeltaNet recurrent state for hybrid models (Qwen3.5): the linear layers
  // carry a per-row conv buffer (B, K-1, conv_dim) and delta-rule state
  // (B, Hv, Dv, Dk) instead of a growing KV. Fixed-size on the sequence axis, so
  // filter/merge/pad_dummies only do batch-axis (axis 0) surgery on them. Lazily
  // set by the model on the first write; pure-attention models never touch them.
  bool has_linear_state(int layer) const { return conv_state_[layer].has_value(); }
  std::pair<mx::array, mx::array> linear_state(int layer) const {
    return {*conv_state_[layer], *recur_state_[layer]};
  }
  void set_linear_state(int layer, const mx::array& conv_state, const mx::array& recur_state) {
    conv_state_[layer] = conv_state;
    recur_state_[layer] = recur_state;
  }

 private:
  int batch_;
  int idx_ = 0;
  // Per layer: empty until written; 1 component dense, 3 quantized.
  std::vector<std::vector<mx::array>> keys_;
  std::vector<std::vector<mx::array>> values_;
  std::vector<std::optional<mx::array>> conv_state_;
  std::vector<std::optional<mx::array>> recur_state_;
  KVQuantConfig qcfg_;
  mx::array offset_;        // (B,)
  mx::array left_padding_;  // (B,)
};

}  // namespace mlxforge
