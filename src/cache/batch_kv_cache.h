// XLLM-010: batched KV cache — the layout the continuous-batching server needs.
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
// shared bookkeeping once per token sweep. filter()/merge() (XLLM-011) do the
// batch-axis surgery the scheduler needs.
#pragma once

#include <optional>
#include <utility>
#include <vector>

#include "mlx/array.h"

namespace xllm {

namespace mx = mlx::core;

class BatchKVCache {
 public:
  static constexpr int kStep = 256;

  // One cache for all layers; `left_padding[i]` is the pad count of batch row i.
  BatchKVCache(int n_layers, const std::vector<int>& left_padding);

  int batch_size() const { return batch_; }
  int idx() const { return idx_; }  // populated sequence length (_idx)
  // Allocated capacity along the sequence axis (0 before the first write).
  int s_cap() const;
  const mx::array& offset() const { return offset_; }              // (B,) int32
  const mx::array& left_padding() const { return left_padding_; }  // (B,) int32

  // Append layer `layer`'s K/V (each (B, n_kv_heads, L, head_dim)) at the
  // current write position, growing capacity in `step` blocks if needed, and
  // return the populated slice [..., :idx+L, :] to attend over.
  std::pair<mx::array, mx::array> update_and_fetch(int layer, const mx::array& k,
                                                   const mx::array& v);

  // Advance the shared offset/idx by `n_tokens` (call once after all layers).
  void advance(int n_tokens);

  // Materialize the whole cache (all layers' K/V + offset/left_padding). Called
  // at chunked-prefill boundaries to bound graph/memory growth.
  void eval_state();

  // Populated K/V slice [..., :idx, :] for a layer (for inspection/tests).
  std::pair<mx::array, mx::array> fetch(int layer) const;

  // Eviction: keep only the given batch rows (take on axis 0) across every
  // layer's K/V plus offset/left_padding, then shift off any common left
  // padding. `keep` indexes the current batch rows.
  void filter(const std::vector<int>& keep);

  // Admission: pad this cache and `other` to a common S_cap (right-justified by
  // write index) and concatenate on the batch axis. Used to admit a freshly
  // prefilled batch into the decode cache.
  void merge(BatchKVCache& other);

  // XLLM-019: append `extra` masked dummy rows on the batch axis to reach a
  // decode bucket. Dummy rows attend only to their own position (left_padding =
  // idx) so they never produce NaNs and — being independent batch rows — cannot
  // affect the real rows. They are trimmed back with filter() after the step.
  void pad_dummies(int extra);

 private:
  int batch_;
  int idx_ = 0;
  std::vector<std::optional<mx::array>> keys_;
  std::vector<std::optional<mx::array>> values_;
  mx::array offset_;        // (B,)
  mx::array left_padding_;  // (B,)
};

}  // namespace xllm
