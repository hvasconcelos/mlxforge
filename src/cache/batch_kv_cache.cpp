#include "cache/batch_kv_cache.h"

#include <algorithm>
#include <stdexcept>
#include <utility>
#include <vector>

#include "cache/block_pool.h"

#include "mlx/ops.h"
#include "mlx/transforms.h"

namespace mlxforge {

namespace {
// offset starts at -left_padding (per row).
mx::array neg(const std::vector<int>& v) {
  std::vector<int> out(v.size());
  for (size_t i = 0; i < v.size(); ++i) out[i] = -v[i];
  return mx::array(out.data(), {static_cast<int>(out.size())}, mx::int32);
}
}  // namespace

BatchKVCache::BatchKVCache(int n_layers, const std::vector<int>& left_padding, KVQuantConfig qcfg)
    : batch_(static_cast<int>(left_padding.size())),
      keys_(n_layers),
      values_(n_layers),
      conv_state_(n_layers),
      recur_state_(n_layers),
      qcfg_(qcfg),
      offset_(neg(left_padding)),
      left_padding_(mx::array(left_padding.data(), {static_cast<int>(left_padding.size())},
                              mx::int32)) {}

BatchKVCache BatchKVCache::from_single_sequence(
    std::vector<std::pair<mx::array, mx::array>> kv_per_layer, int seq, int decode_offset) {
  const int n_layers = static_cast<int>(kv_per_layer.size());
  BatchKVCache c(n_layers, std::vector<int>{0});  // batch 1, no left padding
  for (int l = 0; l < n_layers; ++l) {
    c.keys_[l] = {std::move(kv_per_layer[l].first)};
    c.values_[l] = {std::move(kv_per_layer[l].second)};
  }
  c.idx_ = seq;  // physical sequence length (drives the attention mask)
  int off = decode_offset;
  c.offset_ = mx::array(&off, {1}, mx::int32);  // decoupled RoPE position
  mx::eval(c.offset_);
  return c;
}

BatchKVCache BatchKVCache::from_prefix(int n_layers,
                                       const std::vector<std::shared_ptr<const KVBlock>>& blocks,
                                       int len, KVQuantConfig qcfg) {
  BatchKVCache c(n_layers, std::vector<int>{0}, qcfg);  // batch 1, no left padding
  if (len <= 0 || blocks.empty()) return c;

  // Per layer, stitch the blocks into one [0, len) span and write it through
  // the standard writer (one chunk at position 0) so capacity rounding and
  // buffer layout match a normal prefill. `len` may stop short of the last
  // block's end (the prompt's final token is always recomputed).
  for (int l = 0; l < n_layers; ++l) {
    std::vector<mx::array> k_in, v_in;
    const std::size_t n_comp = blocks[0]->k[l].size();
    for (std::size_t i = 0; i < n_comp; ++i) {
      std::vector<mx::array> kp, vp;
      kp.reserve(blocks.size());
      vp.reserve(blocks.size());
      for (const auto& b : blocks) {
        kp.push_back(b->k[l][i]);
        vp.push_back(b->v[l][i]);
      }
      mx::array kj = kp.size() == 1 ? kp[0] : mx::concatenate(kp, /*axis=*/2);
      mx::array vj = vp.size() == 1 ? vp[0] : mx::concatenate(vp, /*axis=*/2);
      k_in.push_back(slice_seq(kj, 0, len));
      v_in.push_back(slice_seq(vj, 0, len));
    }
    update_kv_components(c.keys_[l], k_in, /*prev=*/0, kStep);
    update_kv_components(c.values_[l], v_in, /*prev=*/0, kStep);
  }
  c.idx_ = len;
  c.offset_ = mx::array(&len, {1}, mx::int32);
  mx::eval(c.offset_);
  return c;
}

int BatchKVCache::s_cap() const {
  return keys_[0].empty() ? 0 : keys_[0][0].shape()[2];
}

std::pair<mx::array, mx::array> BatchKVCache::update_and_fetch(int layer, const mx::array& k,
                                                              const mx::array& v) {
  if (quantized())
    throw std::logic_error("BatchKVCache::update_and_fetch: cache is quantized");
  std::vector<mx::array> ks = update_kv_components(keys_[layer], {k}, idx_, kStep);
  std::vector<mx::array> vs = update_kv_components(values_[layer], {v}, idx_, kStep);
  return {ks[0], vs[0]};
}

QuantizedKVSlice BatchKVCache::update_and_fetch_quantized(int layer, const mx::array& k,
                                                          const mx::array& v) {
  if (!quantized())
    throw std::logic_error("BatchKVCache::update_and_fetch_quantized: cache is dense");
  QuantizedKV ks = quantize_and_update(keys_[layer], k, qcfg_.group_size, qcfg_.bits, idx_, kStep);
  QuantizedKV vs =
      quantize_and_update(values_[layer], v, qcfg_.group_size, qcfg_.bits, idx_, kStep);
  return {ks, vs};
}

std::pair<mx::array, mx::array> BatchKVCache::fetch(int layer) const {
  if (quantized()) throw std::logic_error("BatchKVCache::fetch: cache is quantized");
  return {slice_seq(keys_[layer][0], 0, idx_), slice_seq(values_[layer][0], 0, idx_)};
}

std::pair<mx::array, mx::array> BatchKVCache::fetch_dequantized(int layer) const {
  if (!quantized())
    throw std::logic_error("BatchKVCache::fetch_dequantized: cache is dense");
  auto deq = [&](const std::vector<mx::array>& t) {
    return mx::dequantize(slice_seq(t[0], 0, idx_), slice_seq(t[1], 0, idx_),
                          slice_seq(t[2], 0, idx_), qcfg_.group_size, qcfg_.bits);
  };
  return {deq(keys_[layer]), deq(values_[layer])};
}

void BatchKVCache::advance(int n_tokens) {
  idx_ += n_tokens;
  offset_ = mx::add(offset_, mx::array(n_tokens, mx::int32));
  mx::eval(offset_);  // keep the per-row bookkeeping materialized
}

void BatchKVCache::pad_dummies(int extra) {
  if (extra <= 0) return;
  auto pad_rows = [&](std::vector<mx::array>& comps) {
    for (auto& c : comps) {
      mx::array d = mx::zeros({extra, c.shape()[1], c.shape()[2], c.shape()[3]}, c.dtype());
      c = mx::concatenate({c, d}, /*axis=*/0);
    }
  };
  for (size_t l = 0; l < keys_.size(); ++l) {
    if (keys_[l].empty()) continue;
    pad_rows(keys_[l]);
    pad_rows(values_[l]);
  }
  // Linear-attention layers: append `extra` zero state rows (fixed-size axes).
  for (size_t l = 0; l < conv_state_.size(); ++l) {
    if (!conv_state_[l].has_value()) continue;
    const auto& c = *conv_state_[l];
    const auto& r = *recur_state_[l];
    mx::array dc = mx::zeros({extra, c.shape()[1], c.shape()[2]}, c.dtype());
    mx::array dr = mx::zeros({extra, r.shape()[1], r.shape()[2], r.shape()[3]}, r.dtype());
    conv_state_[l] = mx::concatenate({c, dc}, /*axis=*/0);
    recur_state_[l] = mx::concatenate({r, dr}, /*axis=*/0);
  }
  std::vector<int> doff(extra, 0);
  std::vector<int> dlp(extra, idx_);  // attend only to own position -> no NaN
  offset_ = mx::concatenate({offset_, mx::array(doff.data(), {extra}, mx::int32)}, 0);
  left_padding_ = mx::concatenate({left_padding_, mx::array(dlp.data(), {extra}, mx::int32)}, 0);
  batch_ += extra;
  mx::eval(offset_, left_padding_);
}

void BatchKVCache::eval_state() {
  std::vector<mx::array> state = {offset_, left_padding_};
  for (auto& k : keys_)
    for (auto& c : k) state.push_back(c);
  for (auto& v : values_)
    for (auto& c : v) state.push_back(c);
  for (auto& c : conv_state_)
    if (c.has_value()) state.push_back(*c);
  for (auto& r : recur_state_)
    if (r.has_value()) state.push_back(*r);
  mx::eval(state);
}

namespace {
int scalar_int(const mx::array& a) {
  mx::array e = mx::astype(a, mx::int32);
  mx::eval(e);
  return e.item<int>();
}
}  // namespace

std::pair<std::vector<mx::array>, std::vector<mx::array>> BatchKVCache::fetch_row_components(
    int layer, int row, int left_pad, int len) const {
  auto row_slice = [&](const mx::array& c) {
    const auto& s = c.shape();
    return mx::slice(c, {row, 0, left_pad, 0}, {row + 1, s[1], left_pad + len, s[3]});
  };
  std::vector<mx::array> k, v;
  for (const auto& c : keys_[layer]) k.push_back(row_slice(c));
  for (const auto& c : values_[layer]) v.push_back(row_slice(c));
  return {std::move(k), std::move(v)};
}

std::vector<int> BatchKVCache::left_padding_host() const {
  mx::array c = mx::contiguous(left_padding_);
  mx::eval(c);
  return std::vector<int>(c.data<int32_t>(), c.data<int32_t>() + c.size());
}

void BatchKVCache::filter(const std::vector<int>& keep) {
  mx::array idxs(keep.data(), {static_cast<int>(keep.size())}, mx::int32);
  for (int l = 0; l < static_cast<int>(keys_.size()); ++l) {
    for (auto& c : keys_[l]) c = mx::take(c, idxs, /*axis=*/0);
    for (auto& c : values_[l]) c = mx::take(c, idxs, /*axis=*/0);
    // Linear state is fixed-size; eviction is a plain batch-axis gather.
    if (conv_state_[l].has_value()) {
      conv_state_[l] = mx::take(*conv_state_[l], idxs, /*axis=*/0);
      recur_state_[l] = mx::take(*recur_state_[l], idxs, /*axis=*/0);
    }
  }
  offset_ = mx::take(offset_, idxs, /*axis=*/0);
  left_padding_ = mx::take(left_padding_, idxs, /*axis=*/0);
  batch_ = static_cast<int>(keep.size());

  // Shift left to drop any padding common to all surviving rows.
  const int min_left_pad = scalar_int(mx::min(left_padding_, /*keepdims=*/false));
  if (min_left_pad > 0) {
    for (int l = 0; l < static_cast<int>(keys_.size()); ++l) {
      for (auto& c : keys_[l]) c = slice_seq(c, min_left_pad, c.shape()[2]);
      for (auto& c : values_[l]) c = slice_seq(c, min_left_pad, c.shape()[2]);
    }
    idx_ -= min_left_pad;
    left_padding_ = mx::subtract(left_padding_, mx::array(min_left_pad, mx::int32));
  }
  mx::eval(offset_, left_padding_);
}

void BatchKVCache::merge(BatchKVCache& other) {
  if (qcfg_ != other.qcfg_)
    throw std::logic_error("BatchKVCache::merge: KV quantization configs differ");
  if (other.batch_ == 0) return;
  if (batch_ == 0) {
    keys_ = other.keys_;
    values_ = other.values_;
    conv_state_ = other.conv_state_;
    recur_state_ = other.recur_state_;
    offset_ = other.offset_;
    left_padding_ = other.left_padding_;
    idx_ = other.idx_;
    batch_ = other.batch_;
    return;
  }

  const int max_idx = std::max(idx_, other.idx_);
  const int l1 = s_cap();
  const int l2 = other.s_cap();
  const int max_size = std::max(l1, l2);

  // Pad one cache's layer `l` so it is right-justified at max_idx and sized
  // max_size on the sequence axis. Returns the padded components.
  auto pad_layer = [&](BatchKVCache& c, std::vector<mx::array>& comps) {
    std::vector<mx::array> out;
    out.reserve(comps.size());
    for (const auto& a : comps) {
      mx::array x = a;
      const int len = x.shape()[2];
      const int left = max_idx - c.idx_;
      int right = max_size - len - left;
      if (right < 0) {  // trim the unused tail
        x = slice_seq(x, 0, len + right);
        right = 0;
      }
      if (left != 0 || right != 0) {
        std::vector<std::pair<int, int>> pw = {{0, 0}, {0, 0}, {left, right}, {0, 0}};
        x = mx::pad(x, pw);
      }
      out.push_back(std::move(x));
    }
    return out;
  };

  for (int l = 0; l < static_cast<int>(keys_.size()); ++l) {
    std::vector<mx::array> ka = pad_layer(*this, keys_[l]);
    std::vector<mx::array> kb = pad_layer(other, other.keys_[l]);
    std::vector<mx::array> va = pad_layer(*this, values_[l]);
    std::vector<mx::array> vb = pad_layer(other, other.values_[l]);
    keys_[l].clear();
    values_[l].clear();
    for (size_t i = 0; i < ka.size(); ++i) {
      keys_[l].push_back(mx::concatenate({ka[i], kb[i]}, /*axis=*/0));
      values_[l].push_back(mx::concatenate({va[i], vb[i]}, /*axis=*/0));
    }
    // Linear state: fixed-size, so admission is a plain batch-axis concatenate
    // (no sequence-length right-justification — the recurrent state already
    // summarizes each row's history regardless of length).
    if (conv_state_[l].has_value() && other.conv_state_[l].has_value()) {
      conv_state_[l] = mx::concatenate({*conv_state_[l], *other.conv_state_[l]}, /*axis=*/0);
      recur_state_[l] = mx::concatenate({*recur_state_[l], *other.recur_state_[l]}, /*axis=*/0);
    }
  }

  // left_padding grows by the left-pad each side received; offset is unchanged.
  mx::array lp_a = mx::add(left_padding_, mx::array(max_idx - idx_, mx::int32));
  mx::array lp_b = mx::add(other.left_padding_, mx::array(max_idx - other.idx_, mx::int32));
  left_padding_ = mx::concatenate({lp_a, lp_b}, /*axis=*/0);
  offset_ = mx::concatenate({offset_, other.offset_}, /*axis=*/0);
  idx_ = max_idx;
  batch_ += other.batch_;
  mx::eval(offset_, left_padding_);
}

}  // namespace mlxforge
