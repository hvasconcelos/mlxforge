#include "cache/batch_kv_cache.h"

#include <algorithm>
#include <utility>
#include <vector>

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

// Slice [start, stop) along the sequence axis (2), keeping axes 0/1/3 full.
mx::array slice_seq(const mx::array& a, int start, int stop) {
  const auto& s = a.shape();
  return mx::slice(a, {0, 0, start, 0}, {s[0], s[1], stop, s[3]});
}
}  // namespace

BatchKVCache::BatchKVCache(int n_layers, const std::vector<int>& left_padding)
    : batch_(static_cast<int>(left_padding.size())),
      keys_(n_layers),
      values_(n_layers),
      offset_(neg(left_padding)),
      left_padding_(mx::array(left_padding.data(), {static_cast<int>(left_padding.size())},
                              mx::int32)) {}

int BatchKVCache::s_cap() const {
  return keys_[0].has_value() ? keys_[0]->shape()[2] : 0;
}

std::pair<mx::array, mx::array> BatchKVCache::update_and_fetch(int layer, const mx::array& k,
                                                              const mx::array& v) {
  const int prev = idx_;
  const int L = k.shape()[2];
  const int end = prev + L;
  const int B = k.shape()[0];
  const int H = k.shape()[1];
  const int Dk = k.shape()[3];
  const int Dv = v.shape()[3];

  const int cap = keys_[layer].has_value() ? keys_[layer]->shape()[2] : 0;
  if (!keys_[layer].has_value() || end > cap) {
    const int n_steps = (kStep + L - 1) / kStep;
    const int add = n_steps * kStep;
    mx::array new_k = mx::zeros({B, H, add, Dk}, k.dtype());
    mx::array new_v = mx::zeros({B, H, add, Dv}, v.dtype());
    if (keys_[layer].has_value()) {
      mx::array kk = *keys_[layer];
      mx::array vv = *values_[layer];
      // Drop any unused tail of the last block before growing.
      if (prev % kStep != 0) {
        kk = mx::slice(kk, {0, 0, 0, 0}, {B, H, prev, Dk});
        vv = mx::slice(vv, {0, 0, 0, 0}, {B, H, prev, Dv});
      }
      keys_[layer] = mx::concatenate({kk, new_k}, /*axis=*/2);
      values_[layer] = mx::concatenate({vv, new_v}, /*axis=*/2);
    } else {
      keys_[layer] = new_k;
      values_[layer] = new_v;
    }
  }

  keys_[layer] = mx::slice_update(*keys_[layer], k, {0, 0, prev, 0}, {B, H, end, Dk});
  values_[layer] = mx::slice_update(*values_[layer], v, {0, 0, prev, 0}, {B, H, end, Dv});
  return {mx::slice(*keys_[layer], {0, 0, 0, 0}, {B, H, end, Dk}),
          mx::slice(*values_[layer], {0, 0, 0, 0}, {B, H, end, Dv})};
}

std::pair<mx::array, mx::array> BatchKVCache::fetch(int layer) const {
  return {slice_seq(*keys_[layer], 0, idx_), slice_seq(*values_[layer], 0, idx_)};
}

void BatchKVCache::advance(int n_tokens) {
  idx_ += n_tokens;
  offset_ = mx::add(offset_, mx::array(n_tokens, mx::int32));
  mx::eval(offset_);  // keep the per-row bookkeeping materialized
}

void BatchKVCache::pad_dummies(int extra) {
  if (extra <= 0) return;
  for (size_t l = 0; l < keys_.size(); ++l) {
    if (!keys_[l].has_value()) continue;
    const auto& k = *keys_[l];
    const auto& v = *values_[l];
    mx::array dk = mx::zeros({extra, k.shape()[1], k.shape()[2], k.shape()[3]}, k.dtype());
    mx::array dv = mx::zeros({extra, v.shape()[1], v.shape()[2], v.shape()[3]}, v.dtype());
    keys_[l] = mx::concatenate({k, dk}, /*axis=*/0);
    values_[l] = mx::concatenate({v, dv}, /*axis=*/0);
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
    if (k.has_value()) state.push_back(*k);
  for (auto& v : values_)
    if (v.has_value()) state.push_back(*v);
  mx::eval(state);
}

namespace {
int scalar_int(const mx::array& a) {
  mx::array e = mx::astype(a, mx::int32);
  mx::eval(e);
  return e.item<int>();
}
}  // namespace

void BatchKVCache::filter(const std::vector<int>& keep) {
  mx::array idxs(keep.data(), {static_cast<int>(keep.size())}, mx::int32);
  for (int l = 0; l < static_cast<int>(keys_.size()); ++l) {
    if (keys_[l].has_value()) {
      keys_[l] = mx::take(*keys_[l], idxs, /*axis=*/0);
      values_[l] = mx::take(*values_[l], idxs, /*axis=*/0);
    }
  }
  offset_ = mx::take(offset_, idxs, /*axis=*/0);
  left_padding_ = mx::take(left_padding_, idxs, /*axis=*/0);
  batch_ = static_cast<int>(keep.size());

  // Shift left to drop any padding common to all surviving rows.
  const int min_left_pad = scalar_int(mx::min(left_padding_, /*keepdims=*/false));
  if (min_left_pad > 0) {
    for (int l = 0; l < static_cast<int>(keys_.size()); ++l) {
      if (keys_[l].has_value()) {
        keys_[l] = slice_seq(*keys_[l], min_left_pad, keys_[l]->shape()[2]);
        values_[l] = slice_seq(*values_[l], min_left_pad, values_[l]->shape()[2]);
      }
    }
    idx_ -= min_left_pad;
    left_padding_ = mx::subtract(left_padding_, mx::array(min_left_pad, mx::int32));
  }
  mx::eval(offset_, left_padding_);
}

void BatchKVCache::merge(BatchKVCache& other) {
  if (other.batch_ == 0) return;
  if (batch_ == 0) {
    keys_ = other.keys_;
    values_ = other.values_;
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
  // max_size on the sequence axis. Returns the padded K and V.
  auto pad_layer = [&](BatchKVCache& c, int l) -> std::pair<mx::array, mx::array> {
    mx::array k = *c.keys_[l];
    mx::array v = *c.values_[l];
    const int len = k.shape()[2];
    const int left = max_idx - c.idx_;
    int right = max_size - len - left;
    if (right < 0) {  // trim the unused tail
      k = slice_seq(k, 0, len + right);
      v = slice_seq(v, 0, len + right);
      right = 0;
    }
    if (left != 0 || right != 0) {
      std::vector<std::pair<int, int>> pw = {{0, 0}, {0, 0}, {left, right}, {0, 0}};
      k = mx::pad(k, pw);
      v = mx::pad(v, pw);
    }
    return {k, v};
  };

  for (int l = 0; l < static_cast<int>(keys_.size()); ++l) {
    auto a = pad_layer(*this, l);
    auto b = pad_layer(other, l);
    keys_[l] = mx::concatenate({a.first, b.first}, /*axis=*/0);
    values_[l] = mx::concatenate({a.second, b.second}, /*axis=*/0);
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
