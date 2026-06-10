#include "cache/kv_quant.h"

#include <utility>

#include "mlx/ops.h"

namespace mlxforge {

mx::array slice_seq(const mx::array& a, int start, int stop) {
  const auto& s = a.shape();
  return mx::slice(a, {0, 0, start, 0}, {s[0], s[1], stop, s[3]});
}

std::vector<mx::array> update_kv_components(std::vector<mx::array>& store,
                                            const std::vector<mx::array>& in, int prev,
                                            int step) {
  const int L = in[0].shape()[2];
  const int end = prev + L;

  const int cap = store.empty() ? 0 : store[0].shape()[2];
  if (store.empty() || end > cap) {
    const int n_steps = (step + L - 1) / step;
    const int add = n_steps * step;
    std::vector<mx::array> grown;
    grown.reserve(in.size());
    for (size_t i = 0; i < in.size(); ++i) {
      const auto& s = in[i].shape();
      mx::array fresh = mx::zeros({s[0], s[1], add, s[3]}, in[i].dtype());
      if (store.empty()) {
        grown.push_back(std::move(fresh));
      } else {
        mx::array cur = store[i];
        // Drop any unused tail of the last block before growing.
        if (prev % step != 0) cur = slice_seq(cur, 0, prev);
        grown.push_back(mx::concatenate({cur, fresh}, /*axis=*/2));
      }
    }
    store = std::move(grown);
  }

  std::vector<mx::array> out;
  out.reserve(in.size());
  for (size_t i = 0; i < in.size(); ++i) {
    const auto& s = store[i].shape();
    store[i] = mx::slice_update(store[i], in[i], {0, 0, prev, 0}, {s[0], s[1], end, s[3]});
    out.push_back(slice_seq(store[i], 0, end));
  }
  return out;
}

QuantizedKV quantize_and_update(std::vector<mx::array>& store, const mx::array& x, int group_size,
                                int bits, int pos, int step) {
  std::vector<mx::array> t = update_kv_components(store, mx::quantize(x, group_size, bits), pos,
                                                  step);
  return {t[0], t[1], t[2]};
}

}  // namespace mlxforge
