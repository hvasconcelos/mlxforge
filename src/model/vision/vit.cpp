#include "model/vision/vit.h"

#include <cmath>
#include <tuple>
#include <vector>

#include "mlx/fast.h"
#include "mlx/ops.h"
#include "mlx/transforms.h"

namespace mlxforge {

namespace {
// Read grid_thw (num_images, 3) to the host as flat ints. Vision preprocessing
// runs once at prefill, so the small host round-trip is fine.
std::vector<int> host_grid(const mx::array& grid_thw) {
  mx::array g = mx::contiguous(mx::astype(grid_thw, mx::int32));
  mx::eval(g);
  const int32_t* p = g.data<int32_t>();
  return std::vector<int>(p, p + g.size());
}
}  // namespace

VitEncoder::VitEncoder(VisionConfig cfg, const Weights& weights)
    : cfg_(std::move(cfg)), w_(weights) {}

mx::array VitEncoder::linear(const mx::array& x, const std::string& key) const {
  // ViT weights are unquantized fp16; weight is (out, in), bias is (out,).
  mx::array y = mx::matmul(x, mx::transpose(w_.at(key + ".weight")));
  return mx::add(y, w_.at(key + ".bias"));
}

mx::array VitEncoder::patch_embed(const mx::array& pixel_values) const {
  // The Conv3d kernel spans a whole patch (kernel == stride == patch extent), so
  // it reduces to a linear projection of the flattened patch. The MLX-format
  // weight is (vit_hidden, temporal, patch, patch, in_ch), whereas the processor
  // lays each pixel_values row out as (in_ch, temporal, patch, patch); reorder the
  // patch to (temporal, patch, patch, in_ch) — the reference's moveaxis — so the
  // flattened axes line up, then matmul reproduces the convolution.
  const int P = pixel_values.shape()[0];  // num patches
  const int c = cfg_.in_channels, t = cfg_.temporal_patch_size, k = cfg_.patch_size;
  mx::array x = mx::reshape(pixel_values, {P, c, t, k, k});
  x = mx::transpose(x, {0, 2, 3, 4, 1});  // (P, temporal, patch, patch, in_ch)
  x = mx::reshape(x, {P, -1});

  const mx::array& w = w_.at("visual.patch_embed.proj.weight");
  mx::array w2 = mx::reshape(w, {cfg_.hidden, -1});  // (vit_hidden, patch_flat)
  x = mx::astype(x, w2.dtype());  // compute in the weight's dtype (fp16)
  mx::array out = mx::matmul(x, mx::transpose(w2));  // (num_patches, vit_hidden)
  return mx::add(out, w_.at("visual.patch_embed.proj.bias"));
}

mx::array VitEncoder::rope_2d_freqs(const mx::array& grid_thw) const {
  const std::vector<int> g = host_grid(grid_thw);
  const int num_images = static_cast<int>(g.size()) / 3;
  const int merge = cfg_.spatial_merge_size;

  // Per patch (in merged-block order), its full-resolution (row, col) position.
  std::vector<float> rows, cols;
  for (int im = 0; im < num_images; ++im) {
    const int t = g[im * 3 + 0], h = g[im * 3 + 1], w = g[im * 3 + 2];
    const int mh = h / merge, mw = w / merge;
    for (int f = 0; f < t; ++f)            // temporal frames reuse the same grid
      for (int br = 0; br < mh; ++br)
        for (int bc = 0; bc < mw; ++bc)
          for (int ir = 0; ir < merge; ++ir)
            for (int ic = 0; ic < merge; ++ic) {
              rows.push_back(static_cast<float>(br * merge + ir));
              cols.push_back(static_cast<float>(bc * merge + ic));
            }
  }
  const int tokens = static_cast<int>(rows.size());

  // inv_freq[i] = 10000 ** -(2i / dim), dim = head_dim/2 (the vision RoPE dim).
  const int dim = cfg_.head_dim() / 2;
  const int half = dim / 2;  // frequency bands per axis
  mx::array idx = mx::arange(0, dim, 2, mx::float32);  // (half,)
  mx::array inv_freq = mx::reciprocal(
      mx::power(mx::array(10000.0f), mx::divide(idx, mx::array(static_cast<float>(dim)))));
  inv_freq = mx::reshape(inv_freq, {1, half});

  mx::array rowa(rows.data(), {tokens, 1}, mx::float32);
  mx::array cola(cols.data(), {tokens, 1}, mx::float32);
  // freqs[token] = concat(row * inv_freq, col * inv_freq) -> (tokens, head_dim/2)
  return mx::concatenate({mx::multiply(rowa, inv_freq), mx::multiply(cola, inv_freq)}, 1);
}

mx::array VitEncoder::pos_embed(const mx::array& grid_thw) const {
  const std::vector<int> g = host_grid(grid_thw);
  const int num_images = static_cast<int>(g.size()) / 3;
  const int merge = cfg_.spatial_merge_size;
  const int side = static_cast<int>(std::lround(std::sqrt(
      static_cast<double>(cfg_.num_position_embeddings))));  // square learned grid edge
  const int hidden = cfg_.hidden;
  const mx::array& table = w_.at("visual.pos_embed.weight");  // (side*side, hidden) fp16

  // Bilinear-resample weights for one axis of length n onto [0, side-1].
  auto axis = [&](int n) {
    std::vector<int> fl(n), cl(n);
    std::vector<float> fr(n);
    for (int i = 0; i < n; ++i) {
      const double v = (n == 1) ? 0.0 : static_cast<double>(side - 1) * i / (n - 1);
      const int f = static_cast<int>(v);
      fl[i] = f;
      cl[i] = std::min(f + 1, side - 1);
      fr[i] = static_cast<float>(v - f);
    }
    return std::make_tuple(fl, cl, fr);
  };

  std::vector<mx::array> per_image;
  for (int im = 0; im < num_images; ++im) {
    const int t = g[im * 3 + 0], h = g[im * 3 + 1], w = g[im * 3 + 2];
    auto [hf, hc, hfr] = axis(h);
    auto [wf, wc, wfr] = axis(w);

    const int hw = h * w;
    std::vector<int> i0(hw), i1(hw), i2(hw), i3(hw);
    std::vector<float> a0(hw), a1(hw), a2(hw), a3(hw);
    for (int r = 0; r < h; ++r)
      for (int c = 0; c < w; ++c) {
        const int p = r * w + c;  // (h, w) row-major
        i0[p] = hf[r] * side + wf[c];  a0[p] = (1 - hfr[r]) * (1 - wfr[c]);
        i1[p] = hf[r] * side + wc[c];  a1[p] = (1 - hfr[r]) * wfr[c];
        i2[p] = hc[r] * side + wf[c];  a2[p] = hfr[r] * (1 - wfr[c]);
        i3[p] = hc[r] * side + wc[c];  a3[p] = hfr[r] * wfr[c];
      }
    auto corner = [&](std::vector<int>& idx, std::vector<float>& wt) {
      mx::array g2 = mx::take(table, mx::array(idx.data(), {hw}, mx::int32), 0);  // (hw, hidden)
      mx::array wa(wt.data(), {hw, 1}, mx::float32);
      return mx::multiply(mx::astype(g2, mx::float32), wa);
    };
    mx::array pe = mx::add(mx::add(corner(i0, a0), corner(i1, a1)),
                           mx::add(corner(i2, a2), corner(i3, a3)));  // (hw, hidden), (h,w) order

    if (t > 1) pe = mx::tile(pe, {t, 1});  // temporal frames share the grid
    // Permute (t, h, w) row-major -> merged-block order to match the patch stream.
    pe = mx::reshape(pe, {t, h / merge, merge, w / merge, merge, hidden});
    pe = mx::transpose(pe, {0, 1, 3, 2, 4, 5});
    per_image.push_back(mx::reshape(pe, {t * hw, hidden}));
  }

  mx::array out = num_images == 1 ? per_image[0] : mx::concatenate(per_image, 0);
  return mx::astype(out, table.dtype());  // back to the table's fp16
}

std::string VitEncoder::block_key(int i) const {
  return "visual.blocks." + std::to_string(i);
}

mx::array VitEncoder::layer_norm(const mx::array& x, const std::string& prefix) const {
  return mx::fast::layer_norm(x, w_.at(prefix + ".weight"), w_.at(prefix + ".bias"), 1e-6f);
}

namespace {
// Tanh-approximation GELU (nn.GELU(approx="tanh")), the ViT MLP activation:
// 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 x^3))).
mx::array gelu_tanh(const mx::array& x) {
  const float c = 0.7978845608028654f;  // sqrt(2/pi)
  mx::array x3 = mx::multiply(mx::multiply(x, x), x);
  mx::array inner = mx::multiply(mx::array(c), mx::add(x, mx::multiply(mx::array(0.044715f), x3)));
  return mx::multiply(mx::multiply(mx::array(0.5f), x), mx::add(mx::array(1.0f), mx::tanh(inner)));
}

// NeoX-style half rotation: concat(-x[..., d/2:], x[..., :d/2]).
mx::array rotate_half(const mx::array& x) {
  const int d = x.shape().back();
  mx::array x1 = mx::slice(x, {0, 0, 0}, {x.shape()[0], x.shape()[1], d / 2});
  mx::array x2 = mx::slice(x, {0, 0, d / 2}, {x.shape()[0], x.shape()[1], d});
  return mx::concatenate({mx::negative(x2), x1}, -1);
}
}  // namespace

mx::array VitEncoder::attention(const mx::array& x, int i, const mx::array& freqs) const {
  const int seq = x.shape()[0];
  const int heads = cfg_.num_heads;
  const int hd = cfg_.head_dim();
  const float scale = 1.0f / std::sqrt(static_cast<float>(hd));

  // Fused QKV (with bias): (seq, 3*hidden) -> (seq, 3, heads, hd) -> q/k/v.
  mx::array qkv = mx::reshape(linear(x, block_key(i) + ".attn.qkv"), {seq, 3, heads, hd});
  auto part = [&](int idx) {
    return mx::reshape(mx::slice(qkv, {0, idx, 0, 0}, {seq, idx + 1, heads, hd}), {seq, heads, hd});
  };
  mx::array q = part(0), k = part(1), v = part(2);  // (seq, heads, hd)

  // 2D RoPE on Q/K: cos/sin from the precomputed (seq, hd/2) angle table, tiled
  // to hd, broadcast over heads. Computed in fp32, cast back to the input dtype.
  mx::array cos = mx::cos(freqs), sin = mx::sin(freqs);  // (seq, hd/2)
  cos = mx::reshape(mx::concatenate({cos, cos}, -1), {seq, 1, hd});
  sin = mx::reshape(mx::concatenate({sin, sin}, -1), {seq, 1, hd});
  auto rope = [&](const mx::array& t) {
    mx::array tf = mx::astype(t, mx::float32);
    mx::array r = mx::add(mx::multiply(tf, cos), mx::multiply(rotate_half(tf), sin));
    return mx::astype(r, t.dtype());
  };
  q = rope(q);
  k = rope(k);

  // Full attention over all patches of the (single) image: (seq,heads,hd) ->
  // (1,heads,seq,hd) -> SDPA (no mask) -> (seq, hidden).
  auto to_bhsd = [&](const mx::array& t) {
    return mx::transpose(mx::reshape(t, {1, seq, heads, hd}), {0, 2, 1, 3});
  };
  mx::array out = mx::fast::scaled_dot_product_attention(to_bhsd(q), to_bhsd(k), to_bhsd(v),
                                                         scale, /*mask_mode=*/"");
  out = mx::reshape(mx::transpose(out, {0, 2, 1, 3}), {seq, heads * hd});
  return linear(out, block_key(i) + ".attn.proj");
}

mx::array VitEncoder::vision_mlp(const mx::array& x, int i) const {
  mx::array h = linear(x, block_key(i) + ".mlp.linear_fc1");
  return linear(gelu_tanh(h), block_key(i) + ".mlp.linear_fc2");
}

mx::array VitEncoder::block(const mx::array& x, int i, const mx::array& freqs) const {
  mx::array h = mx::add(x, attention(layer_norm(x, block_key(i) + ".norm1"), i, freqs));
  return mx::add(h, vision_mlp(layer_norm(h, block_key(i) + ".norm2"), i));
}

}  // namespace mlxforge
