#include "model/qwen3_vl.h"

#include <algorithm>
#include <cmath>

#include "mlx/fast.h"
#include "mlx/ops.h"

namespace mlxforge {

namespace {
// inv_freq[i] = rope_theta ** -(2i / head_dim), i in [0, head_dim/2).
mx::array build_inv_freq(const ModelConfig& cfg) {
  mx::array idx = mx::arange(0, cfg.head_dim, 2, mx::float32);
  return mx::reciprocal(
      mx::power(mx::array(cfg.rope_theta), mx::divide(idx, mx::array(static_cast<float>(cfg.head_dim)))));
}

// Interleaved M-RoPE axis selector over head_dim/2 frequencies: temporal by
// default, with height at indices 1,4,7,... and width at 2,5,8,... (each up to
// its mrope_section span). For text tokens t==h==w so the selection is moot.
mx::array build_selector(const ModelConfig& cfg) {
  const int half = cfg.head_dim / 2;
  std::vector<int> sel(half, 0);
  const auto& sec = cfg.mrope_section;
  if (sec.size() == 3) {
    for (int i = 1; i < std::min(sec[1] * 3, half); i += 3) sel[i] = 1;
    for (int i = 2; i < std::min(sec[2] * 3, half); i += 3) sel[i] = 2;
  }
  return mx::array(sel.data(), {half}, mx::int32);
}
}  // namespace

mx::array merge_image_features(const mx::array& token_embeds, const mx::array& image_features,
                               const std::vector<int>& input_ids, int image_token_id) {
  // Splice the sequence into runs of text / image rows, taking text rows from the
  // token embeddings and image rows from the ViT features (consumed in order).
  const int seq = static_cast<int>(input_ids.size());
  const int hidden = token_embeds.shape()[1];
  std::vector<mx::array> parts;
  int i = 0, feat_off = 0;
  while (i < seq) {
    const bool is_image = input_ids[i] == image_token_id;
    int j = i;
    while (j < seq && (input_ids[j] == image_token_id) == is_image) ++j;
    if (is_image) {
      parts.push_back(mx::slice(image_features, {feat_off, 0}, {feat_off + (j - i), hidden}));
      feat_off += j - i;
    } else {
      parts.push_back(mx::slice(token_embeds, {i, 0}, {j, hidden}));
    }
    i = j;
  }
  return mx::concatenate(parts, 0);
}

mx::array mrope_position_ids(const std::vector<int>& input_ids,
                             const std::vector<std::array<int, 3>>& image_grids,
                             const ModelConfig& cfg) {
  const int seq = static_cast<int>(input_ids.size());
  const int merge = cfg.vision ? cfg.vision->spatial_merge_size : 1;
  const int image_tok = cfg.image_token_id;

  std::array<std::vector<int>, 3> rows;  // temporal, height, width
  for (auto& r : rows) r.reserve(seq);

  int base = 0;       // next position to assign (== max so far + 1)
  size_t image_idx = 0;
  int p = 0;
  while (p < seq) {
    if (input_ids[p] == image_tok && image_idx < image_grids.size()) {
      // An image run: t * (h/merge) * (w/merge) collapsed patches, laid out in
      // (frame, height, width) order. Each axis is offset from the same base; the
      // base then jumps past the largest extent (so text resumes after the image).
      const auto& g = image_grids[image_idx++];
      const int t = g[0], gh = g[1] / merge, gw = g[2] / merge;
      for (int f = 0; f < t; ++f)
        for (int hh = 0; hh < gh; ++hh)
          for (int ww = 0; ww < gw; ++ww) {
            rows[0].push_back(base + f);
            rows[1].push_back(base + hh);
            rows[2].push_back(base + ww);
          }
      base += std::max({t, gh, gw});
      p += t * gh * gw;
    } else {
      // Text token: equal positions on all three axes.
      for (auto& r : rows) r.push_back(base);
      ++base;
      ++p;
    }
  }

  // Pack the three rows into a (3, seq) int32 array.
  std::vector<int> flat;
  flat.reserve(static_cast<size_t>(3) * seq);
  for (const auto& r : rows) flat.insert(flat.end(), r.begin(), r.end());
  return mx::array(flat.data(), {3, seq}, mx::int32);
}

Qwen3VLModel::Qwen3VLModel(ModelConfig config, Weights weights)
    : Qwen3Model(std::move(config), std::move(weights)),
      inv_freq_(build_inv_freq(this->config())),
      mrope_selector_(build_selector(this->config())) {}

std::pair<mx::array, mx::array> Qwen3VLModel::mrope_cos_sin(const mx::array& position_ids) const {
  // Per frequency, pick the t/h/w axis position, then angle = position * inv_freq.
  mx::array sel_pos = mx::take(position_ids, mrope_selector_, /*axis=*/0);  // (head_dim/2, seq)
  mx::array pos = mx::astype(mx::transpose(sel_pos, {1, 0}), mx::float32);  // (seq, head_dim/2)
  mx::array freqs = mx::multiply(pos, mx::reshape(inv_freq_, {1, config().head_dim / 2}));
  return {mx::cos(freqs), mx::sin(freqs)};  // each (seq, head_dim/2) float32
}

mx::array Qwen3VLModel::apply_mrope(const mx::array& x, const mx::array& cos,
                                    const mx::array& sin) const {
  // x (B, heads, seq, hd); cos/sin (seq, hd/2). Half-split rotation: tile cos/sin
  // to hd and pair dim d with d+hd/2.
  const int B = x.shape()[0], H = x.shape()[1], S = x.shape()[2], hd = x.shape()[3];
  mx::array cosf = mx::reshape(mx::concatenate({cos, cos}, -1), {1, 1, S, hd});
  mx::array sinf = mx::reshape(mx::concatenate({sin, sin}, -1), {1, 1, S, hd});
  mx::array xf = mx::astype(x, mx::float32);
  mx::array x1 = mx::slice(xf, {0, 0, 0, 0}, {B, H, S, hd / 2});
  mx::array x2 = mx::slice(xf, {0, 0, 0, hd / 2}, {B, H, S, hd});
  mx::array rot = mx::concatenate({mx::negative(x2), x1}, -1);
  mx::array out = mx::add(mx::multiply(xf, cosf), mx::multiply(rot, sinf));
  return mx::astype(out, x.dtype());
}

Qwen3VLModel::RopedQK Qwen3VLModel::roped_qk(int i, const mx::array& hidden,
                                             const mx::array& position_ids) const {
  QKV p = project_qkv(hidden, i);  // QK-Norm applied (Qwen3), no RoPE yet
  auto cs = mrope_cos_sin(position_ids);
  return {apply_mrope(p.q, cs.first, cs.second), apply_mrope(p.k, cs.first, cs.second)};
}

mx::array Qwen3VLModel::mm_attention(int i, const mx::array& x, const mx::array& cos,
                                     const mx::array& sin) const {
  const int B = x.shape()[0], L = x.shape()[1];
  QKV p = project_qkv(x, i);
  mx::array q = apply_mrope(p.q, cos, sin);
  mx::array k = apply_mrope(p.k, cos, sin);
  const float scale = 1.0f / std::sqrt(static_cast<float>(config().head_dim));
  // Multi-token prefill is causal; GQA (32:8 heads) is handled natively by SDPA.
  mx::array out = mx::fast::scaled_dot_product_attention(q, k, p.v, scale, /*mask_mode=*/"causal");
  out = mx::reshape(mx::transpose(out, {0, 2, 1, 3}), {B, L, config().n_heads * config().head_dim});
  return linear(out, layer_key(i, "self_attn.o_proj.weight"));
}

mx::array Qwen3VLModel::forward_multimodal(const std::vector<int>& input_ids,
                                           const mx::array& image_features,
                                           const std::vector<mx::array>& deepstack,
                                           const mx::array& position_ids) const {
  const int seq = static_cast<int>(input_ids.size());
  const int hidden = config().hidden;
  const int image_tok = config().image_token_id;

  // Token embeddings with the ViT features scattered into the image_pad rows.
  std::vector<int> ids = input_ids;
  mx::array tokens(ids.data(), {1, seq}, mx::int32);
  mx::array merged = merge_image_features(mx::reshape(embed(tokens), {seq, hidden}), image_features,
                                          ids, image_tok);
  mx::array h = mx::reshape(merged, {1, seq, hidden});

  auto cs = mrope_cos_sin(position_ids);
  for (int i = 0; i < config().n_layers; ++i) {
    mx::array attended = mx::add(h, mm_attention(i, h, cs.first, cs.second));
    mx::array post = rms_norm(attended, layer_w(i, "post_attention_layernorm.weight"));
    h = mx::add(attended, feed_forward(post, i));
    // DeepStack: add feature j at the image rows after decoder layer j. Reuse the
    // merge splicer with a zero text base so non-image rows contribute nothing.
    if (i < static_cast<int>(deepstack.size())) {
      mx::array zeros = mx::zeros({seq, hidden}, h.dtype());
      mx::array inject = merge_image_features(zeros, deepstack[i], ids, image_tok);
      h = mx::add(h, mx::reshape(inject, {1, seq, hidden}));
    }
  }
  h = rms_norm(h, weights().at("model.norm.weight"));
  const std::string head = weights().has("lm_head.weight") ? "lm_head.weight"
                                                           : "model.embed_tokens.weight";
  return linear(h, head);  // (1, seq, vocab)
}

}  // namespace mlxforge
