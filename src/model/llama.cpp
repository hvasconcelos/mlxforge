#include "model/llama.h"

#include <cmath>
#include <optional>
#include <utility>

#include "mlx/fast.h"
#include "mlx/ops.h"
#include "mlx/transforms.h"

namespace xllm {

namespace {

constexpr float kTwoPi = 6.283185307179586f;

// Precompute RoPE frequencies. For Llama-3.2 (rope_type "llama3") this applies
// the frequency rescaling that mlx_lm's Llama3RoPE precomputes and hands to
// fast::rope via `freqs` (with base disabled). Plain models fall back to the
// standard base**(2i/d) schedule. Returns head_dim/2 float32 values.
mx::array compute_rope_freqs(const ModelConfig& cfg) {
  const int hd = cfg.head_dim;
  const float base = cfg.rope_theta;

  // freqs = base ** (arange(0, hd, 2) / hd)
  mx::array idx = mx::arange(0, hd, 2, mx::float32);  // (hd/2,)
  mx::array exponent = mx::divide(idx, mx::array(static_cast<float>(hd)));
  mx::array freqs = mx::power(mx::array(base), exponent);

  if (!cfg.rope_scaling || cfg.rope_scaling->rope_type != "llama3") {
    mx::eval(freqs);
    return freqs;
  }

  const RopeScaling& rs = *cfg.rope_scaling;
  const float factor = rs.factor;
  const float low_ff = rs.low_freq_factor;
  const float high_ff = rs.high_freq_factor;
  const float old_ctx = static_cast<float>(rs.original_max_position_embeddings);
  const float low_wl = old_ctx / low_ff;
  const float high_wl = old_ctx / high_ff;

  mx::array wavelens = mx::multiply(mx::array(kTwoPi), freqs);

  // freqs = where(wavelens > low_wl, freqs * factor, freqs)
  freqs = mx::where(mx::greater(wavelens, mx::array(low_wl)),
                    mx::multiply(freqs, mx::array(factor)), freqs);

  // is_medium = (wavelens > high_wl) & (wavelens < low_wl)
  mx::array is_medium = mx::logical_and(mx::greater(wavelens, mx::array(high_wl)),
                                        mx::less(wavelens, mx::array(low_wl)));

  // smooth = (old_ctx / wavelens - low_ff) / (high_ff - low_ff)
  mx::array smooth =
      mx::divide(mx::subtract(mx::divide(mx::array(old_ctx), wavelens), mx::array(low_ff)),
                 mx::array(high_ff - low_ff));

  // smooth_freqs = freqs / ((1 - smooth) / factor + smooth)
  mx::array denom =
      mx::add(mx::divide(mx::subtract(mx::array(1.0f), smooth), mx::array(factor)), smooth);
  mx::array smooth_freqs = mx::divide(freqs, denom);

  freqs = mx::where(is_medium, smooth_freqs, freqs);
  mx::eval(freqs);
  return freqs;
}

}  // namespace

bool rope_array_offset_overload_available() {
  mx::array x = mx::zeros({1, 1, 1, 4}, mx::float16);
  mx::array offset = mx::array({0}, mx::int32);  // per-row offset (B,)
  mx::array y = mx::fast::rope(x, /*dims=*/4, /*traditional=*/false, /*base=*/10000.0f,
                               /*scale=*/1.0f, offset, /*freqs=*/std::nullopt);
  mx::eval(y);
  return true;
}

LlamaModel::LlamaModel(ModelConfig config, Weights weights)
    : cfg_(std::move(config)), w_(std::move(weights)), rope_freqs_(compute_rope_freqs(cfg_)) {}

std::string LlamaModel::layer_key(int i, const std::string& suffix) const {
  return "model.layers." + std::to_string(i) + "." + suffix;
}

const mx::array& LlamaModel::layer_w(int i, const std::string& suffix) const {
  return w_.at(layer_key(i, suffix));
}

mx::array LlamaModel::linear(const mx::array& x, const std::string& weight_key) const {
  return mx::matmul(x, mx::transpose(w_.at(weight_key)));  // weight is (out, in)
}

mx::array LlamaModel::embed(const mx::array& tokens) const {
  return mx::take(w_.at("model.embed_tokens.weight"), tokens, /*axis=*/0);
}

mx::array LlamaModel::rms_norm(const mx::array& x, const mx::array& weight) const {
  return mx::fast::rms_norm(x, std::optional<mx::array>(weight), cfg_.rms_eps);
}

mx::array LlamaModel::apply_rope(const mx::array& x, int offset) const {
  return mx::fast::rope(x, cfg_.head_dim, /*traditional=*/false, /*base=*/std::nullopt,
                        /*scale=*/1.0f, offset, rope_freqs_);
}

LlamaModel::QKV LlamaModel::attn_qkv(const mx::array& x, int layer, int offset) const {
  const int B = x.shape()[0];
  const int L = x.shape()[1];

  mx::array x_normed = rms_norm(x, layer_w(layer, "input_layernorm.weight"));

  auto to_heads = [&](const mx::array& proj, int heads) {
    return mx::transpose(mx::reshape(proj, {B, L, heads, cfg_.head_dim}), {0, 2, 1, 3});
  };
  mx::array q = to_heads(linear(x_normed, layer_key(layer, "self_attn.q_proj.weight")), cfg_.n_heads);
  mx::array k = to_heads(linear(x_normed, layer_key(layer, "self_attn.k_proj.weight")), cfg_.n_kv_heads);
  mx::array v = to_heads(linear(x_normed, layer_key(layer, "self_attn.v_proj.weight")), cfg_.n_kv_heads);

  return {apply_rope(q, offset), apply_rope(k, offset), v};
}

mx::array LlamaModel::attention(const mx::array& x, int layer) const {
  const int B = x.shape()[0];
  const int L = x.shape()[1];

  QKV qkv = attn_qkv(x, layer);  // q (B, n_heads, L, head_dim), k/v use n_kv_heads
  const float scale = 1.0f / std::sqrt(static_cast<float>(cfg_.head_dim));

  // GQA is handled natively by SDPA when q_heads > kv_heads.
  mx::array out = mx::fast::scaled_dot_product_attention(qkv.q, qkv.k, qkv.v, scale,
                                                         /*mask_mode=*/"causal");

  // (B, n_heads, L, head_dim) -> (B, L, n_heads*head_dim)
  out = mx::reshape(mx::transpose(out, {0, 2, 1, 3}), {B, L, cfg_.n_heads * cfg_.head_dim});
  return linear(out, layer_key(layer, "self_attn.o_proj.weight"));
}

mx::array LlamaModel::mlp(const mx::array& x, int layer) const {
  mx::array gate = linear(x, layer_key(layer, "mlp.gate_proj.weight"));
  mx::array up = linear(x, layer_key(layer, "mlp.up_proj.weight"));
  mx::array silu = mx::multiply(gate, mx::sigmoid(gate));
  return linear(mx::multiply(silu, up), layer_key(layer, "mlp.down_proj.weight"));
}

mx::array LlamaModel::decoder_block(const mx::array& x, int layer) const {
  mx::array h = mx::add(x, attention(x, layer));
  mx::array post = rms_norm(h, layer_w(layer, "post_attention_layernorm.weight"));
  return mx::add(h, mlp(post, layer));
}

}  // namespace xllm
