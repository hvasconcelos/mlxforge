#include "model/decoder_model.h"

#include <cmath>
#include <limits>
#include <optional>
#include <string_view>
#include <utility>

#include "core/logging.h"
#include "model/sdpa.h"

#include "mlx/fast.h"
#include "mlx/ops.h"
#include "mlx/transforms.h"

namespace mlxforge {

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

  // GGUF checkpoints bake the llama3 rescaling into a per-dimension factor array
  // (no scaling params in the metadata): final freqs = base**(2i/d) * factors.
  // This reproduces mlx_lm's Llama3RoPE `_freqs` exactly (validated against the
  // rope_freqs.npy golden fixture).
  if (cfg.rope_freq_factors) {
    mx::array factors = mx::array(cfg.rope_freq_factors->data(),
                                  {static_cast<int>(cfg.rope_freq_factors->size())}, mx::float32);
    freqs = mx::multiply(freqs, factors);
    mx::eval(freqs);
    return freqs;
  }

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

DecoderModel::DecoderModel(ModelConfig config, Weights weights)
    : cfg_(std::move(config)), w_(std::move(weights)), rope_freqs_(compute_rope_freqs(cfg_)) {
  log::debug("DecoderModel: type={} layers={} hidden={} heads={}/{} head_dim={} vocab={} "
             "quantized={}",
             cfg_.model_type, cfg_.n_layers, cfg_.hidden, cfg_.n_heads, cfg_.n_kv_heads,
             cfg_.head_dim, cfg_.vocab, cfg_.quantized);
}

std::string DecoderModel::layer_key(int i, const std::string& suffix) const {
  return "model.layers." + std::to_string(i) + "." + suffix;
}

const mx::array& DecoderModel::layer_w(int i, const std::string& suffix) const {
  return w_.at(layer_key(i, suffix));
}

mx::array DecoderModel::linear(const mx::array& x, const std::string& weight_key) const {
  // Quantization is detected per-weight (a "<base>.scales" sibling), not from a
  // global flag: a checkpoint may mix quantized and dense weights (GGUF) or vary
  // bit-width per layer (mixed-precision MLX repos).
  QuantParams qp;
  if (w_.is_quantized(weight_key, qp)) {
    static constexpr std::string_view kWeightSuffix = ".weight";
    const std::string base = weight_key.substr(0, weight_key.size() - kWeightSuffix.size());
    return mx::quantized_matmul(x, w_.at(weight_key), w_.at(base + ".scales"),
                                w_.at(base + ".biases"), /*transpose=*/true, qp.group_size,
                                qp.bits);
  }
  return mx::matmul(x, mx::transpose(w_.at(weight_key)));  // weight is (out, in)
}

mx::array DecoderModel::embed(const mx::array& tokens) const {
  QuantParams qp;
  if (w_.is_quantized("model.embed_tokens.weight", qp)) {
    // Gather the quantized rows for these tokens, then dequantize just those.
    mx::array w = mx::take(w_.at("model.embed_tokens.weight"), tokens, /*axis=*/0);
    mx::array s = mx::take(w_.at("model.embed_tokens.scales"), tokens, /*axis=*/0);
    mx::array b = mx::take(w_.at("model.embed_tokens.biases"), tokens, /*axis=*/0);
    return mx::dequantize(w, s, b, qp.group_size, qp.bits);
  }
  return mx::take(w_.at("model.embed_tokens.weight"), tokens, /*axis=*/0);
}

mx::array DecoderModel::rms_norm(const mx::array& x, const mx::array& weight) const {
  return mx::fast::rms_norm(x, std::optional<mx::array>(weight), cfg_.rms_eps);
}

mx::array DecoderModel::apply_rope(const mx::array& x, int offset) const {
  return mx::fast::rope(x, cfg_.head_dim, /*traditional=*/false, /*base=*/std::nullopt,
                        /*scale=*/1.0f, offset, rope_freqs_);
}

mx::array DecoderModel::apply_rope(const mx::array& x, const mx::array& offset) const {
  return mx::fast::rope(x, cfg_.head_dim, /*traditional=*/false, /*base=*/std::nullopt,
                        /*scale=*/1.0f, offset, rope_freqs_);
}

mx::array DecoderModel::norm_qk_head(const mx::array& h, int /*layer*/, bool /*is_query*/) const {
  return h;  // Llama: no per-head Q/K normalization.
}

DecoderModel::QKV DecoderModel::project_qkv(const mx::array& x, int layer) const {
  const int B = x.shape()[0];
  const int L = x.shape()[1];
  mx::array x_normed = rms_norm(x, layer_w(layer, "input_layernorm.weight"));

  // Reshape a projection to per-head form (B, heads, L, head_dim). Q/K route
  // through norm_qk_head() (Qwen3 RMSNorm over head_dim before RoPE; identity
  // for Llama); V is never QK-normed (is_qk == false).
  auto to_heads = [&](const mx::array& proj, int heads, bool is_qk, bool is_query) {
    mx::array h = mx::reshape(proj, {B, L, heads, cfg_.head_dim});
    if (is_qk) h = norm_qk_head(h, layer, is_query);
    return mx::transpose(h, {0, 2, 1, 3});
  };
  return {to_heads(linear(x_normed, layer_key(layer, "self_attn.q_proj.weight")), cfg_.n_heads,
                   /*is_qk=*/true, /*is_query=*/true),
          to_heads(linear(x_normed, layer_key(layer, "self_attn.k_proj.weight")), cfg_.n_kv_heads,
                   /*is_qk=*/true, /*is_query=*/false),
          to_heads(linear(x_normed, layer_key(layer, "self_attn.v_proj.weight")), cfg_.n_kv_heads,
                   /*is_qk=*/false, /*is_query=*/false)};
}

DecoderModel::QKV DecoderModel::attn_qkv(const mx::array& x, int layer, int offset) const {
  QKV p = project_qkv(x, layer);
  return {apply_rope(p.q, offset), apply_rope(p.k, offset), p.v};
}

mx::array DecoderModel::attention(const mx::array& x, int layer, int offset, KVCache* cache) const {
  const int B = x.shape()[0];
  const int L = x.shape()[1];

  QKV qkv = attn_qkv(x, layer, offset);  // q (B, n_heads, L, head_dim), k/v use n_kv_heads
  const float scale = 1.0f / std::sqrt(static_cast<float>(cfg_.head_dim));

  // Multi-token chunks (prefill) are causal; a single decode token attends over
  // the whole cached history unmasked. GQA is handled natively by SDPA.
  const std::string mask_mode = L > 1 ? "causal" : "";
  mx::array out = sdpa_with_cache(qkv.q, qkv.k, qkv.v, cache, layer, scale, mask_mode);

  // (B, n_heads, L, head_dim) -> (B, L, n_heads*head_dim)
  out = mx::reshape(mx::transpose(out, {0, 2, 1, 3}), {B, L, cfg_.n_heads * cfg_.head_dim});
  return linear(out, layer_key(layer, "self_attn.o_proj.weight"));
}

mx::array DecoderModel::mlp(const mx::array& x, int layer) const {
  mx::array gate = linear(x, layer_key(layer, "mlp.gate_proj.weight"));
  mx::array up = linear(x, layer_key(layer, "mlp.up_proj.weight"));
  mx::array silu = mx::multiply(gate, mx::sigmoid(gate));
  return linear(mx::multiply(silu, up), layer_key(layer, "mlp.down_proj.weight"));
}

mx::array DecoderModel::feed_forward(const mx::array& x, int layer) const {
  return mlp(x, layer);  // Llama / dense default.
}

mx::array DecoderModel::decoder_block(const mx::array& x, int layer, int offset,
                                      KVCache* cache) const {
  mx::array h = mx::add(x, attention(x, layer, offset, cache));
  mx::array post = rms_norm(h, layer_w(layer, "post_attention_layernorm.weight"));
  return mx::add(h, feed_forward(post, layer));
}

mx::array DecoderModel::batch_mask(int prev_idx, int n_query, const mx::array& left_padding) const {
  const int t_kv = prev_idx + n_query;
  const int B = left_padding.shape()[0];

  mx::array kpos = mx::arange(0, t_kv, 1, mx::int32);                  // (T_kv,)
  mx::array qpos = mx::arange(prev_idx, prev_idx + n_query, 1, mx::int32);  // (N,)

  // causal[q, k] = qpos[q] >= kpos[k]
  mx::array causal = mx::greater_equal(mx::reshape(qpos, {1, 1, n_query, 1}),
                                       mx::reshape(kpos, {1, 1, 1, t_kv}));
  // valid[b, k] = left_padding[b] <= kpos[k]  (drop the left-pad region)
  mx::array valid = mx::less_equal(mx::reshape(left_padding, {B, 1, 1, 1}),
                                   mx::reshape(kpos, {1, 1, 1, t_kv}));
  mx::array keep = mx::logical_and(causal, valid);  // -> (B, 1, N, T_kv)

  // Additive fp16 mask (avoid boolean masks; #2894).
  const float ninf = -std::numeric_limits<float>::infinity();
  return mx::where(keep, mx::array(0.0f, mx::float16), mx::array(ninf, mx::float16));
}

mx::array DecoderModel::attention_batched(const mx::array& x, int layer, const mx::array& offset,
                                          const mx::array& mask, BatchKVCache& cache) const {
  const int B = x.shape()[0];
  const int L = x.shape()[1];

  QKV p = project_qkv(x, layer);
  mx::array q = apply_rope(p.q, offset);
  mx::array k = apply_rope(p.k, offset);  // append roped K, un-roped V

  const float scale = 1.0f / std::sqrt(static_cast<float>(cfg_.head_dim));
  mx::array out = sdpa_with_cache(q, k, p.v, cache, layer, scale, mask);
  out = mx::reshape(mx::transpose(out, {0, 2, 1, 3}), {B, L, cfg_.n_heads * cfg_.head_dim});
  return linear(out, layer_key(layer, "self_attn.o_proj.weight"));
}

mx::array DecoderModel::forward(const mx::array& tokens, BatchKVCache& cache) const {
  const int N = tokens.shape()[1];
  const int prev = cache.idx();
  mx::array offset = cache.offset();  // per-row, read once before the cache write
  mx::array mask = batch_mask(prev, N, cache.left_padding());

  mx::array h = embed(tokens);
  // Same residual structure as decoder_block(), but with the batched attention
  // path (per-row RoPE offset + additive mask). feed_forward/norms are shared.
  for (int layer = 0; layer < cfg_.n_layers; ++layer) {
    mx::array attended = mx::add(h, attention_batched(h, layer, offset, mask, cache));
    mx::array post = rms_norm(attended, layer_w(layer, "post_attention_layernorm.weight"));
    h = mx::add(attended, feed_forward(post, layer));
  }
  cache.advance(N);

  h = rms_norm(h, w_.at("model.norm.weight"));
  // LM head: a separate lm_head when present, else the tied input embedding.
  const std::string head_key =
      w_.has("lm_head.weight") ? "lm_head.weight" : "model.embed_tokens.weight";
  return linear(h, head_key);
}

mx::array DecoderModel::forward_hidden(const mx::array& tokens, KVCache* cache) const {
  const int offset = cache ? cache->offset() : 0;
  mx::array h = embed(tokens);
  for (int layer = 0; layer < cfg_.n_layers; ++layer) {
    h = decoder_block(h, layer, offset, cache);
  }
  if (cache) cache->advance(tokens.shape()[1]);  // one offset bump per token sweep
  return rms_norm(h, w_.at("model.norm.weight"));  // (B, L, hidden)
}

mx::array DecoderModel::forward(const mx::array& tokens, KVCache* cache) const {
  mx::array h = forward_hidden(tokens, cache);
  // LM head: a separate lm_head when present, else the tied input embedding.
  const std::string head_key =
      w_.has("lm_head.weight") ? "lm_head.weight" : "model.embed_tokens.weight";
  return linear(h, head_key);  // (B, L, vocab)
}

}  // namespace mlxforge
