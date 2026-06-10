#include "model/qwen3_5.h"

#include <cmath>
#include <string>
#include <vector>

#include "model/sdpa.h"

#include "mlx/fast.h"
#include "mlx/ops.h"

namespace mlxforge {

namespace {
// softplus(x) = log(1 + e^x), the numerically stable logaddexp(0, x). Keeps x's
// dtype (mlx_lm computes the gate's softplus in the fp16 activation dtype).
mx::array softplus(const mx::array& x) { return mx::logaddexp(x, mx::zeros_like(x)); }
}  // namespace

mx::array Qwen35Model::partial_rope(const mx::array& x, int offset) const {
  // Plain RoPE on the leading rotary_dim_ dims (base rope_theta, no precomputed
  // freqs); the trailing head dims are left un-rotated. Mirrors mlx_lm's
  // initialize_rope(rope_type="default") -> nn.RoPE(rotary_dim, base=rope_theta).
  return mx::fast::rope(x, rotary_dim_, /*traditional=*/false, /*base=*/cfg_.rope_theta,
                        /*scale=*/1.0f, offset, /*freqs=*/std::nullopt);
}

mx::array Qwen35Model::partial_rope(const mx::array& x, const mx::array& offset) const {
  return mx::fast::rope(x, rotary_dim_, /*traditional=*/false, /*base=*/cfg_.rope_theta,
                        /*scale=*/1.0f, offset, /*freqs=*/std::nullopt);
}

mx::array Qwen35Model::attention(const mx::array& x, int layer, int offset, KVCache* cache) const {
  const int B = x.shape()[0];
  const int L = x.shape()[1];
  const int H = cfg_.n_heads;
  const int KV = cfg_.n_kv_heads;
  const int D = cfg_.head_dim;

  mx::array x_normed = rms_norm(x, layer_w(layer, "input_layernorm.weight"));

  // q_proj is 2x width: per head it is [queries(D) || gate(D)]. Reshape to heads
  // and split the last axis into the query and the output-gate halves.
  mx::array qg = mx::reshape(linear(x_normed, layer_key(layer, "self_attn.q_proj.weight")),
                             {B, L, H, 2 * D});
  mx::array queries = mx::slice(qg, {0, 0, 0, 0}, {B, L, H, D});
  mx::array gate = mx::reshape(mx::slice(qg, {0, 0, 0, D}, {B, L, H, 2 * D}), {B, L, H * D});

  // QK-Norm (RMSNorm over head_dim per Q/K head) before RoPE, then to head-major
  // (B, heads, L, head_dim). V is never QK-normed.
  queries = rms_norm(queries, layer_w(layer, "self_attn.q_norm.weight"));
  queries = mx::transpose(queries, {0, 2, 1, 3});

  mx::array keys = mx::reshape(linear(x_normed, layer_key(layer, "self_attn.k_proj.weight")),
                               {B, L, KV, D});
  keys = mx::transpose(rms_norm(keys, layer_w(layer, "self_attn.k_norm.weight")), {0, 2, 1, 3});

  mx::array values = mx::transpose(
      mx::reshape(linear(x_normed, layer_key(layer, "self_attn.v_proj.weight")), {B, L, KV, D}),
      {0, 2, 1, 3});

  queries = partial_rope(queries, offset);
  keys = partial_rope(keys, offset);

  const float scale = 1.0f / std::sqrt(static_cast<float>(D));
  const std::string mask_mode = L > 1 ? "causal" : "";
  mx::array out = sdpa_with_cache(queries, keys, values, cache, layer, scale, mask_mode);

  // (B, H, L, D) -> (B, L, H*D), apply the sigmoid output gate, then o_proj.
  out = mx::reshape(mx::transpose(out, {0, 2, 1, 3}), {B, L, H * D});
  out = mx::multiply(out, mx::sigmoid(gate));
  return linear(out, layer_key(layer, "self_attn.o_proj.weight"));
}

mx::array Qwen35Model::decoder_block(const mx::array& x, int layer, int offset,
                                     KVCache* cache) const {
  // Linear layers run Gated-DeltaNet (no softmax cache); full layers run gated
  // attention. The residual structure and dense SwiGLU MLP are shared.
  mx::array sub = cfg_.is_linear_layer(layer) ? linear_attention(x, layer, cache)
                                              : attention(x, layer, offset, cache);
  mx::array h = mx::add(x, sub);
  mx::array post = rms_norm(h, layer_w(layer, "post_attention_layernorm.weight"));
  return mx::add(h, feed_forward(post, layer));
}

std::pair<mx::array, mx::array> Qwen35Model::gated_delta(
    const mx::array& q_in, const mx::array& k_in, const mx::array& v, const mx::array& g,
    const mx::array& beta, const std::optional<mx::array>& init_state,
    const std::optional<mx::array>& mask) const {
  const int B = q_in.shape()[0];
  const int S = q_in.shape()[1];
  const int Hk = q_in.shape()[2];
  const int Dk = q_in.shape()[3];
  const int Hv = v.shape()[2];
  const int Dv = v.shape()[3];

  // Grouped value attention: replicate the q/k heads up to Hv when Hv > Hk (a
  // no-op for the 0.8B model, where the head counts are equal).
  mx::array q = q_in, k = k_in;
  if (Hv > Hk) {
    const int rep = Hv / Hk;
    q = mx::repeat(q, rep, /*axis=*/2);
    k = mx::repeat(k, rep, /*axis=*/2);
  }

  // Slice one timestep out of a (B, S, ...) tensor -> (B, ...).
  auto step = [](const mx::array& a, int t) {
    mx::Shape start = a.shape(), stop = a.shape();
    for (auto& e : start) e = 0;
    start[1] = t;
    stop[1] = t + 1;
    return mx::squeeze(mx::slice(a, start, stop), 1);
  };

  // Fixed-size recurrent state S_t in (B, Hv, Dv, Dk), accumulated in fp32 and
  // carried across calls. The delta rule: decay the state, read the current key
  // (kv_mem), form the error-corrected value (delta), write it back keyed by k,
  // then read with q. A masked (left-padding) step freezes the state.
  mx::array state = init_state.has_value() ? mx::astype(*init_state, mx::float32)
                                           : mx::zeros({B, Hv, Dv, Dk}, mx::float32);
  std::vector<mx::array> ys;
  ys.reserve(S);
  for (int t = 0; t < S; ++t) {
    mx::array decay = mx::reshape(step(g, t), {B, Hv, 1, 1});       // exp gating
    mx::array beta_t = mx::reshape(step(beta, t), {B, Hv, 1});
    mx::array q_t = mx::reshape(step(q, t), {B, Hv, 1, Dk});
    mx::array k_t = mx::reshape(step(k, t), {B, Hv, 1, Dk});
    mx::array v_t = step(v, t);                                     // (B, Hv, Dv)

    mx::array prev = state;
    state = mx::multiply(state, decay);
    mx::array kv_mem = mx::sum(mx::multiply(state, k_t), /*axis=*/-1);   // (B, Hv, Dv)
    mx::array delta = mx::multiply(mx::subtract(v_t, kv_mem), beta_t);   // (B, Hv, Dv)
    state = mx::add(state, mx::multiply(k_t, mx::reshape(delta, {B, Hv, Dv, 1})));
    mx::array y_t = mx::sum(mx::multiply(state, q_t), /*axis=*/-1);      // (B, Hv, Dv)
    if (mask.has_value()) {
      // Padded steps (mask == false): freeze the state and emit zeros.
      mx::array m = mx::reshape(step(*mask, t), {B, 1, 1, 1});  // (B,) bool -> broadcast
      state = mx::where(m, state, prev);
      y_t = mx::where(mx::reshape(m, {B, 1, 1}), y_t, mx::zeros_like(y_t));
    }
    ys.push_back(mx::reshape(mx::astype(y_t, q_in.dtype()), {B, 1, Hv, Dv}));
  }
  return {mx::concatenate(ys, /*axis=*/1), state};  // (B, S, Hv, Dv), (B, Hv, Dv, Dk)
}

mx::array Qwen35Model::linear_attention(const mx::array& x, int layer, KVCache* cache) const {
  // Single-sequence path: no left-padding, so the ssm mask is always absent.
  // init_state/conv_state come from the cache on decode steps (nullopt on the
  // first prefill call). The streaming linear state is read and written here.
  const std::optional<mx::array> no_mask = std::nullopt;
  std::optional<mx::array> init_conv, init_recur;
  if (cache && cache->has_linear_state(layer)) {
    auto st = cache->linear_state(layer);
    init_conv = st.first;
    init_recur = st.second;
  }
  mx::array out_conv = mx::zeros({1}), out_recur = mx::zeros({1});  // out-params, set below
  mx::array y = linear_attention_impl(x, layer, init_conv, init_recur, no_mask, out_conv, out_recur);
  if (cache) cache->set_linear_state(layer, out_conv, out_recur);
  return y;
}

mx::array Qwen35Model::linear_attention_impl(const mx::array& x, int layer,
                                             const std::optional<mx::array>& init_conv,
                                             const std::optional<mx::array>& init_recur,
                                             const std::optional<mx::array>& mask,
                                             mx::array& out_conv, mx::array& out_recur) const {
  const int B = x.shape()[0];
  const int S = x.shape()[1];
  const int Hk = cfg_.linear_num_key_heads;
  const int Hv = cfg_.linear_num_value_heads;
  const int Dk = cfg_.linear_key_head_dim;
  const int Dv = cfg_.linear_value_head_dim;
  const int K = cfg_.linear_conv_kernel_dim;
  const int key_dim = Hk * Dk;
  const int value_dim = Hv * Dv;
  const int conv_dim = 2 * key_dim + value_dim;

  mx::array x_normed = rms_norm(x, layer_w(layer, "input_layernorm.weight"));

  mx::array qkv = linear(x_normed, layer_key(layer, "linear_attn.in_proj_qkv.weight"));
  mx::array z = mx::reshape(linear(x_normed, layer_key(layer, "linear_attn.in_proj_z.weight")),
                            {B, S, Hv, Dv});
  mx::array a = linear(x_normed, layer_key(layer, "linear_attn.in_proj_a.weight"));  // (B,S,Hv)
  mx::array b = linear(x_normed, layer_key(layer, "linear_attn.in_proj_b.weight"));  // (B,S,Hv)

  // Zero out padded (left-pad) positions before they enter the conv/recurrence.
  if (mask.has_value()) {
    qkv = mx::where(mx::reshape(*mask, {B, S, 1}), qkv, mx::zeros_like(qkv));
  }

  // Causal depthwise Conv1d over [conv_state (last K-1 inputs) ; qkv], then SiLU.
  // The weight is (conv_dim, K, 1); groups == conv_dim makes it per-channel.
  // Output position t mixes the K most recent inputs, i.e. it is causal. The new
  // conv_state is the last K-1 inputs, carried to the next step.
  mx::array conv_state =
      init_conv.has_value() ? *init_conv : mx::zeros({B, K - 1, conv_dim}, qkv.dtype());
  mx::array conv_in = mx::concatenate({conv_state, qkv}, /*axis=*/1);  // (B, S+K-1, conv_dim)
  out_conv = mx::slice(conv_in, {0, S, 0}, {B, S + K - 1, conv_dim});  // last K-1
  mx::array conv_out = mx::conv1d(conv_in, layer_w(layer, "linear_attn.conv1d.weight"),
                                  /*stride=*/1, /*padding=*/0, /*dilation=*/1, /*groups=*/conv_dim);
  conv_out = mx::multiply(conv_out, mx::sigmoid(conv_out));  // SiLU, (B, S, conv_dim)

  // Split the conv output into per-head q, k, v.
  mx::array q = mx::reshape(mx::slice(conv_out, {0, 0, 0}, {B, S, key_dim}), {B, S, Hk, Dk});
  mx::array k =
      mx::reshape(mx::slice(conv_out, {0, 0, key_dim}, {B, S, 2 * key_dim}), {B, S, Hk, Dk});
  mx::array v =
      mx::reshape(mx::slice(conv_out, {0, 0, 2 * key_dim}, {B, S, conv_dim}), {B, S, Hv, Dv});

  // L2/RMS-normalize each q/k head (no weight) then scale: q by 1/Dk, k by
  // 1/sqrt(Dk). This replaces softmax's normalization in the linear-attention map.
  const float inv_scale = 1.0f / std::sqrt(static_cast<float>(Dk));
  q = mx::multiply(mx::array(inv_scale * inv_scale, mx::float16),
                   mx::fast::rms_norm(q, std::nullopt, 1e-6f));
  k = mx::multiply(mx::array(inv_scale, mx::float16), mx::fast::rms_norm(k, std::nullopt, 1e-6f));

  // Per-head gates: beta = sigmoid(b); decay g = exp(-exp(A_log) * softplus(a +
  // dt_bias)). A_log is kept fp32 (the double exp is fp16-sensitive).
  mx::array A_log = layer_w(layer, "linear_attn.A_log");       // (Hv,) fp32
  mx::array dt_bias = layer_w(layer, "linear_attn.dt_bias");   // (Hv,)
  mx::array g = mx::exp(mx::multiply(mx::negative(mx::exp(mx::astype(A_log, mx::float32))),
                                     softplus(mx::add(a, dt_bias))));  // (B, S, Hv) fp32
  mx::array beta = mx::sigmoid(b);

  auto [out, final_state] = gated_delta(q, k, v, g, beta, init_recur, mask);  // (B, S, Hv, Dv)
  out_recur = final_state;

  // Gated RMSNorm: rms_norm(out) * silu(z), computed in fp32 (precise SwiGLU).
  mx::array normed = mx::fast::rms_norm(
      out, std::optional<mx::array>(layer_w(layer, "linear_attn.norm.weight")), cfg_.rms_eps);
  mx::array zf = mx::astype(z, mx::float32);
  mx::array gated = mx::multiply(mx::multiply(zf, mx::sigmoid(zf)), mx::astype(normed, mx::float32));
  gated = mx::astype(gated, out.dtype());

  return linear(mx::reshape(gated, {B, S, Hv * Dv}), layer_key(layer, "linear_attn.out_proj.weight"));
}

mx::array Qwen35Model::attention_batched_gated(const mx::array& x, int layer,
                                               const mx::array& offset, const mx::array& mask,
                                               const mx::array& real_pos, BatchKVCache& cache) const {
  const int B = x.shape()[0];
  const int L = x.shape()[1];
  const int H = cfg_.n_heads;
  const int KV = cfg_.n_kv_heads;
  const int D = cfg_.head_dim;

  mx::array x_normed = rms_norm(x, layer_w(layer, "input_layernorm.weight"));

  mx::array qg = mx::reshape(linear(x_normed, layer_key(layer, "self_attn.q_proj.weight")),
                             {B, L, H, 2 * D});
  mx::array queries = mx::slice(qg, {0, 0, 0, 0}, {B, L, H, D});
  mx::array gate = mx::reshape(mx::slice(qg, {0, 0, 0, D}, {B, L, H, 2 * D}), {B, L, H * D});

  queries = mx::transpose(rms_norm(queries, layer_w(layer, "self_attn.q_norm.weight")),
                          {0, 2, 1, 3});
  mx::array keys = mx::reshape(linear(x_normed, layer_key(layer, "self_attn.k_proj.weight")),
                               {B, L, KV, D});
  keys = mx::transpose(rms_norm(keys, layer_w(layer, "self_attn.k_norm.weight")), {0, 2, 1, 3});
  mx::array values = mx::transpose(
      mx::reshape(linear(x_normed, layer_key(layer, "self_attn.v_proj.weight")), {B, L, KV, D}),
      {0, 2, 1, 3});

  queries = partial_rope(queries, offset);
  keys = partial_rope(keys, offset);  // append roped K, un-roped V

  const float scale = 1.0f / std::sqrt(static_cast<float>(D));
  mx::array out = sdpa_with_cache(queries, keys, values, cache, layer, scale, mask);
  out = mx::reshape(mx::transpose(out, {0, 2, 1, 3}), {B, L, H * D});

  // A left-padding query position attends to zero valid keys, so SDPA returns NaN
  // for that whole row (softmax over an all -inf mask). Llama tolerates this (the
  // NaN stays confined to padded positions), but Qwen3.5's linear layers mix
  // positions through the conv and read the residual via in_proj_z, so the NaN
  // would leak into real tokens. Zero those padded rows to keep the residual
  // stream finite; they are masked out of every real computation anyway.
  out = mx::where(mx::reshape(real_pos, {B, L, 1}), out, mx::zeros_like(out));

  out = mx::multiply(out, mx::sigmoid(gate));
  return linear(out, layer_key(layer, "self_attn.o_proj.weight"));
}

mx::array Qwen35Model::forward(const mx::array& tokens, BatchKVCache& cache) const {
  const int B = tokens.shape()[0];
  const int S = tokens.shape()[1];
  const int prev = cache.idx();
  mx::array offset = cache.offset();  // per-row, read once before any cache write
  mx::array kv_mask = batch_mask(prev, S, cache.left_padding());

  // ssm mask for the linear layers: token at chunk-position s (absolute prev + s)
  // is real iff its absolute position is past this row's left padding. An all-true
  // mask (decode, or an unpadded prefill) is a numeric no-op.
  mx::array abs_pos = mx::add(mx::array(prev, mx::int32), mx::arange(0, S, 1, mx::int32));  // (S,)
  mx::array real_pos =  // (B, S) bool: token's absolute position is past its left padding
      mx::greater_equal(mx::reshape(abs_pos, {1, S}), mx::reshape(cache.left_padding(), {B, 1}));
  std::optional<mx::array> ssm_mask = real_pos;

  mx::array h = embed(tokens);
  for (int layer = 0; layer < cfg_.n_layers; ++layer) {
    mx::array sub = mx::zeros({1});
    if (cfg_.is_linear_layer(layer)) {
      std::optional<mx::array> init_conv, init_recur;
      if (cache.has_linear_state(layer)) {
        auto st = cache.linear_state(layer);
        init_conv = st.first;
        init_recur = st.second;
      }
      mx::array out_conv = mx::zeros({1}), out_recur = mx::zeros({1});  // out-params, set below
      sub = linear_attention_impl(h, layer, init_conv, init_recur, ssm_mask, out_conv, out_recur);
      cache.set_linear_state(layer, out_conv, out_recur);
    } else {
      sub = attention_batched_gated(h, layer, offset, kv_mask, real_pos, cache);
    }
    mx::array attended = mx::add(h, sub);
    mx::array post = rms_norm(attended, layer_w(layer, "post_attention_layernorm.weight"));
    h = mx::add(attended, feed_forward(post, layer));
  }
  cache.advance(S);

  h = rms_norm(h, w_.at("model.norm.weight"));
  const std::string head_key =
      w_.has("lm_head.weight") ? "lm_head.weight" : "model.embed_tokens.weight";
  return linear(h, head_key);
}

}  // namespace mlxforge
