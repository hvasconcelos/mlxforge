// LLaMA-family decoder model on MLX. Builds the full forward pass in layers:
//   - embedding, RMSNorm, Q/K/V projections, RoPE (the attention front-half)
//   - a single decoder block (GQA attention + SwiGLU MLP + residuals)
//   - the full n-layer stack -> final RMSNorm -> LM head logits
// fp16 and 4-bit-quantized weights, single-stream and batched forward paths are
// all served by this one class (the transformer is shared across the supported
// model families; only tokenizer/chat formatting differs per family).
#pragma once

#include <string>

#include "mlx/array.h"

#include "cache/batch_kv_cache.h"
#include "cache/kv_cache.h"
#include "core/config.h"
#include "core/weights.h"

namespace mlxforge {

namespace mx = mlx::core;

// Confirms the pinned MLX exposes fast::rope(const array& offset, ...) — the
// per-row offset overload that batched (left-padded) decode depends on. Returns
// true if it ran the overload successfully.
bool rope_array_offset_overload_available();

class LlamaModel {
 public:
  LlamaModel(ModelConfig config, Weights weights);

  const ModelConfig& config() const { return cfg_; }
  const Weights& weights() const { return w_; }
  // Precomputed RoPE frequencies (llama3 rescaling), head_dim/2 float32 values.
  const mx::array& rope_freqs() const { return rope_freqs_; }

  // Embedding lookup: tokens (B, L) int32 -> (B, L, hidden) fp16.
  mx::array embed(const mx::array& tokens) const;

  // fast::rms_norm with the configured eps.
  mx::array rms_norm(const mx::array& x, const mx::array& weight) const;

  // Apply llama3 RoPE to x (B, n_heads, L, head_dim). The uniform overload uses
  // one position offset for the whole batch; the array overload takes a per-row
  // offset (B,) for ragged left-padded batches.
  mx::array apply_rope(const mx::array& x, int offset = 0) const;
  mx::array apply_rope(const mx::array& x, const mx::array& offset) const;

  // Front half of a decoder layer: input RMSNorm -> Q/K/V projections ->
  // reshape to heads -> RoPE on Q/K. V is returned un-roped. Each output is
  // (B, n_heads_{q,kv}, L, head_dim).
  struct QKV {
    mx::array q, k, v;
  };
  QKV attn_qkv(const mx::array& x, int layer, int offset = 0) const;

  // Self-attention sublayer: QKV (RoPE at `offset`) -> optional cache append ->
  // SDPA (causal for multi-token chunks, unmasked for single-token decode) ->
  // o_proj. Input/output are the residual-stream shape (B, L, hidden).
  mx::array attention(const mx::array& x, int layer, int offset = 0,
                      KVCache* cache = nullptr) const;

  // SwiGLU MLP sublayer: down(silu(gate(x)) * up(x)).
  mx::array mlp(const mx::array& x, int layer) const;

  // One full decoder layer: attention + MLP with residuals.
  mx::array decoder_block(const mx::array& x, int layer, int offset = 0,
                          KVCache* cache = nullptr) const;

  // Full forward pass: embedding -> n_layers decoder layers -> final RMSNorm ->
  // LM head (separate lm_head if present, else tied embedding). tokens (B, L) ->
  // logits (B, L, vocab). With a cache, RoPE/attention use cache.offset() and
  // the cache is appended and advanced by L (prefill once, then decode L=1).
  mx::array forward(const mx::array& tokens, KVCache* cache = nullptr) const;

  // Batched forward over a BatchKVCache: per-row RoPE offsets (cache.offset())
  // and a ragged additive fp16 mask drive correct left-padded attention. tokens
  // (B, L) -> logits (B, L, vocab). The whole batch is one graph (a single eval
  // by the caller realizes the step). Used by the continuous-batching scheduler.
  mx::array forward(const mx::array& tokens, BatchKVCache& cache) const;

  // Per-step additive fp16 attention mask [B, 1, N, T_kv] for a batched step:
  // 0 where a key is causally valid and not left-padding, -inf otherwise.
  mx::array batch_mask(int prev_idx, int n_query, const mx::array& left_padding) const;

 private:
  // input RMSNorm -> Q/K/V projections -> reshape to heads, WITHOUT RoPE.
  QKV project_qkv(const mx::array& x, int layer) const;
  // Batched self-attention: per-row RoPE offset, cache append, additive mask.
  mx::array attention_batched(const mx::array& x, int layer, const mx::array& offset,
                              const mx::array& mask, BatchKVCache& cache) const;

  // y = x @ W^T for an HF Linear weight W (out, in) stored under `weight_key`.
  mx::array linear(const mx::array& x, const std::string& weight_key) const;
  // Weight key for layer `i`, suffix e.g. "self_attn.q_proj.weight".
  std::string layer_key(int i, const std::string& suffix) const;
  // Norm/embedding weight for layer `i`, suffix e.g. "input_layernorm.weight".
  const mx::array& layer_w(int i, const std::string& suffix) const;

  ModelConfig cfg_;
  Weights w_;
  mx::array rope_freqs_;
  // Qwen3 normalizes each Q/K head (RMSNorm over head_dim) before RoPE. Detected
  // from the presence of the q_norm/k_norm weights; false for Llama-family models.
  bool has_qk_norm_ = false;
};

}  // namespace mlxforge
