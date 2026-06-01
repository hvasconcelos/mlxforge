// LLaMA decoder model on MLX. Built up across stories:
//   XLLM-006: embedding, RMSNorm, Q/K/V projections, RoPE (this file's start)
//   XLLM-007: single decoder block (attention + SwiGLU)
//   XLLM-008: full 16-layer stack -> logits
#pragma once

#include <string>

#include "mlx/array.h"

#include "core/config.h"
#include "core/weights.h"

namespace xllm {

namespace mx = mlx::core;

// Confirms the pinned MLX exposes fast::rope(const array& offset, ...) — the
// per-row offset overload that batched decode (XLLM-010+) depends on. Returns
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

  // Apply llama3 RoPE to x (B, n_heads, L, head_dim) at a uniform position offset.
  mx::array apply_rope(const mx::array& x, int offset = 0) const;

  // Front half of a decoder layer: input RMSNorm -> Q/K/V projections ->
  // reshape to heads -> RoPE on Q/K. V is returned un-roped. Each output is
  // (B, n_heads_{q,kv}, L, head_dim).
  struct QKV {
    mx::array q, k, v;
  };
  QKV attn_qkv(const mx::array& x, int layer, int offset = 0) const;

  // Self-attention sublayer (prefill, causal): QKV -> SDPA -> o_proj.
  // Input/output are the residual-stream shape (B, L, hidden).
  mx::array attention(const mx::array& x, int layer) const;

  // SwiGLU MLP sublayer: down(silu(gate(x)) * up(x)).
  mx::array mlp(const mx::array& x, int layer) const;

  // One full decoder layer (prefill, causal): attention + MLP with residuals.
  mx::array decoder_block(const mx::array& x, int layer) const;

 private:
  // y = x @ W^T for an HF Linear weight W (out, in) stored under `weight_key`.
  mx::array linear(const mx::array& x, const std::string& weight_key) const;
  // Weight key for layer `i`, suffix e.g. "self_attn.q_proj.weight".
  std::string layer_key(int i, const std::string& suffix) const;
  // Norm/embedding weight for layer `i`, suffix e.g. "input_layernorm.weight".
  const mx::array& layer_w(int i, const std::string& suffix) const;

  ModelConfig cfg_;
  Weights w_;
  mx::array rope_freqs_;
};

}  // namespace xllm
