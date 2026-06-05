// Qwen3.5 hybrid decoder model (text tower only).
//
// Qwen3.5 interleaves two token-mixing families (config().is_linear_layer):
//   - full (softmax) attention every full_attention_interval-th layer, and
//   - Gated-DeltaNet linear attention on the rest.
// This file implements the full-attention family; it extends Qwen3 attention
// (per-head QK-Norm) with two deltas:
//   - a sigmoid output gate: q_proj is 2x width (queries||gate) and the attention
//     output is scaled by sigmoid(gate) before o_proj (attn_output_gate), and
//   - partial RoPE: only the leading int(head_dim * partial_rotary_factor)
//     dimensions are rotated (plain RoPE, base rope_theta; mrope is a no-op for
//     text), the tail passes through un-rotated.
// The MLP is dense SwiGLU (feed_forward() inherited), so decoder_block() drives
// the full-attention layers unchanged. The Gated-DeltaNet linear layers and the
// mixed (KV + recurrent state) cache are added in later phases.
#pragma once

#include <optional>
#include <utility>

#include "model/decoder_model.h"

namespace mlxforge {

class Qwen35Model : public DecoderModel {
 public:
  Qwen35Model(ModelConfig config, Weights weights)
      : DecoderModel(std::move(config), std::move(weights)),
        rotary_dim_(static_cast<int>(static_cast<float>(this->config().head_dim) *
                                     this->config().partial_rotary_factor)) {}

  // Gated, partial-RoPE full-attention sublayer (the Qwen3.5 full-attention
  // delta). Same residual-stream signature as DecoderModel::attention(); used on
  // the full-attention layers (config().is_linear_layer(layer) == false).
  // Public: a golden-reference test exercises it in isolation.
  mx::array attention(const mx::array& x, int layer, int offset = 0,
                      KVCache* cache = nullptr) const override;

  // Gated-DeltaNet linear-attention sublayer (used on the linear layers,
  // config().is_linear_layer(layer) == true). Input RMSNorm -> in_proj_{qkv,a,b,z}
  // -> causal depthwise Conv1d -> SiLU -> L2/RMS-normed Q/K -> delta-rule
  // recurrence with exponential gating -> gated RMSNorm(z) -> out_proj. Same
  // residual-stream signature as attention() (the input RMSNorm is bundled in).
  // With a cache it reads/writes that layer's conv buffer and recurrent state, so
  // chunked prefill and single-token decode carry state across calls. Public:
  // golden-reference tested.
  mx::array linear_attention(const mx::array& x, int layer, KVCache* cache = nullptr) const;

  // One decoder layer, routed by config().is_linear_layer(layer): the
  // Gated-DeltaNet sublayer on linear layers, the gated attention() on full
  // layers, then the shared residual + dense SwiGLU MLP. The KVCache carries the
  // linear layers' conv/recurrent state for single-sequence streaming decode.
  mx::array decoder_block(const mx::array& x, int layer, int offset = 0,
                          KVCache* cache = nullptr) const override;

  // Batched hybrid forward for the continuous-batching scheduler. Dispatches each
  // layer like decoder_block(), but batched: full layers use per-row RoPE offsets
  // + an additive KV mask; linear layers carry per-row conv/recurrent state in the
  // BatchKVCache and apply the left-padding ssm mask. tokens (B, L) -> logits.
  mx::array forward(const mx::array& tokens, BatchKVCache& cache) const override;

  // Un-hide the inherited single-sequence / no-cache forward() overloads, which
  // declaring forward(BatchKVCache&) above would otherwise shadow.
  using DecoderModel::forward;

  // Number of head dimensions rotated by RoPE (the rest pass through un-rotated).
  int rotary_dim() const { return rotary_dim_; }

 private:
  // Plain RoPE over the leading rotary_dim_ dims, base rope_theta. mrope is a
  // no-op for 1-D text positions, so this matches mlx_lm's nn.RoPE(rotary_dim).
  // The array overload takes a per-row offset (B,) for ragged left-padded batches.
  mx::array partial_rope(const mx::array& x, int offset) const;
  mx::array partial_rope(const mx::array& x, const mx::array& offset) const;

  // Batched gated, partial-RoPE full attention: like attention() but per-row RoPE
  // offset + additive KV mask, appending to the BatchKVCache. `real_pos` (B, L)
  // bool zeroes fully-padded query rows (whose SDPA softmax over an all -inf mask
  // is NaN). (B, L, hidden) -> (B, L, hidden).
  mx::array attention_batched_gated(const mx::array& x, int layer, const mx::array& offset,
                                    const mx::array& mask, const mx::array& real_pos,
                                    BatchKVCache& cache) const;

  // The shared Gated-DeltaNet sublayer body. Seeds the conv buffer and recurrent
  // state from init_conv/init_recur (nullopt -> zeros) and writes the updated
  // states back through out_conv/out_recur, so the single-sequence (KVCache) and
  // batched (BatchKVCache) callers differ only in how they fetch/store state and
  // build the left-padding mask. `mask` (B, S) bool zeroes/freezes padded steps.
  mx::array linear_attention_impl(const mx::array& x, int layer,
                                  const std::optional<mx::array>& init_conv,
                                  const std::optional<mx::array>& init_recur,
                                  const std::optional<mx::array>& mask, mx::array& out_conv,
                                  mx::array& out_recur) const;

  // The delta-rule recurrence, run sequentially over the S timesteps. q/k/v are
  // (B, S, H, D*), g (decay) and beta are (B, S, Hv). Returns the output
  // (B, S, Hv, Dv) and the final recurrent state (B, Hv, Dv, Dk), kept in fp32.
  // `init_state` seeds the recurrence (nullopt -> zeros, the prefill start).
  // `mask` (B, S) bool, when present, zeroes padded steps' contribution and
  // freezes the state across them (ragged left-padded batches).
  std::pair<mx::array, mx::array> gated_delta(const mx::array& q, const mx::array& k,
                                              const mx::array& v, const mx::array& g,
                                              const mx::array& beta,
                                              const std::optional<mx::array>& init_state,
                                              const std::optional<mx::array>& mask) const;

  int rotary_dim_;
};

}  // namespace mlxforge
