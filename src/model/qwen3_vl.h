// Qwen3-VL vision-language model: the Qwen3 dense text decoder fused with the ViT
// vision tower. Two deltas over the text path, both confined to prefill:
//   - interleaved M-RoPE: image tokens carry distinct 3D (temporal, height,
//     width) rotary positions; text tokens have t==h==w (so M-RoPE collapses to
//     ordinary 1D RoPE there).
//   - DeepStack: per-layer ViT features added into the first decoder layers at
//     the image-token positions.
// This header currently exposes the M-RoPE position computation; the model class
// and the fused forward land in subsequent steps.
#pragma once

#include <array>
#include <utility>
#include <vector>

#include "mlx/array.h"

#include "core/config.h"
#include "core/weights.h"
#include "model/qwen3.h"

namespace mlxforge {

namespace mx = mlx::core;

// Interleaved-M-RoPE 3D position ids for one sequence (no left padding):
// input_ids (host) + per-image (t, h, w) patch grids -> (3, seq) int32, the
// temporal/height/width position rows. Text tokens get equal positions
// incrementing by one; an image run lays its (t, h/merge, w/merge) grid out in
// (frame, height, width) order at the running base, then the base advances by
// the largest of those three extents. Video tokens are not yet handled.
mx::array mrope_position_ids(const std::vector<int>& input_ids,
                             const std::vector<std::array<int, 3>>& image_grids,
                             const ModelConfig& cfg);

// Scatter ViT features into the image-placeholder rows of the token embeddings.
// token_embeds (seq, hidden) fp16; image_features (num_image_tokens, hidden) in
// sequence order; input_ids identify the image_token_id positions. Returns the
// merged (seq, hidden). Non-image rows are the original embeddings.
mx::array merge_image_features(const mx::array& token_embeds, const mx::array& image_features,
                               const std::vector<int>& input_ids, int image_token_id);

// Qwen3-VL model: a Qwen3 dense decoder (QK-Norm, inherited) with interleaved
// M-RoPE in place of 1D RoPE and DeepStack injection into the first decoder
// layers. The text path (attention/MLP/QK-norm) is reused from Qwen3Model; only
// the position embedding and the per-layer vision injection differ.
class Qwen3VLModel : public Qwen3Model {
 public:
  Qwen3VLModel(ModelConfig config, Weights weights);

  struct RopedQK {
    mx::array q, k;  // (1, n_heads/n_kv_heads, seq, head_dim) after M-RoPE
  };
  // Layer `i` Q/K after QK-Norm and interleaved M-RoPE. hidden (1, seq, hidden),
  // position_ids (3, seq). Exposed for golden gating (llm_q_rope0 / llm_k_rope0).
  RopedQK roped_qk(int i, const mx::array& hidden, const mx::array& position_ids) const;

  // Full multimodal prefill for a single sequence: token ids + merged ViT
  // features (scattered into the image_pad rows) + DeepStack features + 3D M-RoPE
  // position ids -> logits (1, seq, vocab). DeepStack feature j is added at the
  // image rows after decoder layer j.
  mx::array forward_multimodal(const std::vector<int>& input_ids,
                               const mx::array& image_features,
                               const std::vector<mx::array>& deepstack,
                               const mx::array& position_ids) const;

 private:
  // cos/sin for interleaved M-RoPE: position_ids (3, seq) -> each (seq, head_dim/2)
  // float32. Frequency d uses the axis selected by mrope_selector_.
  std::pair<mx::array, mx::array> mrope_cos_sin(const mx::array& position_ids) const;
  // Apply half-split rotation with the M-RoPE cos/sin to x (B, heads, seq, hd).
  mx::array apply_mrope(const mx::array& x, const mx::array& cos, const mx::array& sin) const;
  // Self-attention for layer `i` with M-RoPE on Q/K (causal SDPA, GQA-native).
  mx::array mm_attention(int i, const mx::array& x, const mx::array& cos,
                         const mx::array& sin) const;

  mx::array inv_freq_;        // (head_dim/2,) float32 inverse frequencies
  mx::array mrope_selector_;  // (head_dim/2,) int32: t/h/w axis per frequency
};

}  // namespace mlxforge
