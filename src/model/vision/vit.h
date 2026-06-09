// Qwen3-VL vision transformer (ViT) encoder.
//
// Consumes preprocessed image patches (pixel_values) + the patch grid (grid_thw)
// and produces the merged patch embeddings the language model attends over, plus
// the per-layer DeepStack features injected into the first decoder layers.
//
// Unlike the text decoder, the ViT weights are *unquantized* fp16 and every
// Linear / LayerNorm carries a bias. The encoder borrows the owning model's
// Weights (keys canonicalized to "visual.*" by sanitize_key) and is built up
// stage by stage, each gated against reference/fixtures_qwen3_vl:
//   patch_embed  -> vit_patch_embed   (Conv3d-as-matmul patchify)
//   pos_embed    -> vit_pos_embed     (interpolated learned position embeds)
//   rot_pos_emb  -> vit_rotary        (2D RoPE frequencies)
//   block(0)     -> vit_block0        (attention + MLP)
//   forward      -> vit_out, deepstack_{0,1,2}
#pragma once

#include "mlx/array.h"

#include "core/config.h"
#include "core/weights.h"

namespace mlxforge {

namespace mx = mlx::core;

class VitEncoder {
 public:
  // Borrows `weights` (the owning model outlives the encoder); copies the small
  // VisionConfig by value.
  VitEncoder(VisionConfig cfg, const Weights& weights);

  const VisionConfig& config() const { return cfg_; }

  // Conv3d patch embedding as a matmul: pixel_values (num_patches, in_ch *
  // temporal_patch * patch * patch) -> (num_patches, vit_hidden). The conv kernel
  // spans a whole patch, so it reduces to a linear projection of the (reordered)
  // flattened patch.
  mx::array patch_embed(const mx::array& pixel_values) const;

  // 2D RoPE frequency angles per patch, in the encoder's merged-block patch
  // order: grid_thw (num_images, 3) int -> (num_tokens, head_dim/2) float32.
  // Each patch's height/width grid positions index a shared inverse-frequency
  // table; the height and width angle halves are concatenated.
  mx::array rope_2d_freqs(const mx::array& grid_thw) const;

  // Interpolated learned position embeddings: grid_thw -> (num_tokens,
  // vit_hidden), in merged-block patch order. The square learned grid
  // (num_position_embeddings) is bilinearly resampled to each image's patch grid,
  // then permuted into the merger's block order.
  mx::array pos_embed(const mx::array& grid_thw) const;

  // One ViT transformer block (layer index `i`): norm1 -> 2D-RoPE attention ->
  // residual -> norm2 -> gelu-tanh MLP -> residual. `freqs` is rope_2d_freqs().
  // x is (num_tokens, vit_hidden). Gated (block 0) against vit_block0.
  mx::array block(const mx::array& x, int i, const mx::array& freqs) const;

 private:
  // y = x @ W^T + b for a ViT Linear stored under "<key>.weight" / "<key>.bias".
  mx::array linear(const mx::array& x, const std::string& key) const;
  // LayerNorm (weight + bias) under "<prefix>.weight" / "<prefix>.bias".
  mx::array layer_norm(const mx::array& x, const std::string& prefix) const;
  // Full self-attention over all patches (single image), 2D RoPE on Q/K.
  mx::array attention(const mx::array& x, int i, const mx::array& freqs) const;
  // gelu-tanh MLP: linear_fc2(gelu_tanh(linear_fc1(x))).
  mx::array vision_mlp(const mx::array& x, int i) const;
  // Weight-key prefix for block `i`, e.g. "visual.blocks.3".
  std::string block_key(int i) const;

  VisionConfig cfg_;
  const Weights& w_;
};

}  // namespace mlxforge
