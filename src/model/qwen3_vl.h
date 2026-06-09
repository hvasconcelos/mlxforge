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
#include <vector>

#include "mlx/array.h"

#include "core/config.h"

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

}  // namespace mlxforge
