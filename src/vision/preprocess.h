// Engine-side image preprocessing for the Qwen3-VL ViT.
//
// Turns a decoded RGB image into the model-ready `pixel_values` + patch grid the
// ViT consumes. This first stage covers rescale + normalize + patchify for an
// image whose dimensions are already a multiple of patch_size*merge_size (the
// smart-resize stage — which must match HF's bicubic resampling — lands next).
//
// The patch layout mirrors the HF Qwen3VLImageProcessor exactly: rows are in
// merged-block order (the order the ViT's RoPE / position embeds assume) and each
// row is the patch flattened as (channel, temporal, patch_h, patch_w).
#pragma once

#include <array>

#include "mlx/array.h"

#include "core/config.h"

namespace mlxforge {

namespace mx = mlx::core;

// Image normalization / patchification parameters. patch/temporal/merge come
// from the model's VisionConfig; mean/std/rescale come from the HF
// preprocessor_config (Qwen3-VL defaults shown).
struct PreprocessConfig {
  int patch_size = 16;
  int temporal_patch_size = 2;
  int merge_size = 2;
  float rescale_factor = 1.0f / 255.0f;
  std::array<float, 3> image_mean = {0.5f, 0.5f, 0.5f};
  std::array<float, 3> image_std = {0.5f, 0.5f, 0.5f};
  // Smart-resize bounds (total pixels). The image is resized so its area falls
  // within [min_pixels, max_pixels] and both sides are multiples of
  // patch_size*merge_size. Defaults mirror Qwen3-VL's preprocessor_config; lower
  // max_pixels to cap the image-token count (faster, less detail).
  int min_pixels = 256 * 256;   // 65536
  int max_pixels = 4096 * 4096; // 16777216

  // Patch/temporal/merge from the model config; normalization at Qwen3-VL defaults.
  static PreprocessConfig from(const VisionConfig& v) {
    PreprocessConfig c;
    c.patch_size = v.patch_size;
    c.temporal_patch_size = v.temporal_patch_size;
    c.merge_size = v.spatial_merge_size;
    return c;
  }
};

struct Preprocessed {
  mx::array pixel_values;        // (grid_h*grid_w, channels*tps*ps*ps) float32
  std::array<int, 3> grid_thw;   // (temporal, height, width) in patches
};

// Rescale + normalize + patchify an RGB image whose height and width are already
// multiples of patch_size*merge_size. `image_rgb` is (H, W, 3) uint8. Throws if
// the dimensions are not aligned (use preprocess_image to resize first).
Preprocessed patchify_image(const mx::array& image_rgb, const PreprocessConfig& cfg);

// Full preprocessing for an arbitrary RGB image: smart-resize (HF qwen2_vl
// algorithm) to an area within [min_pixels, max_pixels] with both sides a
// multiple of patch_size*merge_size, then patchify. `image_rgb` is (H, W, 3)
// uint8. The resize uses a cubic filter (stb) and is not bit-identical to HF's
// PIL bicubic, so results on non-aligned images may differ slightly from the
// reference; the aligned patchify path stays golden-exact.
Preprocessed preprocess_image(const mx::array& image_rgb, const PreprocessConfig& cfg);

// HF qwen2_vl smart_resize: round (height, width) to multiples of `factor`,
// rescaling so the area stays within [min_pixels, max_pixels]. Exposed for tests.
std::array<int, 2> smart_resize(int height, int width, int factor, int min_pixels,
                                int max_pixels);

// Number of <|image_pad|> placeholder tokens an image of (height, width) pixels
// expands to under `cfg`: the smart-resized patch grid collapsed by merge_size²
// (grid_t == 1 for a still image). Computed from dimensions alone, so a non-worker
// thread can size the chat-template expansion without decoding the image.
int image_token_count(int height, int width, const PreprocessConfig& cfg);

}  // namespace mlxforge
