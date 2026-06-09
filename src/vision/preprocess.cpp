#include "vision/preprocess.h"

#include <cmath>
#include <stdexcept>
#include <vector>

#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb_image_resize2.h"

#include "mlx/ops.h"
#include "mlx/transforms.h"

namespace mlxforge {

namespace {
// Bilinear/cubic resize of an (H, W, 3) uint8 image to (th, tw, 3). Resampling
// happens in raw pixel space (no sRGB linearization), the closest stb match to
// PIL's bicubic behavior.
mx::array resize_rgb(const mx::array& image_rgb, int th, int tw) {
  mx::array src = mx::contiguous(mx::astype(image_rgb, mx::uint8));
  mx::eval(src);
  const int H = src.shape()[0], W = src.shape()[1];
  std::vector<uint8_t> out(static_cast<size_t>(th) * tw * 3);
  stbir_resize_uint8_linear(src.data<uint8_t>(), W, H, /*in_stride=*/0, out.data(), tw, th,
                            /*out_stride=*/0, STBIR_RGB);
  return mx::array(out.data(), {th, tw, 3}, mx::uint8);
}
}  // namespace

std::array<int, 2> smart_resize(int height, int width, int factor, int min_pixels, int max_pixels) {
  const auto round_by = [factor](double v) {
    return static_cast<int>(std::lround(v / factor)) * factor;
  };
  int h_bar = round_by(height);
  int w_bar = round_by(width);
  if (static_cast<long long>(h_bar) * w_bar > max_pixels) {
    const double beta = std::sqrt(static_cast<double>(height) * width / max_pixels);
    h_bar = std::max(factor, static_cast<int>(std::floor(height / beta / factor)) * factor);
    w_bar = std::max(factor, static_cast<int>(std::floor(width / beta / factor)) * factor);
  } else if (static_cast<long long>(h_bar) * w_bar < min_pixels) {
    const double beta = std::sqrt(static_cast<double>(min_pixels) / (static_cast<double>(height) * width));
    h_bar = static_cast<int>(std::ceil(height * beta / factor)) * factor;
    w_bar = static_cast<int>(std::ceil(width * beta / factor)) * factor;
  }
  return {h_bar, w_bar};
}

Preprocessed patchify_image(const mx::array& image_rgb, const PreprocessConfig& cfg) {
  const int H = image_rgb.shape()[0];
  const int W = image_rgb.shape()[1];
  const int C = image_rgb.shape()[2];
  const int ps = cfg.patch_size, tps = cfg.temporal_patch_size, ms = cfg.merge_size;
  const int factor = ps * ms;
  if (H % factor != 0 || W % factor != 0) {
    throw std::runtime_error("patchify_image: image dimensions must be a multiple of "
                             "patch_size*merge_size (resize is a separate stage)");
  }
  const int gh = H / ps, gw = W / ps;

  // Rescale (uint8 -> [0,1]) and per-channel normalize: (x*rescale - mean) / std.
  mx::array img = mx::multiply(mx::astype(image_rgb, mx::float32), mx::array(cfg.rescale_factor));
  mx::array mean(cfg.image_mean.data(), {1, 1, C}, mx::float32);
  mx::array std(cfg.image_std.data(), {1, 1, C}, mx::float32);
  img = mx::divide(mx::subtract(img, mean), std);  // (H, W, C)

  // HWC -> CHW, then duplicate along the temporal axis (a still image fills all
  // temporal_patch_size frames). Shape: (1, 1, tps, C, H, W).
  img = mx::transpose(img, {2, 0, 1});  // (C, H, W)
  img = mx::broadcast_to(mx::reshape(img, {1, 1, 1, C, H, W}), {1, 1, tps, C, H, W});

  // Split H -> (gh/ms, ms, ps) and W -> (gw/ms, ms, ps), then permute into
  // merged-block order with each row flattened as (C, tps, ps, ps). Mirrors the
  // HF Qwen3VLImageProcessor reshape/transpose exactly.
  img = mx::reshape(img, {1, 1, tps, C, gh / ms, ms, ps, gw / ms, ms, ps});
  img = mx::transpose(img, {0, 1, 4, 7, 5, 8, 3, 2, 6, 9});
  mx::array pixel_values = mx::reshape(img, {gh * gw, C * tps * ps * ps});

  return {pixel_values, {1, gh, gw}};
}

int image_token_count(int height, int width, const PreprocessConfig& cfg) {
  const int factor = cfg.patch_size * cfg.merge_size;
  const std::array<int, 2> hw = smart_resize(height, width, factor, cfg.min_pixels, cfg.max_pixels);
  const int gh = hw[0] / cfg.patch_size, gw = hw[1] / cfg.patch_size;  // patch grid
  return (gh * gw) / (cfg.merge_size * cfg.merge_size);                // grid_t == 1
}

Preprocessed preprocess_image(const mx::array& image_rgb, const PreprocessConfig& cfg) {
  const int H = image_rgb.shape()[0], W = image_rgb.shape()[1];
  const int factor = cfg.patch_size * cfg.merge_size;
  std::array<int, 2> hw = smart_resize(H, W, factor, cfg.min_pixels, cfg.max_pixels);
  const mx::array resized = (hw[0] == H && hw[1] == W) ? image_rgb : resize_rgb(image_rgb, hw[0], hw[1]);
  return patchify_image(resized, cfg);
}

}  // namespace mlxforge
