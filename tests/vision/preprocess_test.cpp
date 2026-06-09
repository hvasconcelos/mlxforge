// Engine-side image preprocessing, gated against reference/fixtures_qwen3_vl.
// Pure logic: the committed image_rgb -> pixel_values, no model needed.
#include <doctest/doctest.h>

#include "mlx/ops.h"
#include "mlx/transforms.h"

#include "vision/image_decode.h"
#include "vision/preprocess.h"
#include "support/reference.h"

using namespace mlxforge;
using namespace mlxforge::test;

TEST_CASE("Qwen3-VL preprocessing: normalize + patchify matches the reference") {
  // image_rgb is the decoded 64x64 RGB input; the fixture used a 64x64 image with
  // an identity resize, so this gates rescale + normalize + patchify.
  mx::array image_rgb = load_qwen3_vl_npy("image_rgb.npy");  // (64, 64, 3) uint8

  PreprocessConfig cfg;  // Qwen3-VL defaults (patch 16, temporal 2, merge 2, mean/std 0.5)
  Preprocessed out = patchify_image(image_rgb, cfg);
  mx::eval(out.pixel_values);

  CHECK(out.grid_thw[0] == 1);
  CHECK(out.grid_thw[1] == 4);
  CHECK(out.grid_thw[2] == 4);
  assert_close(out.pixel_values, load_qwen3_vl_npy("pixel_values.npy"));
}

TEST_CASE("Qwen3-VL image decode: PNG file decodes to the reference RGB") {
  // image.png and image_rgb.npy are the same picture (PNG is lossless), so the
  // decoded pixels must match the committed RGB array exactly.
  mx::array decoded = decode_image_file(qwen3_vl_ref_path("image.png"));
  mx::eval(decoded);
  CHECK(decoded.shape() == mx::Shape{64, 64, 3});
  assert_close(decoded, load_qwen3_vl_npy("image_rgb.npy"), /*rtol=*/0.0f, /*atol=*/0.0f);
}
