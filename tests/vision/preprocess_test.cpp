// Engine-side image preprocessing, gated against reference/fixtures_qwen3_vl.
// Pure logic: the committed image_rgb -> pixel_values, no model needed.
#include <doctest/doctest.h>

#include <cstdint>
#include <fstream>
#include <iterator>
#include <vector>

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

TEST_CASE("Qwen3-VL smart_resize rounds and rescales like the HF reference") {
  const int F = 32;  // patch_size(16) * merge_size(2)
  const int MIN = 256 * 256, MAX = 4096 * 4096;

  // Already a multiple of F and in range -> only rounding (here, identity).
  CHECK(smart_resize(800, 1216, F, MIN, MAX) == std::array<int, 2>{800, 1216});
  // Not a multiple of F -> round each side to the nearest multiple.
  CHECK(smart_resize(805, 1200, F, MIN, MAX) == std::array<int, 2>{800, 1216});
  // Below min_pixels -> upscale (64x64 = 4096 px < 65536).
  CHECK(smart_resize(64, 64, F, MIN, MAX) == std::array<int, 2>{256, 256});
  // Above max_pixels -> downscale to the cap.
  CHECK(smart_resize(5000, 5000, F, MIN, MAX) == std::array<int, 2>{4096, 4096});
  // The fixture's small bounds leave a 64x64 image untouched.
  CHECK(smart_resize(64, 64, F, 256, 4096) == std::array<int, 2>{64, 64});
}

TEST_CASE("Qwen3-VL image_info reads dimensions without decoding") {
  std::ifstream f(qwen3_vl_ref_path("image.png"), std::ios::binary);
  std::vector<std::uint8_t> bytes((std::istreambuf_iterator<char>(f)),
                                  std::istreambuf_iterator<char>());
  std::array<int, 2> hw = image_info(bytes.data(), bytes.size());
  CHECK(hw[0] == 64);  // height
  CHECK(hw[1] == 64);  // width
}

TEST_CASE("Qwen3-VL image_token_count from dimensions matches the grid") {
  PreprocessConfig prod;  // production defaults (min 65536, max 16777216)
  // 64x64 is below min_pixels -> upscaled to 256x256 -> 16x16 patches -> /merge².
  CHECK(image_token_count(64, 64, prod) == 64);
  PreprocessConfig tiny = prod;
  tiny.min_pixels = 256;
  tiny.max_pixels = 4096;
  // The fixtures' bounds leave 64x64 untouched -> 4x4 patches -> /merge².
  CHECK(image_token_count(64, 64, tiny) == 4);
}

TEST_CASE("Qwen3-VL image decode: PNG file decodes to the reference RGB") {
  // image.png and image_rgb.npy are the same picture (PNG is lossless), so the
  // decoded pixels must match the committed RGB array exactly.
  mx::array decoded = decode_image_file(qwen3_vl_ref_path("image.png"));
  mx::eval(decoded);
  CHECK(decoded.shape() == mx::Shape{64, 64, 3});
  assert_close(decoded, load_qwen3_vl_npy("image_rgb.npy"), /*rtol=*/0.0f, /*atol=*/0.0f);
}
