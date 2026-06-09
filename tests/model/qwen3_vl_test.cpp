// Qwen3-VL fusion stages (M-RoPE positions, image merge, DeepStack), gated
// against reference/fixtures_qwen3_vl. The M-RoPE position test is pure logic
// (committed fixtures only); the rest self-skip without the model snapshot.
#include <doctest/doctest.h>

#include <array>
#include <cstdint>
#include <vector>

#include "mlx/ops.h"
#include "mlx/transforms.h"

#include "core/config.h"
#include "model/qwen3_vl.h"
#include "support/reference.h"

using namespace mlxforge;
using namespace mlxforge::test;

namespace {
// Read image_grid_thw.npy to a host list of (t, h, w).
std::vector<std::array<int, 3>> grid_fixture() {
  mx::array g = mx::contiguous(mx::astype(load_qwen3_vl_npy("image_grid_thw.npy"), mx::int32));
  mx::eval(g);
  const int32_t* p = g.data<int32_t>();
  std::vector<std::array<int, 3>> grids;
  for (int i = 0; i < g.shape()[0]; ++i) grids.push_back({p[i * 3], p[i * 3 + 1], p[i * 3 + 2]});
  return grids;
}
}  // namespace

TEST_CASE("Qwen3-VL: interleaved M-RoPE position ids match the reference") {
  // Pure logic: only the committed input_ids/grid fixtures + the fields
  // mrope_position_ids reads (image_token_id, spatial_merge_size).
  std::vector<int> ids = load_qwen3_vl_token_ids("input_ids.npy");
  ModelConfig cfg;
  cfg.image_token_id = 151655;
  VisionConfig vc;
  vc.spatial_merge_size = 2;
  cfg.vision = vc;

  mx::array pos = mrope_position_ids(ids, grid_fixture(), cfg);
  mx::eval(pos);

  // pos_ids fixture is (3, 1, seq); compare against (3, seq).
  mx::array expected = mx::reshape(load_qwen3_vl_npy("pos_ids.npy"), {3, static_cast<int>(ids.size())});
  assert_close(mx::astype(pos, mx::float32), mx::astype(expected, mx::float32));
}

TEST_CASE("Qwen3-VL: image-feature merge matches the reference") {
  // Pure fixture logic: embeds_text (token embeddings) + vit_out (image features)
  // scattered into the image_pad rows must equal embeds_merged.
  std::vector<int> ids = load_qwen3_vl_token_ids("input_ids.npy");
  const int seq = static_cast<int>(ids.size());
  mx::array text = mx::reshape(load_qwen3_vl_npy("embeds_text.npy"), {seq, -1});
  mx::array feats = load_qwen3_vl_npy("vit_out.npy");

  mx::array merged = merge_image_features(text, feats, ids, /*image_token_id=*/151655);
  mx::eval(merged);

  mx::array expected = mx::reshape(load_qwen3_vl_npy("embeds_merged.npy"), {seq, -1});
  assert_close(merged, expected);
}
