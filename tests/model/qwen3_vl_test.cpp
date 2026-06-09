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
#include "support/model_fixture.h"
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

TEST_CASE("Qwen3-VL: layer-0 Q/K after interleaved M-RoPE match the reference") {
  if (!qwen3_vl_model_available()) {
    MESSAGE("Qwen3-VL model not found in HF cache; skipping M-RoPE Q/K test");
    return;
  }
  const Qwen3VLModel& m = shared_qwen3_vl_model();
  const int seq = static_cast<int>(load_qwen3_vl_token_ids("input_ids.npy").size());
  mx::array hidden = load_qwen3_vl_npy("embeds_merged.npy");  // (1, seq, hidden)
  mx::array pos = mx::reshape(load_qwen3_vl_npy("pos_ids.npy"), {3, seq});

  Qwen3VLModel::RopedQK qk = m.roped_qk(0, hidden, pos);
  mx::eval(qk.q, qk.k);
  assert_close(qk.q, load_qwen3_vl_npy("llm_q_rope0.npy"));
  assert_close(qk.k, load_qwen3_vl_npy("llm_k_rope0.npy"));
}

TEST_CASE("Qwen3-VL: full multimodal forward matches the reference logits") {
  if (!qwen3_vl_model_available()) {
    MESSAGE("Qwen3-VL model not found in HF cache; skipping multimodal forward test");
    return;
  }
  const Qwen3VLModel& m = shared_qwen3_vl_model();
  std::vector<int> ids = load_qwen3_vl_token_ids("input_ids.npy");

  // Use the committed ViT outputs as image features (the ViT is gated separately),
  // and the gated M-RoPE positions.
  mx::array feats = load_qwen3_vl_npy("vit_out.npy");
  std::vector<mx::array> deepstack = {load_qwen3_vl_npy("deepstack_0.npy"),
                                      load_qwen3_vl_npy("deepstack_1.npy"),
                                      load_qwen3_vl_npy("deepstack_2.npy")};
  ModelConfig cfg;
  cfg.image_token_id = 151655;
  VisionConfig vc;
  vc.spatial_merge_size = 2;
  cfg.vision = vc;
  mx::array pos = mrope_position_ids(ids, grid_fixture(), cfg);

  mx::array logits = m.forward_multimodal(ids, feats, deepstack, pos);  // (1, seq, vocab)
  const int seq = static_cast<int>(ids.size());
  const int vocab = logits.shape()[2];
  mx::array last = mx::reshape(mx::slice(logits, {0, seq - 1, 0}, {1, seq, vocab}), {1, vocab});
  mx::eval(last);

  // Exact next-token argmax (the primary end-to-end gate), plus a loose logits
  // check (36 fp16 layers accumulate well past the single-stage tolerance).
  std::vector<int> expect_argmax = load_qwen3_vl_token_ids("argmax.npy");
  std::vector<int> got = {static_cast<int>(mx::argmax(last, /*axis=*/-1).item<int>())};
  assert_tokens_equal(got, expect_argmax);
  assert_close(last, load_qwen3_vl_npy("logits_last.npy"), /*rtol=*/5e-2f, /*atol=*/5e-2f);
}
