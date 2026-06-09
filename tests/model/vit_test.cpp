// Qwen3-VL ViT encoder, gated stage by stage against reference/fixtures_qwen3_vl
// (dumped from mlx-vlm on the same 4bit checkpoint). Self-skips when the model
// snapshot is absent — a green run without it only covers the pure-logic units.
#include <doctest/doctest.h>

#include "mlx/ops.h"

#include "support/model_fixture.h"
#include "support/reference.h"

using namespace mlxforge;
using namespace mlxforge::test;

TEST_CASE("Qwen3-VL ViT: patch embedding matches the reference") {
  if (!qwen3_vl_model_available()) {
    MESSAGE("Qwen3-VL model not found in HF cache; skipping ViT patch-embed test");
    return;
  }
  const VitEncoder& vit = shared_qwen3_vl_vit();

  // pixel_values (num_patches, in_ch*temporal*patch*patch) -> patch embeddings.
  mx::array pixel_values = load_qwen3_vl_npy("pixel_values.npy");
  mx::array pe = vit.patch_embed(pixel_values);
  mx::eval(pe);

  assert_close(pe, load_qwen3_vl_npy("vit_patch_embed.npy"));
}

TEST_CASE("Qwen3-VL ViT: 2D RoPE frequencies match the reference") {
  if (!qwen3_vl_model_available()) {
    MESSAGE("Qwen3-VL model not found in HF cache; skipping ViT rope test");
    return;
  }
  const VitEncoder& vit = shared_qwen3_vl_vit();
  mx::array rot = vit.rope_2d_freqs(load_qwen3_vl_npy("image_grid_thw.npy"));
  mx::eval(rot);
  assert_close(rot, load_qwen3_vl_npy("vit_rotary.npy"));
}

TEST_CASE("Qwen3-VL ViT: interpolated position embeddings match the reference") {
  if (!qwen3_vl_model_available()) {
    MESSAGE("Qwen3-VL model not found in HF cache; skipping ViT pos-embed test");
    return;
  }
  const VitEncoder& vit = shared_qwen3_vl_vit();
  mx::array pos = vit.pos_embed(load_qwen3_vl_npy("image_grid_thw.npy"));
  mx::eval(pos);
  assert_close(pos, load_qwen3_vl_npy("vit_pos_embed.npy"));
}

TEST_CASE("Qwen3-VL ViT: block 0 (attention + MLP) matches the reference") {
  if (!qwen3_vl_model_available()) {
    MESSAGE("Qwen3-VL model not found in HF cache; skipping ViT block-0 test");
    return;
  }
  const VitEncoder& vit = shared_qwen3_vl_vit();
  mx::array grid = load_qwen3_vl_npy("image_grid_thw.npy");
  // Block input is patch_embed + interpolated pos_embed, in the patch dtype (fp16).
  mx::array hs = mx::add(vit.patch_embed(load_qwen3_vl_npy("pixel_values.npy")), vit.pos_embed(grid));
  mx::array out = vit.block(hs, 0, vit.rope_2d_freqs(grid));
  mx::eval(out);
  assert_close(out, load_qwen3_vl_npy("vit_block0.npy"));
}
