// Qwen3-VL fusion stages (M-RoPE positions, image merge, DeepStack), gated
// against reference/fixtures_qwen3_vl. The M-RoPE position test is pure logic
// (committed fixtures only); the rest self-skip without the model snapshot.
#include <doctest/doctest.h>

#include <array>
#include <cstdint>
#include <fstream>
#include <memory>
#include <vector>

#include "mlx/ops.h"
#include "mlx/transforms.h"

#include "cache/kv_cache.h"
#include "core/config.h"
#include "core/weights.h"
#include "model/model_factory.h"
#include "model/qwen3_vl.h"
#include "runtime/multimodal_stream.h"
#include "runtime/worker.h"
#include "scheduler/request.h"
#include "scheduler/scheduler.h"
#include "tokenizer/tokenizer.h"
#include "vision/preprocess.h"
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

TEST_CASE("Qwen3-VL: model factory dispatches a vision config to Qwen3VLModel") {
  if (!qwen3_vl_model_available()) {
    MESSAGE("Qwen3-VL model not found in HF cache; skipping factory-dispatch test");
    return;
  }
  ModelConfig cfg = ModelConfig::from_file(qwen3_vl_model_dir() + "/config.json");
  REQUIRE(cfg.has_vision_tower());
  std::unique_ptr<DecoderModel> model = create_model(cfg, load_weights(qwen3_vl_model_dir(), cfg));
  CHECK(dynamic_cast<Qwen3VLModel*>(model.get()) != nullptr);
}

TEST_CASE("Qwen3-VL: end-to-end from RGB pixels to next-token argmax") {
  if (!qwen3_vl_model_available()) {
    MESSAGE("Qwen3-VL model not found in HF cache; skipping end-to-end test");
    return;
  }
  // The whole engine-side vision chain composed: preprocess -> ViT -> merge +
  // M-RoPE + DeepStack -> logits. (Only the prompt ids come from a fixture, since
  // the multimodal chat template lands in a later phase.)
  const VitEncoder& vit = shared_qwen3_vl_vit();
  const Qwen3VLModel& m = shared_qwen3_vl_model();

  PreprocessConfig pcfg = PreprocessConfig::from(*qwen3_vl_config().vision);
  Preprocessed pre = patchify_image(load_qwen3_vl_npy("image_rgb.npy"), pcfg);
  mx::array grid(pre.grid_thw.data(), {1, 3}, mx::int32);

  VitEncoder::Output v = vit.forward(pre.pixel_values, grid);

  std::vector<int> ids = load_qwen3_vl_token_ids("input_ids.npy");
  std::vector<std::array<int, 3>> grids = {pre.grid_thw};
  mx::array pos = mrope_position_ids(ids, grids, qwen3_vl_config());

  mx::array logits = m.forward_multimodal(ids, v.hidden, v.deepstack, pos);
  const int seq = static_cast<int>(ids.size()), vocab = logits.shape()[2];
  mx::array last = mx::reshape(mx::slice(logits, {0, seq - 1, 0}, {1, seq, vocab}), {1, vocab});

  std::vector<int> got = {static_cast<int>(mx::argmax(last, -1).item<int>())};
  assert_tokens_equal(got, load_qwen3_vl_token_ids("argmax.npy"));
}

TEST_CASE("Qwen3-VL: chat template with an image matches the reference input_ids") {
  if (!qwen3_vl_model_available()) {
    MESSAGE("Qwen3-VL model not found in HF cache; skipping chat-template test");
    return;
  }
  // ChatML with one image whose grid (1,4,4) collapses to 4 <|image_pad|> tokens.
  Tokenizer tok = Tokenizer::from_file(qwen3_vl_model_dir() + "/tokenizer.json", /*bos_id=*/-1,
                                       ChatFormat::Qwen3);
  Tokenizer::Message m;
  m.role = "user";
  m.content = "What is in this image?";
  m.image_token_counts = {4};

  std::vector<int> ids = tok.apply_chat_template({m}, /*add_generation_prompt=*/true);
  assert_tokens_equal(ids, load_qwen3_vl_token_ids("input_ids.npy"));
}

TEST_CASE("Qwen3-VL: greedy multimodal generation matches the reference") {
  if (!qwen3_vl_model_available()) {
    MESSAGE("Qwen3-VL model not found in HF cache; skipping greedy generation test");
    return;
  }
  // Full-recompute greedy loop (the cached decode path is a later optimization):
  // each step re-runs the multimodal forward over the growing sequence with the
  // fixed image features, recomputing M-RoPE positions. Must match the reference
  // token-for-token.
  const Qwen3VLModel& m = shared_qwen3_vl_model();
  mx::array feats = load_qwen3_vl_npy("vit_out.npy");
  std::vector<mx::array> deepstack = {load_qwen3_vl_npy("deepstack_0.npy"),
                                      load_qwen3_vl_npy("deepstack_1.npy"),
                                      load_qwen3_vl_npy("deepstack_2.npy")};
  std::vector<std::array<int, 3>> grids = grid_fixture();

  std::vector<int> ids = load_qwen3_vl_token_ids("input_ids.npy");
  std::vector<int> greedy;
  for (int step = 0; step < 10; ++step) {
    mx::array pos = mrope_position_ids(ids, grids, qwen3_vl_config());
    mx::array logits = m.forward_multimodal(ids, feats, deepstack, pos);
    const int seq = static_cast<int>(ids.size()), vocab = logits.shape()[2];
    mx::array last = mx::reshape(mx::slice(logits, {0, seq - 1, 0}, {1, seq, vocab}), {1, vocab});
    const int nxt = static_cast<int>(mx::argmax(last, -1).item<int>());
    greedy.push_back(nxt);
    ids.push_back(nxt);
  }
  assert_tokens_equal(greedy, load_qwen3_vl_token_ids("greedy_tokens.npy"));
}

TEST_CASE("Qwen3-VL: greedy_generate_multimodal runtime path reproduces the greedy tokens") {
  if (!qwen3_vl_model_available()) {
    MESSAGE("Qwen3-VL model not found in HF cache; skipping multimodal runtime test");
    return;
  }
  // The runtime entrypoint (the CLI's multimodal core), end to end.
  const Qwen3VLModel& m = shared_qwen3_vl_model();
  mx::array feats = load_qwen3_vl_npy("vit_out.npy");
  std::vector<mx::array> deepstack = {load_qwen3_vl_npy("deepstack_0.npy"),
                                      load_qwen3_vl_npy("deepstack_1.npy"),
                                      load_qwen3_vl_npy("deepstack_2.npy")};
  std::vector<int> ids = load_qwen3_vl_token_ids("input_ids.npy");
  mx::array pos = mrope_position_ids(ids, grid_fixture(), qwen3_vl_config());

  // No EOS stop so the run matches the fixed-length reference greedy stream.
  GenerateResult r = greedy_generate_multimodal(m, ids, feats, deepstack, pos,
                                                /*max_tokens=*/10, /*eos_ids=*/{});
  assert_tokens_equal(r.tokens, load_qwen3_vl_token_ids("greedy_tokens.npy"));
}

TEST_CASE("Qwen3-VL: generate_from_image composes the full pipeline") {
  if (!qwen3_vl_model_available()) {
    MESSAGE("Qwen3-VL model not found in HF cache; skipping generate_from_image test");
    return;
  }
  // Everything through real entrypoints: preprocess -> ViT -> chat template (with
  // the computed image-token count) -> M-RoPE -> generate, from raw RGB + text.
  Tokenizer tok = Tokenizer::from_file(qwen3_vl_model_dir() + "/tokenizer.json", /*bos_id=*/-1,
                                       ChatFormat::Qwen3);
  GenerateResult r = generate_from_image(shared_qwen3_vl_model(), shared_qwen3_vl_vit(), tok,
                                         "What is in this image?", load_qwen3_vl_npy("image_rgb.npy"),
                                         /*max_tokens=*/10, /*eos_ids=*/{});
  assert_tokens_equal(r.tokens, load_qwen3_vl_token_ids("greedy_tokens.npy"));
}

TEST_CASE("Qwen3-VL: worker serves a multimodal request from another thread") {
  if (!qwen3_vl_model_available()) {
    MESSAGE("Qwen3-VL model not found in HF cache; skipping multimodal worker test");
    return;
  }
  // End-to-end through the scheduler/worker: a cross-thread multimodal request
  // (text + raw image bytes) is served single-stream on the worker thread and
  // streams the reference greedy tokens.
  const std::string dir = qwen3_vl_model_dir();
  Tokenizer tok = Tokenizer::from_file(dir + "/tokenizer.json", /*bos_id=*/-1, ChatFormat::Qwen3);

  mlxforge::Scheduler sched;
  mlxforge::Worker worker(
      [dir] {
        mlxforge::ModelConfig c = mlxforge::ModelConfig::from_file(dir + "/config.json");
        return mlxforge::create_model(c, mlxforge::load_weights(dir, c));
      },
      &sched, &tok);
  worker.start();

  // Load the raw PNG bytes for the request.
  std::ifstream f(qwen3_vl_ref_path("image.png"), std::ios::binary);
  std::vector<std::uint8_t> img_bytes((std::istreambuf_iterator<char>(f)),
                                      std::istreambuf_iterator<char>());

  auto req = std::make_shared<mlxforge::Request>();
  req->mm_text = "What is in this image?";
  req->mm_image = img_bytes;
  req->max_tokens = 10;
  // Empty eos_ids so the run matches the fixed-length reference greedy stream.
  sched.submit(req);

  std::vector<int> got;
  int t = 0;
  while (req->tokens.pop(t)) got.push_back(t);
  worker.stop();

  assert_tokens_equal(got, load_qwen3_vl_token_ids("greedy_tokens.npy"));
  CHECK(req->finish_reason == "length");
}

TEST_CASE("Qwen3-VL: cached KV decode reproduces the greedy tokens") {
  if (!qwen3_vl_model_available()) {
    MESSAGE("Qwen3-VL model not found in HF cache; skipping cached-decode test");
    return;
  }
  // The efficient path: prefill once into a KV cache, then single-token decode
  // steps at the post-image M-RoPE positions. Must match the full-recompute
  // greedy stream token-for-token.
  const Qwen3VLModel& m = shared_qwen3_vl_model();
  mx::array feats = load_qwen3_vl_npy("vit_out.npy");
  std::vector<mx::array> deepstack = {load_qwen3_vl_npy("deepstack_0.npy"),
                                      load_qwen3_vl_npy("deepstack_1.npy"),
                                      load_qwen3_vl_npy("deepstack_2.npy")};
  std::vector<int> ids = load_qwen3_vl_token_ids("input_ids.npy");
  mx::array pos = mrope_position_ids(ids, grid_fixture(), qwen3_vl_config());

  // The first generated token sits one past the prompt's max M-RoPE position.
  int next_pos = static_cast<int>(mx::max(pos).item<int>()) + 1;

  KVCache cache(qwen3_vl_config().n_layers);
  mx::array prefill_logits = m.prefill(ids, feats, deepstack, pos, cache);
  const int seq = static_cast<int>(ids.size()), vocab = prefill_logits.shape()[2];
  mx::array last = mx::reshape(mx::slice(prefill_logits, {0, seq - 1, 0}, {1, seq, vocab}), {1, vocab});

  int tok = static_cast<int>(mx::argmax(last, -1).item<int>());
  std::vector<int> greedy = {tok};
  for (int i = 1; i < 10; ++i) {
    mx::array logits = m.decode_step(tok, next_pos++, cache);
    tok = static_cast<int>(mx::argmax(logits, -1).item<int>());
    greedy.push_back(tok);
  }
  assert_tokens_equal(greedy, load_qwen3_vl_token_ids("greedy_tokens.npy"));
}
