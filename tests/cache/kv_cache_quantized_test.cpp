// Quantized-KV golden gates. The correctness check for the quantized cache +
// quantized SDPA: a teacher-forced greedy walk over mlx-lm's QuantizedKVCache
// reference stream (fixtures greedy_tokens_kvq{8,4}.npy + per-step top-2
// margins, dumped by reference/dump_ref.py) must reproduce the reference token
// at every step whose margin clears the cross-context noise. Exact full-stream
// equality is NOT a sound gate here: quantized matmuls are fusion-context
// sensitive (the same mlx-lm function on the same values shifts by ~1 logit
// between lazy and materialized inputs), so knife-edge steps may flip without
// either side being wrong. Teacher-forcing keeps every step's cache content
// identical to the reference run, so each step is gated independently.
// Llama covers GQA n_repeats=4 / head_dim=64; Qwen3 covers QK-Norm /
// head_dim=128. The batched case is gated exactly (C++ vs C++): the quantized
// BatchKVCache must reproduce the single-stream quantized rows.
#include <doctest/doctest.h>

#include <algorithm>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "capi/mlxforge.h"
#include "cache/batch_kv_cache.h"
#include "cache/kv_cache.h"
#include "runtime/engine.h"
#include "support/model_fixture.h"
#include "support/reference.h"

#include "mlx/ops.h"
#include "mlx/transforms.h"

using namespace mlxforge::test;

namespace {

// Greedy next-token of the last position of each batch row: (B, L, vocab) -> (B,).
mx::array greedy_last(const mx::array& logits) {
  const int L = logits.shape()[1];
  const int vocab = logits.shape()[2];
  mx::array last = mx::reshape(
      mx::slice(logits, {0, L - 1, 0}, {logits.shape()[0], L, vocab}), {logits.shape()[0], vocab});
  return mx::astype(mx::argmax(last, /*axis=*/-1), mx::int32);  // (B,)
}

std::vector<int> to_vec(const mx::array& a) {
  mx::array c = mx::contiguous(mx::astype(a, mx::int32));
  mx::eval(c);
  const int32_t* p = c.data<int32_t>();
  return std::vector<int>(p, p + c.size());
}

// Greedy run for one prompt through a (possibly quantized) single-stream cache.
std::vector<int> solo_run(const mlxforge::DecoderModel& model, const std::vector<int>& ids,
                          int steps, mlxforge::KVQuantConfig qc) {
  mlxforge::KVCache cache(model.config().n_layers, qc);
  mx::array prompt(ids.data(), {1, static_cast<int>(ids.size())}, mx::int32);
  int next = to_vec(greedy_last(model.forward(prompt, &cache)))[0];
  std::vector<int> out = {next};
  for (int s = 1; s < steps; ++s) {
    mx::array step(&next, {1, 1}, mx::int32);
    next = to_vec(greedy_last(model.forward(step, &cache)))[0];
    out.push_back(next);
  }
  return out;
}

std::vector<float> load_floats(const std::string& path) {
  mx::array a = mx::contiguous(mx::astype(mx::load(path), mx::float32));
  mx::eval(a);
  return std::vector<float>(a.data<float>(), a.data<float>() + a.size());
}

// Steps whose reference top-2 margin is below this are knife edges that the
// fusion-context noise (~1 logit) can legitimately flip; above it a mismatch is
// a real bug.
constexpr float kMarginThreshold = 2.0f;

// Teacher-forced margin-gated gate: walk the reference stream feeding the
// REFERENCE token each step (so the cache content tracks the reference run
// regardless of our own argmax), asserting our prediction equals the reference
// wherever its margin clears the threshold.
void gated_stream_check(const mlxforge::DecoderModel& model, const std::vector<int>& prompt_ids,
                        const std::vector<int>& ref, const std::vector<float>& gaps, int bits) {
  mlxforge::KVCache cache(model.config().n_layers, mlxforge::KVQuantConfig{bits, 64});
  mx::array prompt(prompt_ids.data(), {1, static_cast<int>(prompt_ids.size())}, mx::int32);
  int pred = to_vec(greedy_last(model.forward(prompt, &cache)))[0];
  int asserted = 0;
  for (size_t i = 0; i < ref.size(); ++i) {
    if (gaps[i] >= kMarginThreshold) {
      CAPTURE(i);
      CAPTURE(gaps[i]);
      CHECK(pred == ref[i]);
      ++asserted;
    }
    if (i + 1 == ref.size()) break;
    int forced = ref[i];  // teacher-force the reference's own token
    mx::array step(&forced, {1, 1}, mx::int32);
    pred = to_vec(greedy_last(model.forward(step, &cache)))[0];
  }
  CHECK(asserted > 0);  // the fixture must gate something
}

}  // namespace

TEST_CASE("Llama: quantized-KV stream matches mlx-lm's QuantizedKVCache (margin-gated)") {
  if (!model_available()) {
    MESSAGE("MLXFORGE_MODEL_DIR not present; skipping");
    return;
  }
  mlxforge::LlamaModel& model = shared_model();
  std::vector<int> ids = load_token_ids("prompt_0_ids.npy");

  for (int bits : {8, 4}) {
    CAPTURE(bits);
    const std::string suffix = bits == 8 ? "kvq8.npy" : "kvq4.npy";
    gated_stream_check(model, ids, load_token_ids("greedy_tokens_" + suffix),
                       load_floats(ref_path("greedy_gaps_" + suffix)), bits);
  }
}

TEST_CASE("Qwen3: quantized-KV stream matches mlx-lm's QuantizedKVCache (margin-gated)") {
  if (!qwen3_model_available() || !std::ifstream(qwen3_ref_path("greedy_tokens_kvq8.npy")).good()) {
    MESSAGE("Qwen3 model/quantized fixtures not present; skipping");
    return;
  }
  mlxforge::Qwen3Model& model = shared_qwen3_model();
  std::vector<int> ids = load_qwen3_token_ids("prompt_0_ids.npy");

  for (int bits : {8, 4}) {
    CAPTURE(bits);
    const std::string suffix = bits == 8 ? "kvq8.npy" : "kvq4.npy";
    gated_stream_check(model, ids, load_qwen3_token_ids("greedy_tokens_" + suffix),
                       load_floats(qwen3_ref_path("greedy_gaps_" + suffix)), bits);
  }
}

TEST_CASE("batched quantized decode matches single-stream quantized runs") {
  if (!model_available()) {
    MESSAGE("MLXFORGE_MODEL_DIR not present; skipping");
    return;
  }
  mlxforge::LlamaModel& model = shared_model();
  const mlxforge::KVQuantConfig qc{8, 64};
  constexpr int kSteps = 8;

  // Three fixed prompts of different lengths (ragged -> left-padding).
  std::vector<std::vector<int>> prompts = {
      load_token_ids("prompt_0_ids.npy"),
      load_token_ids("prompt_1_ids.npy"),
      load_token_ids("prompt_2_ids.npy"),
  };
  const int B = static_cast<int>(prompts.size());

  std::vector<std::vector<int>> solo(B);
  for (int b = 0; b < B; ++b) solo[b] = solo_run(model, prompts[b], kSteps, qc);

  int p_max = 0;
  for (auto& p : prompts) p_max = std::max(p_max, static_cast<int>(p.size()));
  std::vector<int> left_padding(B);
  std::vector<int> padded(B * p_max, 0);  // pad id 0 (masked out)
  for (int b = 0; b < B; ++b) {
    const int pad = p_max - static_cast<int>(prompts[b].size());
    left_padding[b] = pad;
    for (size_t j = 0; j < prompts[b].size(); ++j) padded[b * p_max + pad + j] = prompts[b][j];
  }

  mlxforge::BatchKVCache cache(model.config().n_layers, left_padding, qc);
  mx::array tokens(padded.data(), {B, p_max}, mx::int32);

  mx::array next = greedy_last(model.forward(tokens, cache));
  mx::eval(next);
  std::vector<std::vector<int>> batched(B);
  for (int b = 0; b < B; ++b) batched[b].push_back(to_vec(next)[b]);

  for (int s = 1; s < kSteps; ++s) {
    mx::array step = mx::reshape(next, {B, 1});
    next = greedy_last(model.forward(step, cache));
    mx::eval(next);  // single eval per decode step, whole batch
    std::vector<int> row = to_vec(next);
    for (int b = 0; b < B; ++b) batched[b].push_back(row[b]);
  }

  for (int b = 0; b < B; ++b) {
    INFO("row " << b);
    assert_tokens_equal(batched[b], solo[b]);
  }
}

TEST_CASE("engine rejects invalid or unsupported KV quantization (no silent fallback)") {
  if (!model_available()) {
    MESSAGE("MLXFORGE_MODEL_DIR not present; skipping");
    return;
  }
  // Validation runs in the Engine constructor after the (cheap) config/tokenizer
  // head-load and before any weights load, so these are fast.
  auto cfg_with = [&](int bits, int group = 64) {
    mlxforge::EngineConfig ec;
    ec.model_spec = model_dir();
    ec.kv_bits = bits;
    ec.kv_group_size = group;
    return ec;
  };
  CHECK_THROWS_AS(mlxforge::Engine(cfg_with(5)), std::runtime_error);
  CHECK_THROWS_AS(mlxforge::Engine(cfg_with(8, 48)), std::runtime_error);

  // The hybrid (Qwen3.5) family has no quantized golden reference yet.
  if (qwen3_5_model_available()) {
    mlxforge::EngineConfig ec;
    ec.model_spec = qwen3_5_model_dir();
    ec.kv_bits = 8;
    CHECK_THROWS_AS(mlxforge::Engine(std::move(ec)), std::runtime_error);
  }

  // Same contract through the C ABI: create2 fails with an error message.
  mlxforge_engine_opts2 opts = {};
  opts.struct_size = sizeof(opts);
  opts.kv_bits = 5;
  char* err = nullptr;
  mlxforge_engine* eng = mlxforge_engine_create2(model_dir().c_str(), &opts, &err);
  CHECK(eng == nullptr);
  REQUIRE(err != nullptr);
  CHECK(std::string(err).find("kv_bits") != std::string::npos);
  mlxforge_string_free(err);
}
