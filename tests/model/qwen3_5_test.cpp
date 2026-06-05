// Qwen3.5 hybrid-model golden-reference checks. This file covers the gated,
// partial-RoPE full-attention layer (Phase 2): the attention sublayer and the
// full decoder block at layer 3 (the first full-attention layer), vs the
// fixtures dumped by `reference/dump_ref.py --model qwen3_5`. The Gated-DeltaNet
// linear layers and the full forward pass are covered in later phases.
//
// Self-skips unless both the model (MLXFORGE_MODEL_DIR_QWEN3_5) and its committed
// fixtures are present.
#include <doctest/doctest.h>

#include <fstream>
#include <memory>
#include <sstream>
#include <vector>

#include <nlohmann/json.hpp>

#include "core/config.h"
#include "mlx/ops.h"
#include "model/model_factory.h"
#include "runtime/worker.h"
#include "scheduler/request.h"
#include "scheduler/scheduler.h"
#include "support/model_fixture.h"
#include "support/reference.h"
#include "tokenizer/tokenizer.h"

using namespace mlxforge::test;

namespace {
bool qwen3_5_fixtures_present() {
  return std::ifstream(qwen3_5_ref_path("manifest.json")).good();
}
bool qwen3_5_ready() { return qwen3_5_model_available() && qwen3_5_fixtures_present(); }

// The first full-attention layer (full_attention_interval == 4 -> index 3).
constexpr int kFullAttnLayer = 3;
}  // namespace

TEST_CASE("Qwen3.5: hybrid config and weights load") {
  if (!qwen3_5_ready()) {
    MESSAGE("Qwen3.5 model/fixtures not present; skipping golden-reference check");
    return;
  }
  mlxforge::Qwen35Model& model = shared_qwen3_5_model();
  const mlxforge::ModelConfig& cfg = model.config();

  // Hybrid layout: layer 3 is full attention, layers 0-2 are linear.
  CHECK(cfg.full_attention_interval == 4);
  CHECK(cfg.is_linear_layer(0));
  CHECK_FALSE(cfg.is_linear_layer(kFullAttnLayer));
  // Partial RoPE rotates a quarter of head_dim (256 * 0.25 = 64).
  CHECK(model.rotary_dim() == 64);

  // The full-attention layer carries gated q_proj (2x width) and QK-Norm; the
  // linear layer carries the Gated-DeltaNet projections.
  CHECK(model.weights().has("model.layers.3.self_attn.q_norm.weight"));
  CHECK(model.weights().has("model.layers.3.self_attn.k_norm.weight"));
  CHECK(model.weights().has("model.layers.0.linear_attn.conv1d.weight"));
}

TEST_CASE("Qwen3.5: gated partial-RoPE attention sublayer matches the reference") {
  if (!qwen3_5_ready()) {
    MESSAGE("Qwen3.5 model/fixtures not present; skipping golden-reference check");
    return;
  }
  mlxforge::Qwen35Model& model = shared_qwen3_5_model();

  std::vector<int> ids = load_qwen3_5_token_ids("prompt_0_ids.npy");
  mx::array tokens(ids.data(), {1, static_cast<int>(ids.size())}, mx::int32);
  mx::array emb = model.embed(tokens);
  assert_close(emb, load_qwen3_5_npy("embeddings.npy"));

  // attention() bundles the input RMSNorm, so it receives the raw residual
  // (embeddings); the reference dumps self_attn(input_layernorm(embeddings)),
  // i.e. exactly this sublayer's output before the residual add. This is the gate
  // on QK-Norm + the sigmoid output gate + partial RoPE all at once.
  mx::array attn = model.attention(emb, kFullAttnLayer);
  assert_close(attn, load_qwen3_5_npy("attn_out3.npy"));
}

TEST_CASE("Qwen3.5: full-attention decoder block matches the reference") {
  if (!qwen3_5_ready()) {
    MESSAGE("Qwen3.5 model/fixtures not present; skipping golden-reference check");
    return;
  }
  mlxforge::Qwen35Model& model = shared_qwen3_5_model();

  std::vector<int> ids = load_qwen3_5_token_ids("prompt_0_ids.npy");
  mx::array tokens(ids.data(), {1, static_cast<int>(ids.size())}, mx::int32);
  mx::array emb = model.embed(tokens);

  // decoder_block() = emb + attention() then + dense SwiGLU mlp(post-norm). The
  // base drives it via the virtual attention() override; the MLP is inherited.
  assert_close(model.decoder_block(emb, kFullAttnLayer), load_qwen3_5_npy("block3.npy"));
}

TEST_CASE("Qwen3.5: Gated-DeltaNet linear-attention sublayer matches the reference") {
  if (!qwen3_5_ready()) {
    MESSAGE("Qwen3.5 model/fixtures not present; skipping golden-reference check");
    return;
  }
  mlxforge::Qwen35Model& model = shared_qwen3_5_model();
  // Layer 0 is a linear layer; it carries the Gated-DeltaNet projections.
  REQUIRE(model.config().is_linear_layer(0));

  std::vector<int> ids = load_qwen3_5_token_ids("prompt_0_ids.npy");
  mx::array tokens(ids.data(), {1, static_cast<int>(ids.size())}, mx::int32);
  mx::array emb = model.embed(tokens);

  // The whole Gated-DeltaNet sublayer in one shot: conv -> SiLU -> normed Q/K ->
  // delta-rule recurrence -> gated RMSNorm -> out_proj. The recurrence is the
  // most numerically delicate path in the engine, so this is the gate that proves
  // it. linear_attention() bundles the input RMSNorm, matching the dumped
  // linear_attn(input_layernorm(embeddings)).
  mx::array out = model.linear_attention(emb, /*layer=*/0);
  assert_close(out, load_qwen3_5_npy("attn_out0.npy"));
}

TEST_CASE("Qwen3.5: linear-attention decoder block matches the reference") {
  if (!qwen3_5_ready()) {
    MESSAGE("Qwen3.5 model/fixtures not present; skipping golden-reference check");
    return;
  }
  mlxforge::Qwen35Model& model = shared_qwen3_5_model();

  std::vector<int> ids = load_qwen3_5_token_ids("prompt_0_ids.npy");
  mx::array tokens(ids.data(), {1, static_cast<int>(ids.size())}, mx::int32);
  mx::array emb = model.embed(tokens);

  // decoder_block() dispatches layer 0 to the Gated-DeltaNet sublayer, then adds
  // the residual and the dense SwiGLU MLP.
  assert_close(model.decoder_block(emb, /*layer=*/0), load_qwen3_5_npy("block0.npy"));
}

TEST_CASE("Qwen3.5: full forward logits + first-token argmax match the reference") {
  if (!qwen3_5_ready()) {
    MESSAGE("Qwen3.5 model/fixtures not present; skipping golden-reference check");
    return;
  }
  mlxforge::Qwen35Model& model = shared_qwen3_5_model();
  std::vector<int> ids = load_qwen3_5_token_ids("prompt_0_ids.npy");
  const int T = static_cast<int>(ids.size());
  mx::array tokens(ids.data(), {1, T}, mx::int32);

  // The whole hybrid stack composes through the inherited no-cache forward():
  // embed -> 24 decoder_block() (linear/full dispatch) -> final norm -> tied
  // (quantized) LM head. This is the end-to-end gate on the text model.
  mx::array logits = model.forward(tokens);  // (1, T, vocab)
  const int vocab = logits.shape()[2];
  mx::array last = mx::reshape(mx::slice(logits, {0, T - 1, 0}, {1, T, vocab}), {1, vocab});
  // Loose logit tolerance: 24 layers, 4-bit weights, and 18 sequential fp32
  // Gated-DeltaNet recurrences drift the raw logits more than the dense models, so
  // a few near-zero entries land just past the tight bound. The exact argmax here
  // and the exact greedy stream below are the real correctness gates.
  assert_close(last, load_qwen3_5_npy("logits_last.npy"), /*rtol=*/5e-2f, /*atol=*/5e-2f);

  std::vector<int> argmax = load_qwen3_5_token_ids("argmax.npy");
  mx::array got = mx::astype(mx::argmax(last, /*axis=*/-1), mx::int32);
  mx::eval(got);
  CHECK(got.data<int32_t>()[0] == argmax[0]);
}

TEST_CASE("Qwen3.5: single-sequence cached decode equals the no-cache forward") {
  if (!qwen3_5_ready()) {
    MESSAGE("Qwen3.5 model/fixtures not present; skipping golden-reference check");
    return;
  }
  mlxforge::Qwen35Model& model = shared_qwen3_5_model();
  std::vector<int> prompt = load_qwen3_5_token_ids("prompt_0_ids.npy");

  // Oracle: the no-cache full-recompute greedy stream (already proven to match
  // mlx-lm). The KVCache path must reproduce it, which proves the streaming state
  // carry: the conv buffer and the delta-rule recurrent state are threaded across
  // the prefill and each single-token decode step for the linear layers, and the
  // KV history for the full layers.
  constexpr int kSteps = 12;
  std::vector<int> oracle;
  {
    std::vector<int> ids = prompt;
    for (int s = 0; s < kSteps; ++s) {
      mx::array t(ids.data(), {1, static_cast<int>(ids.size())}, mx::int32);
      mx::array logits = model.forward(t);
      const int T = static_cast<int>(ids.size());
      const int vocab = logits.shape()[2];
      mx::array last = mx::reshape(mx::slice(logits, {0, T - 1, 0}, {1, T, vocab}), {1, vocab});
      mx::array nxt = mx::astype(mx::argmax(last, -1), mx::int32);
      mx::eval(nxt);
      ids.push_back(nxt.data<int32_t>()[0]);
      oracle.push_back(ids.back());
    }
  }

  // Cached path: prefill once, then feed one token per step through the KVCache.
  mlxforge::KVCache cache(model.config().n_layers);
  std::vector<int> cached;
  auto argmax_last = [&](const mx::array& logits) {
    const int L = logits.shape()[1];
    const int vocab = logits.shape()[2];
    mx::array last = mx::reshape(mx::slice(logits, {0, L - 1, 0}, {1, L, vocab}), {1, vocab});
    mx::array nxt = mx::astype(mx::argmax(last, -1), mx::int32);
    mx::eval(nxt);
    return nxt.data<int32_t>()[0];
  };
  {
    mx::array t(prompt.data(), {1, static_cast<int>(prompt.size())}, mx::int32);
    int next = argmax_last(model.forward(t, &cache));
    cached.push_back(next);
    for (int s = 1; s < kSteps; ++s) {
      mx::array step(&next, {1, 1}, mx::int32);
      next = argmax_last(model.forward(step, &cache));
      cached.push_back(next);
    }
  }
  assert_tokens_equal(cached, oracle);
}

TEST_CASE("Qwen3.5: batched ragged decode matches single-sequence runs") {
  if (!qwen3_5_ready()) {
    MESSAGE("Qwen3.5 model/fixtures not present; skipping golden-reference check");
    return;
  }
  mlxforge::Qwen35Model& model = shared_qwen3_5_model();
  const int n_layers = model.config().n_layers;
  constexpr int kSteps = 6;

  std::vector<std::vector<int>> prompts = {
      load_qwen3_5_token_ids("prompt_0_ids.npy"),
      load_qwen3_5_token_ids("prompt_1_ids.npy"),
      load_qwen3_5_token_ids("prompt_2_ids.npy"),
  };
  const int B = static_cast<int>(prompts.size());

  auto greedy_last = [](const mx::array& logits) {
    const int L = logits.shape()[1];
    const int vocab = logits.shape()[2];
    mx::array last = mx::reshape(
        mx::slice(logits, {0, L - 1, 0}, {logits.shape()[0], L, vocab}), {logits.shape()[0], vocab});
    return mx::astype(mx::argmax(last, /*axis=*/-1), mx::int32);  // (B,)
  };
  auto to_vec = [](const mx::array& a) {
    mx::array c = mx::contiguous(mx::astype(a, mx::int32));
    mx::eval(c);
    return std::vector<int>(c.data<int32_t>(), c.data<int32_t>() + c.size());
  };

  // Solo reference per prompt via the single-sequence hybrid KVCache (validated
  // against the no-cache forward above).
  std::vector<std::vector<int>> solo(B);
  for (int b = 0; b < B; ++b) {
    mlxforge::KVCache cache(n_layers);
    mx::array prompt(prompts[b].data(), {1, static_cast<int>(prompts[b].size())}, mx::int32);
    int next = to_vec(greedy_last(model.forward(prompt, &cache)))[0];
    solo[b] = {next};
    for (int s = 1; s < kSteps; ++s) {
      mx::array step(&next, {1, 1}, mx::int32);
      next = to_vec(greedy_last(model.forward(step, &cache)))[0];
      solo[b].push_back(next);
    }
  }

  // Batched ragged run: left-pad to a common width, prefill once, then decode the
  // whole batch one token per step. The hybrid BatchKVCache carries each row's KV
  // (full layers) and conv/recurrent state (linear layers); the ssm mask drops the
  // left padding. Must match the solo runs row-for-row.
  int p_max = 0;
  for (auto& p : prompts) p_max = std::max(p_max, static_cast<int>(p.size()));
  std::vector<int> left_padding(B);
  std::vector<int> padded(static_cast<size_t>(B) * p_max, 0);
  for (int b = 0; b < B; ++b) {
    const int pad = p_max - static_cast<int>(prompts[b].size());
    left_padding[b] = pad;
    for (size_t j = 0; j < prompts[b].size(); ++j) padded[b * p_max + pad + j] = prompts[b][j];
  }

  mlxforge::BatchKVCache cache(n_layers, left_padding);
  mx::array tokens(padded.data(), {B, p_max}, mx::int32);
  mx::array next = greedy_last(model.forward(tokens, cache));
  mx::eval(next);
  std::vector<std::vector<int>> batched(B);
  for (int b = 0; b < B; ++b) batched[b].push_back(to_vec(next)[b]);
  for (int s = 1; s < kSteps; ++s) {
    mx::array step = mx::reshape(next, {B, 1});
    next = greedy_last(model.forward(step, cache));
    mx::eval(next);
    std::vector<int> row = to_vec(next);
    for (int b = 0; b < B; ++b) batched[b].push_back(row[b]);
  }

  for (int b = 0; b < B; ++b) {
    INFO("row " << b);
    assert_tokens_equal(batched[b], solo[b]);
  }
}

TEST_CASE("Qwen3.5: greedy continuation reproduces the reference token stream") {
  if (!qwen3_5_ready()) {
    MESSAGE("Qwen3.5 model/fixtures not present; skipping golden-reference check");
    return;
  }
  mlxforge::Qwen35Model& model = shared_qwen3_5_model();
  std::vector<int> ids = load_qwen3_5_token_ids("prompt_0_ids.npy");
  std::vector<int> expected = load_qwen3_5_token_ids("greedy_tokens.npy");

  // Full-recompute greedy loop (no cache), mirroring the reference oracle: at each
  // step run the whole prefix and take the argmax. Exercises the entire forward
  // pass repeatedly; the token stream must match exactly.
  std::vector<int> got;
  for (size_t i = 0; i < expected.size(); ++i) {
    mx::array tokens(ids.data(), {1, static_cast<int>(ids.size())}, mx::int32);
    mx::array logits = model.forward(tokens);
    const int T = static_cast<int>(ids.size());
    const int vocab = logits.shape()[2];
    mx::array last = mx::reshape(mx::slice(logits, {0, T - 1, 0}, {1, T, vocab}), {1, vocab});
    mx::array nxt = mx::astype(mx::argmax(last, /*axis=*/-1), mx::int32);
    mx::eval(nxt);
    const int tok = nxt.data<int32_t>()[0];
    got.push_back(tok);
    ids.push_back(tok);
  }
  assert_tokens_equal(got, expected);
}

TEST_CASE("Qwen3.5: tokenizer matches the mlx-lm golden ids (ChatML)") {
  if (!qwen3_5_ready()) {
    MESSAGE("Qwen3.5 model/fixtures not present; skipping");
    return;
  }
  const std::string dir = qwen3_5_model_dir();
  mlxforge::ModelConfig cfg = mlxforge::ModelConfig::from_file(dir + "/config.json");
  // model_type "qwen3_5" must select the Qwen3.5 ChatML chat format.
  CHECK(mlxforge::chat_format_from_model_type(cfg.model_type) == mlxforge::ChatFormat::Qwen35);

  // Qwen has no BOS token; the byte-level BPE tokenizer.json loads like Qwen3's.
  mlxforge::Tokenizer tok = mlxforge::Tokenizer::from_file(
      dir + "/tokenizer.json", cfg.bos_token_id,
      mlxforge::chat_format_from_model_type(cfg.model_type));

  std::ifstream f(qwen3_5_ref_path("tokenizer_corpus.json"), std::ios::binary);
  std::ostringstream ss;
  ss << f.rdbuf();
  nlohmann::json corpus = nlohmann::json::parse(ss.str());
  REQUIRE(corpus.is_array());
  for (const auto& entry : corpus) {
    const std::string text = entry["text"].get<std::string>();
    const std::vector<int> expected = entry["ids"].get<std::vector<int>>();
    INFO("input: " << text);
    assert_tokens_equal(tok.encode(text), expected);
  }

  // ChatML chat prompt matches the real Qwen3.5 template (thinking on by default,
  // and the thinking-disabled variant).
  std::vector<mlxforge::Tokenizer::Message> messages = {
      {"user", "What is the capital of France?"}};
  CHECK(tok.apply_chat_template(messages) == load_qwen3_5_token_ids("chat_ids.npy"));
  CHECK(tok.apply_chat_template(messages, /*add_generation_prompt=*/true, /*today_date=*/"",
                                /*tools=*/{}, /*enable_thinking=*/false) ==
        load_qwen3_5_token_ids("chat_ids_nothink.npy"));
}

TEST_CASE("Qwen3.5: continuous-batching worker generates the correct tokens") {
  if (!qwen3_5_ready()) {
    MESSAGE("Qwen3.5 model/fixtures not present; skipping golden-reference check");
    return;
  }
  constexpr int kSteps = 6;
  std::vector<std::vector<int>> prompts = {
      load_qwen3_5_token_ids("prompt_0_ids.npy"),
      load_qwen3_5_token_ids("prompt_1_ids.npy"),
      load_qwen3_5_token_ids("prompt_2_ids.npy"),
  };
  const int B = static_cast<int>(prompts.size());

  // Solo greedy reference per prompt (single-sequence hybrid KVCache), computed on
  // this thread before the worker starts (no concurrent MLX use).
  std::vector<std::vector<int>> solo(B);
  {
    mlxforge::Qwen35Model& model = shared_qwen3_5_model();
    for (int b = 0; b < B; ++b) {
      mlxforge::KVCache cache(model.config().n_layers);
      mx::array prompt(prompts[b].data(), {1, static_cast<int>(prompts[b].size())}, mx::int32);
      auto step_argmax = [&](const mx::array& logits) {
        const int L = logits.shape()[1], V = logits.shape()[2];
        mx::array last = mx::reshape(mx::slice(logits, {0, L - 1, 0}, {1, L, V}), {1, V});
        mx::array a = mx::astype(mx::argmax(last, -1), mx::int32);
        mx::eval(a);
        return a.data<int32_t>()[0];
      };
      int next = step_argmax(model.forward(prompt, &cache));
      solo[b] = {next};
      for (int s = 1; s < kSteps; ++s) {
        mx::array t(&next, {1, 1}, mx::int32);
        next = step_argmax(model.forward(t, &cache));
        solo[b].push_back(next);
      }
    }
  }

  // Drive the same prompts through the continuous-batching worker, which builds
  // the model via create_model() (-> Qwen35Model) and runs the hybrid BatchKVCache
  // through admission (merge), bucketed decode (pad_dummies) and eviction (filter).
  const std::string dir = qwen3_5_model_dir();
  mlxforge::Scheduler sched;
  mlxforge::Worker worker(
      [dir] {
        mlxforge::ModelConfig c = mlxforge::ModelConfig::from_file(dir + "/config.json");
        auto w = mlxforge::load_weights(dir, c);
        return mlxforge::create_model(std::move(c), std::move(w));
      },
      &sched);
  worker.start();

  std::vector<std::shared_ptr<mlxforge::Request>> reqs(B);
  for (int b = 0; b < B; ++b) {
    reqs[b] = std::make_shared<mlxforge::Request>();
    reqs[b]->prompt_ids = prompts[b];
    reqs[b]->params.temperature = 0.0f;  // greedy
    reqs[b]->max_tokens = kSteps;
    reqs[b]->eos_ids = {};  // run the full kSteps regardless of content
    sched.submit(reqs[b]);
  }

  std::vector<std::vector<int>> got(B);
  for (int b = 0; b < B; ++b) {
    int tok = 0;
    while (reqs[b]->tokens.pop(tok)) got[b].push_back(tok);
  }
  worker.stop();

  for (int b = 0; b < B; ++b) {
    INFO("request " << b);
    assert_tokens_equal(got[b], solo[b]);
  }
}
