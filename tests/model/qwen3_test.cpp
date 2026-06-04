// Qwen3 dense model golden-reference checks: the QK-Norm attention front-half,
// decoder block 0, the full-forward first-token argmax, and the tokenizer corpus
// (single-digit pre-tokenization + ChatML) — all vs the fixtures dumped by
// `reference/dump_ref.py --model qwen3`. Self-skips unless both the model
// (MLXFORGE_MODEL_DIR_QWEN3) and its committed fixtures are present, since the
// Qwen3 fixtures only exist once someone with the model runs the dump.
#include <doctest/doctest.h>

#include <fstream>
#include <sstream>
#include <vector>

#include <nlohmann/json.hpp>

#include "core/config.h"
#include "mlx/ops.h"
#include "support/model_fixture.h"
#include "support/reference.h"
#include "tokenizer/tokenizer.h"

using namespace mlxforge::test;

namespace {
bool qwen3_fixtures_present() { return std::ifstream(qwen3_ref_path("manifest.json")).good(); }
bool qwen3_ready() { return qwen3_model_available() && qwen3_fixtures_present(); }

std::string read_file(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  std::ostringstream ss;
  ss << f.rdbuf();
  return ss.str();
}
}  // namespace

TEST_CASE("Qwen3: QK-Norm attention front-half matches the reference") {
  if (!qwen3_ready()) {
    MESSAGE("Qwen3 model/fixtures not present; skipping golden-reference check");
    return;
  }
  mlxforge::LlamaModel& model = shared_qwen3_model();

  // QK-Norm is the defining Qwen3 delta — the per-head q/k norm weights must load.
  CHECK(model.weights().has("model.layers.0.self_attn.q_norm.weight"));
  CHECK(model.weights().has("model.layers.0.self_attn.k_norm.weight"));

  std::vector<int> ids = load_qwen3_token_ids("prompt_0_ids.npy");
  mx::array tokens(ids.data(), {1, static_cast<int>(ids.size())}, mx::int32);

  mx::array emb = model.embed(tokens);
  assert_close(emb, load_qwen3_npy("embeddings.npy"));

  mx::array normed =
      model.rms_norm(emb, model.weights().at("model.layers.0.input_layernorm.weight"));
  assert_close(normed, load_qwen3_npy("attn_norm0.npy"));

  // Post-(q_norm + RoPE) Q, post-(k_norm + RoPE) K, and un-roped V. If QK-Norm
  // were skipped, q/k would diverge from the reference here — this is the test
  // that proves the Qwen3 attention path.
  mlxforge::LlamaModel::QKV qkv = model.attn_qkv(emb, /*layer=*/0);
  assert_close(qkv.q, load_qwen3_npy("q_rope0.npy"));
  assert_close(qkv.k, load_qwen3_npy("k_rope0.npy"));
  assert_close(qkv.v, load_qwen3_npy("v0.npy"));
}

TEST_CASE("Qwen3: decoder block 0 output matches the reference") {
  if (!qwen3_ready()) {
    MESSAGE("Qwen3 model/fixtures not present; skipping golden-reference check");
    return;
  }
  mlxforge::LlamaModel& model = shared_qwen3_model();
  std::vector<int> ids = load_qwen3_token_ids("prompt_0_ids.npy");
  mx::array tokens(ids.data(), {1, static_cast<int>(ids.size())}, mx::int32);
  mx::array emb = model.embed(tokens);

  assert_close(model.decoder_block(emb, /*layer=*/0), load_qwen3_npy("block0.npy"));
}

TEST_CASE("Qwen3: full forward logits + first-token argmax match the reference") {
  if (!qwen3_ready()) {
    MESSAGE("Qwen3 model/fixtures not present; skipping golden-reference check");
    return;
  }
  mlxforge::LlamaModel& model = shared_qwen3_model();
  std::vector<int> ids = load_qwen3_token_ids("prompt_0_ids.npy");
  const int T = static_cast<int>(ids.size());
  mx::array tokens(ids.data(), {1, T}, mx::int32);

  mx::array logits = model.forward(tokens);  // (1, T, vocab)
  const int vocab = logits.shape()[2];
  mx::array last = mx::reshape(mx::slice(logits, {0, T - 1, 0}, {1, T, vocab}), {1, vocab});
  // Looser than the Llama bound: Qwen3-0.6B is 28 layers (vs 16), so fp16
  // accumulation drifts a touch more in the raw logits. The exact argmax below is
  // the real correctness gate; the logit closeness is a sanity check.
  assert_close(last, load_qwen3_npy("logits_last.npy"), /*rtol=*/3e-2f, /*atol=*/3e-2f);

  std::vector<int> argmax = load_qwen3_token_ids("argmax.npy");
  mx::array got = mx::astype(mx::argmax(last, /*axis=*/-1), mx::int32);
  mx::eval(got);
  CHECK(got.data<int32_t>()[0] == argmax[0]);
}

TEST_CASE("Qwen3: tokenizer matches the mlx-lm golden ids (single-digit runs, ChatML)") {
  if (!qwen3_ready()) {
    MESSAGE("Qwen3 model/fixtures not present; skipping");
    return;
  }
  const std::string dir = qwen3_model_dir();
  // Qwen3 has no BOS token; encode must mirror mlx-lm (which prepends none).
  mlxforge::ModelConfig cfg = mlxforge::ModelConfig::from_file(dir + "/config.json");
  mlxforge::Tokenizer tok = mlxforge::Tokenizer::from_file(
      dir + "/tokenizer.json", cfg.bos_token_id,
      mlxforge::chat_format_from_model_type(cfg.model_type));

  nlohmann::json corpus = nlohmann::json::parse(read_file(qwen3_ref_path("tokenizer_corpus.json")));
  REQUIRE(corpus.is_array());
  REQUIRE(corpus.size() > 0);
  for (const auto& entry : corpus) {
    const std::string text = entry["text"].get<std::string>();
    const std::vector<int> expected = entry["ids"].get<std::vector<int>>();
    INFO("input: " << text);
    assert_tokens_equal(tok.encode(text), expected);
  }

  // ChatML chat prompt matches the real Qwen3 template (thinking on by default).
  std::vector<mlxforge::Tokenizer::Message> messages = {
      {"user", "What is the capital of France?"}};
  CHECK(tok.apply_chat_template(messages) == load_qwen3_token_ids("chat_ids.npy"));
  CHECK(tok.apply_chat_template(messages, /*add_generation_prompt=*/true, /*today_date=*/"",
                                /*tools=*/{}, /*enable_thinking=*/false) ==
        load_qwen3_token_ids("chat_ids_nothink.npy"));
}
