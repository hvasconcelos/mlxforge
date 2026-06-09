// Golden gate for the Qwen3-Embedding path. A bare engine.embed() on a
// Qwen3-Embedding checkpoint must reproduce the model's pooled sentence
// embedding (last-token over the appended EOS, L2-normalized) AND its exact
// tokenization (no BOS, trailing <|endoftext|>=151643). Self-skips when the
// model isn't cached. The texts here MUST match reference/dump_ref.py's
// qwen3_embedding spec, or the token gate fails by design.
#include <doctest/doctest.h>

#include <chrono>
#include <string>
#include <thread>
#include <vector>

#include "mlx/mlx.h"
#include "runtime/engine.h"
#include "support/reference.h"

using namespace mlxforge::test;
namespace mx = mlx::core;

namespace {

const std::string kEmbedModelDir = MLXFORGE_MODEL_DIR_QWEN3_EMBEDDING;

// Must match reference/dump_ref.py MODELS["qwen3_embedding"] exactly.
const std::string kTask =
    "Given a web search query, retrieve relevant passages that answer the query";
const std::string kQuery = "What is the capital of China?";
const std::string kDoc = "The capital of China is Beijing.";

double dot(const std::vector<float>& a, const std::vector<float>& b) {
  double s = 0;
  for (size_t i = 0; i < a.size(); ++i) s += static_cast<double>(a[i]) * b[i];
  return s;
}

std::vector<float> to_vec(const mx::array& a) {
  mx::array c = mx::contiguous(mx::astype(a, mx::float32));
  mx::eval(c);
  return std::vector<float>(c.data<float>(), c.data<float>() + c.size());
}

}  // namespace

TEST_CASE("Qwen3-Embedding pooled vectors and tokenization match the golden reference") {
  if (kEmbedModelDir.empty()) {
    MESSAGE("MLXFORGE_MODEL_DIR_QWEN3_EMBEDDING not present; skipping");
    return;
  }
  mlxforge::EngineConfig cfg;
  cfg.model_spec = kEmbedModelDir;
  mlxforge::Engine engine(cfg);
  while (!engine.ready()) std::this_thread::sleep_for(std::chrono::milliseconds(10));

  // The checkpoint ships a sentence-transformers last-token pooling sidecar, so
  // the DEFAULT embed() path must self-select last-token pooling + trailing EOS.
  mlxforge::EmbedOptions query_opts;
  query_opts.instruction = kTask;
  const std::vector<float> q = engine.embed(kQuery, query_opts);  // detected last+eos
  const std::vector<float> d = engine.embed(kDoc);                // detected last+eos

  REQUIRE(q.size() > 0);
  REQUIRE(q.size() == d.size());

  // 1) Pooled vector gate: elementwise close AND cosine ~1 (both are unit-norm).
  const mx::array q_ref = load_qwen3_embedding_npy("embed_query.npy");
  const mx::array d_ref = load_qwen3_embedding_npy("embed_doc.npy");
  const mx::array q_act(q.data(), {static_cast<int>(q.size())}, mx::float32);
  const mx::array d_act(d.data(), {static_cast<int>(d.size())}, mx::float32);
  assert_close(q_act, q_ref);
  assert_close(d_act, d_ref);
  CHECK(dot(q, to_vec(q_ref)) == doctest::Approx(1.0).epsilon(0.02));
  CHECK(dot(d, to_vec(d_ref)) == doctest::Approx(1.0).epsilon(0.02));

  // 2) Unit normalization.
  CHECK(dot(q, q) == doctest::Approx(1.0).epsilon(0.02));
  CHECK(dot(d, d) == doctest::Approx(1.0).epsilon(0.02));

  // 3) Tokenization gate: the engine's ids (no BOS + appended EOS) must match the
  // reference HF tokenization for both the instruction-wrapped query and the doc.
  const std::vector<int> q_ids_ref = load_qwen3_embedding_token_ids("embed_query_ids.npy");
  const std::vector<int> d_ids_ref = load_qwen3_embedding_token_ids("embed_doc_ids.npy");
  const int eos = engine.config().eos_token_ids.front();
  std::vector<int> q_ids = engine.tokenizer().encode("Instruct: " + kTask + "\nQuery:" + kQuery);
  q_ids.push_back(eos);
  std::vector<int> d_ids = engine.tokenizer().encode(kDoc);
  d_ids.push_back(eos);
  assert_tokens_equal(q_ids, q_ids_ref);
  assert_tokens_equal(d_ids, d_ids_ref);
}

TEST_CASE("Qwen3-Embedding ranks a relevant document above an unrelated one") {
  if (kEmbedModelDir.empty()) {
    MESSAGE("MLXFORGE_MODEL_DIR_QWEN3_EMBEDDING not present; skipping");
    return;
  }
  mlxforge::EngineConfig cfg;
  cfg.model_spec = kEmbedModelDir;
  mlxforge::Engine engine(cfg);
  while (!engine.ready()) std::this_thread::sleep_for(std::chrono::milliseconds(10));

  mlxforge::EmbedOptions query_opts;
  query_opts.instruction = kTask;
  const std::vector<float> q = engine.embed(kQuery, query_opts);
  const std::vector<float> relevant = engine.embed("Beijing is the capital city of China.");
  const std::vector<float> unrelated = engine.embed("The stock market fell sharply today.");

  CHECK(dot(q, relevant) > dot(q, unrelated));
}
