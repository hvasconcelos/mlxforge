// Validates the from-scratch byte-level BPE tokenizer against committed golden
// ids dumped from mlx-lm (reference/fixtures/tokenizer_corpus.json), over a
// corpus that exercises the pre-tokenizer's edge cases (whitespace runs,
// newlines, contractions, digits, CJK, accented Latin, emoji/ZWJ, code, inline
// special tokens). Model-gated: self-skips when the model isn't present.
#include <doctest/doctest.h>

#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "core/gguf.h"
#include "support/model_fixture.h"
#include "support/reference.h"
#include "tokenizer/bpe.h"
#include "tokenizer/tokenizer.h"

using namespace mlxforge::test;

namespace {

std::string tokenizer_path() { return std::string(MLXFORGE_MODEL_DIR) + "/tokenizer.json"; }

std::string read_file(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  std::ostringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

std::string gguf_path() { return MLXFORGE_GGUF_MODEL; }
bool gguf_available() { return !gguf_path().empty() && std::ifstream(gguf_path()).good(); }

}  // namespace

TEST_CASE("BpeTokenizer matches the mlx-lm golden ids on a diverse corpus") {
  if (!model_available()) {
    MESSAGE("MLXFORGE_MODEL_DIR not present; skipping");
    return;
  }
  // Llama-3.2 is byte-level BPE, so the wrapper uses our own implementation.
  CHECK(mlxforge::BpeTokenizer::is_supported(read_file(tokenizer_path())));

  mlxforge::Tokenizer tok = mlxforge::Tokenizer::from_file(tokenizer_path());

  const std::string corpus_path = std::string(MLXFORGE_REF_FIXTURES_DIR) + "/tokenizer_corpus.json";
  nlohmann::json corpus = nlohmann::json::parse(read_file(corpus_path));
  REQUIRE(corpus.is_array());
  REQUIRE(corpus.size() > 0);

  for (const auto& entry : corpus) {
    const std::string text = entry["text"].get<std::string>();
    const std::vector<int> expected = entry["ids"].get<std::vector<int>>();
    INFO("input: " << text);
    // tok.encode prepends BOS, matching mlx-lm's tok.encode (how the fixture was dumped).
    assert_tokens_equal(tok.encode(text), expected);

    // Decode round-trips for the special-token-free strings (decode skips BOS
    // and other special ids).
    if (text.find("<|") == std::string::npos) {
      CHECK(tok.decode(tok.encode(text)) == text);
    }
  }
}

TEST_CASE("GGUF tokenizer rebuilt from metadata matches the golden ids") {
  if (!gguf_available()) {
    MESSAGE("MLXFORGE_GGUF_MODEL not present; skipping");
    return;
  }
  // Rebuild the tokenizer purely from GGUF metadata (no tokenizer.json) and
  // require byte-identical ids to the same golden corpus — the GGUF Llama-3.2
  // tokenizer must reproduce the safetensors tokenizer exactly.
  mlxforge::GgufModel g = mlxforge::load_gguf_model(gguf_path());
  mlxforge::Tokenizer tok = mlxforge::Tokenizer::from_gguf(g.tokens, g.merges, g.token_types,
                                                           g.pre, g.bos_id);

  const std::string corpus_path = std::string(MLXFORGE_REF_FIXTURES_DIR) + "/tokenizer_corpus.json";
  nlohmann::json corpus = nlohmann::json::parse(read_file(corpus_path));
  REQUIRE(corpus.is_array());

  for (const auto& entry : corpus) {
    const std::string text = entry["text"].get<std::string>();
    const std::vector<int> expected = entry["ids"].get<std::vector<int>>();
    INFO("input: " << text);
    assert_tokens_equal(tok.encode(text), expected);
    if (text.find("<|") == std::string::npos) {
      CHECK(tok.decode(tok.encode(text)) == text);
    }
  }
}
