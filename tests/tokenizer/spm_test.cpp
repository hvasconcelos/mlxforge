// Validates the from-scratch SentencePiece-BPE tokenizer (Gemma) against
// committed golden ids dumped from the HF tokenizer
// (reference/fixtures_gemma/tokenizer_corpus.json), over a corpus exercising the
// metaspace normalization, whitespace-run merges, byte_fallback, and special-
// token segmentation. Model-gated: self-skips when the Gemma tokenizer dir isn't
// present (only its tokenizer files are needed, no weights).
#include <doctest/doctest.h>

#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "support/reference.h"
#include "tokenizer/bpe.h"
#include "tokenizer/spm.h"
#include "tokenizer/tokenizer.h"

namespace {

std::string gemma_dir() { return MLXFORGE_MODEL_DIR_GEMMA; }
std::string tokenizer_path() { return gemma_dir() + "/tokenizer.json"; }
bool gemma_available() {
  return !gemma_dir().empty() && std::ifstream(tokenizer_path()).good();
}

std::string read_file(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  std::ostringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

}  // namespace

TEST_CASE("SpmBpeTokenizer recognizes the Gemma tokenizer and rejects byte-level BPE") {
  if (!gemma_available()) {
    MESSAGE("MLXFORGE_MODEL_DIR_GEMMA not present; skipping");
    return;
  }
  // Gemma is SentencePiece-BPE (byte_fallback) -> SPM backend, NOT the byte-level
  // BPE backend (whose decoder is ByteLevel).
  const std::string blob = read_file(tokenizer_path());
  CHECK(mlxforge::SpmBpeTokenizer::is_supported(blob));
  CHECK_FALSE(mlxforge::BpeTokenizer::is_supported(blob));
}

TEST_CASE("SpmBpeTokenizer matches the Gemma golden ids on a diverse corpus") {
  if (!gemma_available()) {
    MESSAGE("MLXFORGE_MODEL_DIR_GEMMA not present; skipping");
    return;
  }
  using namespace mlxforge::test;
  // Gemma's <bos> is id 2; the wrapper prepends it (the golden ids include it via
  // the tokenizer's post-processor). Chat format is irrelevant to raw encode.
  mlxforge::Tokenizer tok =
      mlxforge::Tokenizer::from_file(tokenizer_path(), /*bos_id=*/2, mlxforge::ChatFormat::Llama3);

  const std::string corpus_path = std::string(MLXFORGE_REF_FIXTURES_DIR_GEMMA) + "/tokenizer_corpus.json";
  nlohmann::json corpus = nlohmann::json::parse(read_file(corpus_path));
  REQUIRE(corpus.is_array());
  REQUIRE(corpus.size() > 0);

  for (const auto& entry : corpus) {
    const std::string text = entry["text"].get<std::string>();
    const std::vector<int> expected = entry["ids"].get<std::vector<int>>();
    INFO("input: " << text);
    assert_tokens_equal(tok.encode(text), expected);

    // Decode round-trips for strings free of special-token literals (decode skips
    // BOS and other special ids, and undoes metaspace + byte_fallback).
    if (text.find('<') == std::string::npos && text.find("\xE2\x96\x81") == std::string::npos) {
      CHECK(tok.decode(tok.encode(text)) == text);
    }
  }
}
