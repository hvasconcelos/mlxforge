// Differential test for the from-scratch BpeTokenizer: it must produce
// byte-identical token ids to the HF tokenizer (tokenizers-cpp), the oracle,
// over a hand-picked corpus AND a deterministic Unicode fuzz. Also checks
// decode round-trips. Model-gated: self-skips when the model isn't present.
#include <doctest/doctest.h>

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include <tokenizers_cpp.h>

#include "support/model_fixture.h"
#include "support/reference.h"
#include "tokenizer/bpe.h"

using namespace mlxforge::test;

namespace {

std::string tokenizer_path() { return std::string(MLXFORGE_MODEL_DIR) + "/tokenizer.json"; }

std::string read_file(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  std::ostringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

std::vector<int> oracle_encode(tokenizers::Tokenizer* oracle, const std::string& s) {
  std::vector<int32_t> ids = oracle->Encode(s);
  return std::vector<int>(ids.begin(), ids.end());
}

void append_utf8(uint32_t cp, std::string& out) {
  if (cp < 0x80) {
    out.push_back(static_cast<char>(cp));
  } else if (cp < 0x800) {
    out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
    out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
  } else if (cp < 0x10000) {
    out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
    out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
    out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
  } else {
    out.push_back(static_cast<char>(0xF0 | (cp >> 18)));
    out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
    out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
    out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
  }
}

}  // namespace

TEST_CASE("BpeTokenizer matches the HF tokenizer on a diverse corpus") {
  if (!model_available()) {
    MESSAGE("MLXFORGE_MODEL_DIR not present; skipping");
    return;
  }
  const std::string blob = read_file(tokenizer_path());
  mlxforge::BpeTokenizer bpe = mlxforge::BpeTokenizer::from_blob(blob);
  std::unique_ptr<tokenizers::Tokenizer> oracle = tokenizers::Tokenizer::FromBlobJSON(blob);

  const std::vector<std::string> corpus = {
      "",
      "The capital of France is Paris.",
      "Hello, world!",
      "don't I'll we've they're it's can't",
      "DON'T SHOUT",
      "spaces    here     and       more",
      "  leading and trailing  ",
      "tabs\tand\tmore\ttabs",
      "newlines\n\nand\r\nwindows\r\nendings",
      "mixed \n  \t whitespace \n\n",
      "1 12 123 1234 100000 3.14159",
      "snake_case camelCase kebab-case",
      "for (int i = 0; i < n; ++i) { sum += a[i]; }",
      "café naïve résumé Zürich",
      "Ünïcödé ßharp",
      "你好世界，今天天气很好。",
      "こんにちは世界",
      "Привет мир",
      "emoji 😀 and 👨‍👩‍👧‍👦 family",
      "math ∑∫√≠≤ symbols",
      "<|begin_of_text|>hi<|eot_id|>",
      "<|start_header_id|>user<|end_header_id|>\n\nWhat?<|eot_id|>",
      "a<|eot_id|><|eot_id|>b",
      "URL: https://example.com/path?q=1&x=2#frag",
      "@user #hashtag $100 50% (parens) [brackets] {braces}",
  };

  for (const std::string& s : corpus) {
    INFO("input: " << s);
    assert_tokens_equal(bpe.encode(s), oracle_encode(oracle.get(), s));
  }

  // Decode round-trips for the special-token-free strings.
  for (const std::string& s : corpus) {
    if (s.find("<|") != std::string::npos) continue;
    INFO("roundtrip: " << s);
    CHECK(bpe.decode(bpe.encode(s)) == s);
  }
}

TEST_CASE("BpeTokenizer matches the HF tokenizer under Unicode fuzzing") {
  if (!model_available()) {
    MESSAGE("MLXFORGE_MODEL_DIR not present; skipping");
    return;
  }
  const std::string blob = read_file(tokenizer_path());
  mlxforge::BpeTokenizer bpe = mlxforge::BpeTokenizer::from_blob(blob);
  std::unique_ptr<tokenizers::Tokenizer> oracle = tokenizers::Tokenizer::FromBlobJSON(blob);

  // Codepoint pools biased toward the regex's boundaries (whitespace, newlines,
  // quotes, digits, punctuation) plus letters and some astral-plane characters.
  std::vector<uint32_t> pool;
  auto add_range = [&](uint32_t lo, uint32_t hi) {
    for (uint32_t c = lo; c <= hi; ++c) pool.push_back(c);
  };
  add_range('a', 'z');
  add_range('A', 'Z');
  add_range('0', '9');
  for (uint32_t c : {0x20u, 0x20u, 0x20u, 0x09u, 0x0Au, 0x0Du, 0x27u, 0x2Eu, 0x2Cu, 0x21u, 0x3Fu,
                     0x2Du, 0x5Fu, 0x2Fu, 0x40u, 0x23u, 0x24u, 0x28u, 0x29u})
    pool.push_back(c);
  add_range(0x00C0, 0x00FF);  // Latin-1 accented
  add_range(0x0400, 0x041F);  // Cyrillic
  add_range(0x4E00, 0x4E20);  // CJK
  for (uint32_t c : {0x1F600u, 0x1F601u, 0x1F44Du, 0x200Du, 0x1F468u})  // emoji + ZWJ
    pool.push_back(c);

  std::mt19937 rng(0xC0FFEE);  // fixed seed -> deterministic
  std::uniform_int_distribution<size_t> len_dist(0, 24);
  std::uniform_int_distribution<size_t> idx_dist(0, pool.size() - 1);

  const int kIters = 4000;
  int mismatches = 0;
  for (int iter = 0; iter < kIters && mismatches == 0; ++iter) {
    std::string s;
    const size_t len = len_dist(rng);
    for (size_t k = 0; k < len; ++k) append_utf8(pool[idx_dist(rng)], s);

    std::vector<int> got = bpe.encode(s);
    std::vector<int> want = oracle_encode(oracle.get(), s);
    if (got != want) {
      ++mismatches;
      INFO("fuzz iter " << iter << " mismatch on input bytes:");
      std::string hex;
      for (unsigned char c : s) {
        char b[4];
        std::snprintf(b, sizeof(b), "%02X ", c);
        hex += b;
      }
      INFO("hex: " << hex);
      assert_tokens_equal(got, want);
    }
  }
  CHECK(mismatches == 0);
}
