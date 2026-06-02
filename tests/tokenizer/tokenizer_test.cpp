// C++ tokenizer — encode/decode round-trip, chat template parity, and
// streaming UTF-8-safe detokenization, all vs the mlx-lm reference.
#include <doctest/doctest.h>

#include <string>
#include <vector>

#include "support/model_fixture.h"
#include "support/reference.h"
#include "tokenizer/tokenizer.h"

using namespace mlxforge::test;

namespace {
bool valid_utf8(const std::string& s) {
  size_t i = 0;
  while (i < s.size()) {
    uint8_t c = static_cast<uint8_t>(s[i]);
    size_t n;
    if (c < 0x80)
      n = 1;  // 0xxxxxxx
    else if ((c >> 5) == 0x6)
      n = 2;  // 110xxxxx
    else if ((c >> 4) == 0xE)
      n = 3;  // 1110xxxx
    else if ((c >> 3) == 0x1E)
      n = 4;  // 11110xxx
    else
      n = 0;  // invalid lead byte
    if (n == 0 || i + n > s.size()) return false;
    for (size_t k = 1; k < n; ++k)
      if ((static_cast<uint8_t>(s[i + k]) & 0xC0) != 0x80) return false;
    i += n;
  }
  return true;
}
std::string tokenizer_path() { return std::string(MLXFORGE_MODEL_DIR) + "/tokenizer.json"; }
}  // namespace

TEST_CASE("encode/decode round-trip matches the reference tokenizer") {
  if (!model_available()) {
    MESSAGE("MLXFORGE_MODEL_DIR not present; skipping");
    return;
  }
  mlxforge::Tokenizer tok = mlxforge::Tokenizer::from_file(tokenizer_path());

  // Encode matches mlx-lm's tokenization (with BOS) for the fixed prompt.
  CHECK(tok.encode("The capital of France is") == load_token_ids("prompt_0_ids.npy"));

  // Round-trip on a fixed string set, including multi-byte UTF-8.
  for (const std::string& s :
       {"Hello, world!", "The capital of France is Paris.", "caf\xC3\xA9 na\xC3\xAFve",
        "\xF0\x9F\x98\x80 emoji"}) {
    CHECK(tok.decode(tok.encode(s)) == s);
  }
}

TEST_CASE("chat template matches mlx-lm exactly") {
  if (!model_available()) {
    MESSAGE("MLXFORGE_MODEL_DIR not present; skipping");
    return;
  }
  mlxforge::Tokenizer tok = mlxforge::Tokenizer::from_file(tokenizer_path());
  std::vector<mlxforge::Tokenizer::Message> messages = {
      {"user", "What is the capital of France?"}};
  // The reference dump was taken on 01 Jun 2026 (the template injects the date).
  std::vector<int> ids = tok.apply_chat_template(messages, /*add_generation_prompt=*/true,
                                                 /*today_date=*/"01 Jun 2026");
  CHECK(ids == load_token_ids("chat_ids.npy"));
}

TEST_CASE("streaming detokenizer never emits broken UTF-8") {
  if (!model_available()) {
    MESSAGE("MLXFORGE_MODEL_DIR not present; skipping");
    return;
  }
  mlxforge::Tokenizer tok = mlxforge::Tokenizer::from_file(tokenizer_path());
  std::vector<int> stream = load_token_ids("greedy_tokens.npy");

  mlxforge::StreamingDetokenizer detok(tok);
  std::string assembled;
  for (int id : stream) {
    std::string piece = detok.add(id);
    CHECK(valid_utf8(piece));  // every emitted piece is complete UTF-8
    assembled += piece;
  }
  assembled += detok.finish();

  // Incremental assembly equals a one-shot decode of the whole stream.
  CHECK(assembled == tok.decode(stream));
  CHECK(valid_utf8(assembled));
}
