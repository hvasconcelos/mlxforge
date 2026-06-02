// ModelConfig parsing (pure logic, no GPU / no weights).
#include <doctest/doctest.h>

#include <string>

#include "core/config.h"

using mlxforge::ModelConfig;

namespace {
std::string fixture(const char* name) {
  return std::string(MLXFORGE_TEST_FIXTURES_DIR) + "/" + name;
}
}  // namespace

TEST_CASE("ModelConfig loads the Llama-3.2-1B config with correct values") {
  ModelConfig c = ModelConfig::from_file(fixture("config_llama32_1b.json"));
  CHECK(c.n_layers == 16);
  CHECK(c.hidden == 2048);
  CHECK(c.n_heads == 32);
  CHECK(c.n_kv_heads == 8);
  CHECK(c.head_dim == 64);
  CHECK(c.vocab == 128256);
  CHECK(c.intermediate_size == 8192);
  CHECK(c.max_position_embeddings == 131072);
  CHECK(c.tie_word_embeddings == true);
  CHECK(c.bos_token_id == 128000);
  CHECK(c.eos_token_ids == std::vector<int>{128001, 128008, 128009});

  // rope_theta and rms_eps come from config, not hard-coded.
  CHECK(c.rope_theta == doctest::Approx(500000.0f));
  CHECK(c.rms_eps == doctest::Approx(1e-5f));

  REQUIRE(c.rope_scaling.has_value());
  CHECK(c.rope_scaling->rope_type == "llama3");
  CHECK(c.rope_scaling->factor == doctest::Approx(32.0f));
  CHECK(c.rope_scaling->original_max_position_embeddings == 8192);
}

TEST_CASE("ModelConfig errors clearly on a missing required field") {
  // Valid except num_key_value_heads is absent.
  auto j = nlohmann::json::parse(R"({
    "num_hidden_layers": 16, "hidden_size": 2048, "num_attention_heads": 32,
    "vocab_size": 128256, "intermediate_size": 8192,
    "rope_theta": 500000.0, "rms_norm_eps": 1e-5
  })");
  CHECK_THROWS_WITH_AS(ModelConfig::from_json(j),
                       "config.json: missing required field 'num_key_value_heads'",
                       std::runtime_error);
}

TEST_CASE("ModelConfig ignores unknown/extra keys") {
  auto j = nlohmann::json::parse(R"({
    "num_hidden_layers": 2, "hidden_size": 8, "num_attention_heads": 4,
    "num_key_value_heads": 2, "vocab_size": 100, "intermediate_size": 16,
    "rope_theta": 10000.0, "rms_norm_eps": 1e-6,
    "some_future_field": 123, "architectures": ["X"], "torch_dtype": "bfloat16"
  })");
  ModelConfig c = ModelConfig::from_json(j);  // must not throw
  CHECK(c.n_layers == 2);
  CHECK(c.head_dim == 2);  // derived: hidden(8) / n_heads(4)
  CHECK_FALSE(c.rope_scaling.has_value());
  CHECK(c.eos_token_ids.empty());
}
