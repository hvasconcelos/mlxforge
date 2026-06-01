// MLXFORGE-004: weight key sanitize + shard-index parsing (pure logic, no GPU/weights).
#include <doctest/doctest.h>

#include <fstream>
#include <string>

#include "core/weights.h"

using namespace mlxforge;

namespace {
nlohmann::json load_fixture(const char* name) {
  std::ifstream f(std::string(MLXFORGE_TEST_FIXTURES_DIR) + "/" + name);
  REQUIRE(f.good());
  nlohmann::json j;
  f >> j;
  return j;
}
}  // namespace

TEST_CASE("sanitize_key canonicalizes a table of raw keys") {
  // Already-canonical keys pass through unchanged.
  CHECK(sanitize_key("model.layers.0.self_attn.q_proj.weight").value() ==
        "model.layers.0.self_attn.q_proj.weight");
  CHECK(sanitize_key("model.norm.weight").value() == "model.norm.weight");

  // A wrapped language-tower prefix is stripped.
  CHECK(sanitize_key("language_model.model.layers.3.mlp.up_proj.weight").value() ==
        "model.layers.3.mlp.up_proj.weight");
  CHECK(sanitize_key("language_model.lm_head.weight").value() == "lm_head.weight");

  // Rotary inv_freq buffers are dropped (not weights).
  CHECK_FALSE(sanitize_key("model.layers.0.self_attn.rotary_emb.inv_freq").has_value());
  CHECK_FALSE(sanitize_key("model.rotary_emb.inv_freq").has_value());
}

TEST_CASE("parse_shard_index resolves the file each tensor key lives in") {
  auto index = load_fixture("shard_index.json");
  auto weight_map = parse_shard_index(index);

  CHECK(weight_map.size() == 5);
  CHECK(weight_map.at("model.embed_tokens.weight") == "model-00001-of-00002.safetensors");
  CHECK(weight_map.at("model.layers.0.self_attn.k_proj.weight") ==
        "model-00001-of-00002.safetensors");
  CHECK(weight_map.at("model.norm.weight") == "model-00002-of-00002.safetensors");

  // Unique shard files, sorted.
  auto files = shard_files(weight_map);
  CHECK(files == std::vector<std::string>{"model-00001-of-00002.safetensors",
                                          "model-00002-of-00002.safetensors"});
}

TEST_CASE("parse_shard_index rejects an index without a weight_map") {
  auto j = nlohmann::json::parse(R"({"metadata": {"total_size": 1}})");
  CHECK_THROWS_AS(parse_shard_index(j), std::runtime_error);
}
