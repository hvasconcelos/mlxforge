// weight key sanitize + shard-index parsing (pure logic, no GPU/weights).
#include <doctest/doctest.h>

#include <fstream>
#include <string>

#include "core/weights.h"
#include "mlx/array.h"

using namespace mlxforge;
namespace mx = mlx::core;

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

TEST_CASE("Weights::is_quantized detects quantization per-weight") {
  Weights w;
  const mx::array dummy = mx::array(0.0f);  // shape doesn't matter for detection

  // A quantized weight carries a "<base>.scales" sibling; its params come from
  // the `quant` map.
  w.tensors.emplace("model.layers.0.mlp.down_proj.weight", dummy);
  w.tensors.emplace("model.layers.0.mlp.down_proj.scales", dummy);
  w.tensors.emplace("model.layers.0.mlp.down_proj.biases", dummy);
  w.quant.emplace("model.layers.0.mlp.down_proj", QuantParams{32, 8});

  // A dense weight in the same model has no ".scales".
  w.tensors.emplace("model.layers.0.input_layernorm.weight", dummy);

  // A quantized weight absent from `quant` falls back to QuantParams defaults.
  w.tensors.emplace("model.embed_tokens.weight", dummy);
  w.tensors.emplace("model.embed_tokens.scales", dummy);

  QuantParams qp;
  CHECK(w.is_quantized("model.layers.0.mlp.down_proj.weight", qp));
  CHECK(qp.group_size == 32);
  CHECK(qp.bits == 8);

  CHECK_FALSE(w.is_quantized("model.layers.0.input_layernorm.weight", qp));

  QuantParams def;
  CHECK(w.is_quantized("model.embed_tokens.weight", def));
  CHECK(def.group_size == 64);  // QuantParams default
  CHECK(def.bits == 4);

  // A key that doesn't end in ".weight" is never quantized.
  CHECK_FALSE(w.is_quantized("model.layers.0.mlp.down_proj.scales", qp));
}
