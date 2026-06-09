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

TEST_CASE("ModelConfig parses the Qwen3.5 hybrid text_config") {
  // A representative subset of mlx-community/Qwen3.5-0.8B-4bit's config.json: the
  // text hyperparameters nest under "text_config", rope_theta lives in a
  // "rope_parameters" sub-object, the quantization block stays at the top level,
  // and the hybrid linear-attention fields select Gated-DeltaNet vs full layers.
  auto j = nlohmann::json::parse(R"({
    "architectures": ["Qwen3_5ForConditionalGeneration"],
    "model_type": "qwen3_5",
    "quantization": {"group_size": 64, "bits": 4},
    "tie_word_embeddings": true,
    "vision_config": {"depth": 12, "hidden_size": 768},
    "text_config": {
      "model_type": "qwen3_5_text",
      "hidden_size": 1024,
      "intermediate_size": 3584,
      "num_hidden_layers": 24,
      "num_attention_heads": 8,
      "num_key_value_heads": 2,
      "head_dim": 256,
      "rms_norm_eps": 1e-06,
      "vocab_size": 248320,
      "max_position_embeddings": 262144,
      "tie_word_embeddings": true,
      "eos_token_id": 248044,
      "attn_output_gate": true,
      "full_attention_interval": 4,
      "linear_num_key_heads": 16,
      "linear_num_value_heads": 16,
      "linear_key_head_dim": 128,
      "linear_value_head_dim": 128,
      "linear_conv_kernel_dim": 4,
      "mlp_only_layers": [],
      "rope_parameters": {
        "mrope_interleaved": true,
        "mrope_section": [11, 11, 10],
        "rope_type": "default",
        "rope_theta": 10000000,
        "partial_rotary_factor": 0.25
      }
    }
  })");
  ModelConfig c = ModelConfig::from_json(j);

  // model_type is the top-level architecture id (selects the chat format), not
  // the nested "*_text" variant.
  CHECK(c.model_type == "qwen3_5");

  // Core hyperparameters read from text_config.
  CHECK(c.n_layers == 24);
  CHECK(c.hidden == 1024);
  CHECK(c.n_heads == 8);
  CHECK(c.n_kv_heads == 2);
  CHECK(c.head_dim == 256);  // explicit, not hidden/n_heads (128)
  CHECK(c.vocab == 248320);
  CHECK(c.intermediate_size == 3584);
  CHECK(c.rms_eps == doctest::Approx(1e-6f));
  CHECK(c.tie_word_embeddings == true);
  CHECK(c.eos_token_ids == std::vector<int>{248044});

  // rope_theta and partial_rotary_factor come from the rope_parameters sub-object.
  CHECK(c.rope_theta == doctest::Approx(10000000.0f));
  CHECK(c.partial_rotary_factor == doctest::Approx(0.25f));

  // Hybrid linear-attention fields.
  CHECK(c.attn_output_gate == true);
  CHECK(c.full_attention_interval == 4);
  CHECK(c.linear_num_key_heads == 16);
  CHECK(c.linear_num_value_heads == 16);
  CHECK(c.linear_key_head_dim == 128);
  CHECK(c.linear_value_head_dim == 128);
  CHECK(c.linear_conv_kernel_dim == 4);

  // Quantization (top-level block) still parses.
  CHECK(c.quantized);
  CHECK(c.quant_group_size == 64);
  CHECK(c.quant_bits == 4);

  // Interleaved M-RoPE parsed from the rope_parameters sub-object.
  CHECK(c.mrope_interleaved == true);
  CHECK(c.mrope_section == std::vector<int>{11, 11, 10});
  // The (minimal) vision_config is captured even when most fields are absent.
  REQUIRE(c.has_vision_tower());
  CHECK(c.vision->depth == 12);
  CHECK(c.vision->hidden == 768);

  // Layer dispatch: every 4th layer (indices 3, 7, ...) is full attention; the
  // rest are Gated-DeltaNet linear-attention. Mirrors the mlx_lm reference.
  CHECK(c.is_linear_layer(0));
  CHECK(c.is_linear_layer(1));
  CHECK(c.is_linear_layer(2));
  CHECK_FALSE(c.is_linear_layer(3));
  CHECK(c.is_linear_layer(4));
  CHECK_FALSE(c.is_linear_layer(7));
  CHECK_FALSE(c.is_linear_layer(23));
  // Not a MoE model: no sparse layers.
  CHECK_FALSE(c.is_moe_layer(0));
}

TEST_CASE("ModelConfig parses the Qwen3-VL vision_config + interleaved M-RoPE") {
  // A representative subset of mlx-community/Qwen3-VL-4B-Instruct's config.json:
  // the text decoder nests under "text_config" (a plain Qwen3 backbone with
  // QK-norm), the ViT under "vision_config", interleaved M-RoPE lives in the text
  // rope_scaling, and the vision special-token ids sit at the top level.
  auto j = nlohmann::json::parse(R"({
    "architectures": ["Qwen3VLForConditionalGeneration"],
    "model_type": "qwen3_vl",
    "image_token_id": 151655,
    "video_token_id": 151656,
    "vision_start_token_id": 151652,
    "vision_end_token_id": 151653,
    "tie_word_embeddings": true,
    "text_config": {
      "model_type": "qwen3_vl_text",
      "vocab_size": 151936,
      "hidden_size": 2560,
      "intermediate_size": 9728,
      "num_hidden_layers": 36,
      "num_attention_heads": 32,
      "num_key_value_heads": 8,
      "head_dim": 128,
      "rms_norm_eps": 1e-06,
      "max_position_embeddings": 262144,
      "rope_theta": 5000000,
      "eos_token_id": 151645,
      "bos_token_id": 151643,
      "rope_scaling": {
        "mrope_interleaved": true,
        "mrope_section": [24, 20, 20],
        "rope_type": "default"
      }
    },
    "vision_config": {
      "depth": 24,
      "hidden_size": 1024,
      "intermediate_size": 4096,
      "num_heads": 16,
      "in_channels": 3,
      "patch_size": 16,
      "spatial_merge_size": 2,
      "temporal_patch_size": 2,
      "out_hidden_size": 2560,
      "num_position_embeddings": 2304,
      "deepstack_visual_indexes": [5, 11, 17]
    }
  })");
  ModelConfig c = ModelConfig::from_json(j);

  // Text backbone reads from text_config (a Qwen3 dense decoder, GQA 32:8).
  CHECK(c.model_type == "qwen3_vl");
  CHECK(c.n_layers == 36);
  CHECK(c.hidden == 2560);
  CHECK(c.n_heads == 32);
  CHECK(c.n_kv_heads == 8);
  CHECK(c.head_dim == 128);
  CHECK(c.rope_theta == doctest::Approx(5000000.0f));

  // Vision special tokens (top level).
  CHECK(c.image_token_id == 151655);
  CHECK(c.video_token_id == 151656);
  CHECK(c.vision_start_token_id == 151652);
  CHECK(c.vision_end_token_id == 151653);

  // Interleaved M-RoPE from the text rope_scaling; entries sum to head_dim/2 = 64.
  CHECK(c.mrope_interleaved == true);
  CHECK(c.mrope_section == std::vector<int>{24, 20, 20});

  // Vision tower.
  REQUIRE(c.has_vision_tower());
  const auto& v = *c.vision;
  CHECK(v.depth == 24);
  CHECK(v.hidden == 1024);
  CHECK(v.intermediate_size == 4096);
  CHECK(v.num_heads == 16);
  CHECK(v.head_dim() == 64);
  CHECK(v.in_channels == 3);
  CHECK(v.patch_size == 16);
  CHECK(v.spatial_merge_size == 2);
  CHECK(v.temporal_patch_size == 2);
  CHECK(v.out_hidden_size == 2560);
  CHECK(v.num_position_embeddings == 2304);
  CHECK(v.merge_unit() == 4);  // 4 patches -> 1 LLM token
  CHECK(v.deepstack_visual_indexes == std::vector<int>{5, 11, 17});
}

TEST_CASE("ModelConfig: text-only model has no vision tower, no M-RoPE") {
  auto j = nlohmann::json::parse(R"({
    "num_hidden_layers": 2, "hidden_size": 8, "num_attention_heads": 4,
    "num_key_value_heads": 2, "vocab_size": 100, "intermediate_size": 16,
    "rope_theta": 10000.0, "rms_norm_eps": 1e-6
  })");
  ModelConfig c = ModelConfig::from_json(j);
  CHECK_FALSE(c.has_vision_tower());
  CHECK(c.mrope_section.empty());
  CHECK_FALSE(c.mrope_interleaved);
  CHECK(c.image_token_id == -1);
  CHECK(c.vision_start_token_id == -1);
}

TEST_CASE("ModelConfig defaults: non-hybrid model has no linear layers, full rotary") {
  auto j = nlohmann::json::parse(R"({
    "num_hidden_layers": 2, "hidden_size": 8, "num_attention_heads": 4,
    "num_key_value_heads": 2, "vocab_size": 100, "intermediate_size": 16,
    "rope_theta": 10000.0, "rms_norm_eps": 1e-6
  })");
  ModelConfig c = ModelConfig::from_json(j);
  CHECK(c.full_attention_interval == 0);
  CHECK_FALSE(c.is_linear_layer(0));
  CHECK_FALSE(c.is_linear_layer(1));
  CHECK_FALSE(c.attn_output_gate);
  CHECK(c.partial_rotary_factor == doctest::Approx(1.0f));  // full rotary
}

TEST_CASE("ModelConfig parses mixed-precision quantization overrides") {
  auto j = nlohmann::json::parse(R"({
    "num_hidden_layers": 2, "hidden_size": 8, "num_attention_heads": 4,
    "num_key_value_heads": 2, "vocab_size": 100, "intermediate_size": 16,
    "rope_theta": 10000.0, "rms_norm_eps": 1e-6,
    "quantization": {
      "group_size": 64,
      "bits": 4,
      "model.layers.0.mlp.down_proj": {"group_size": 32, "bits": 8},
      "model.layers.1.self_attn.q_proj": false
    }
  })");
  ModelConfig c = ModelConfig::from_json(j);
  CHECK(c.quantized);
  CHECK(c.quant_group_size == 64);  // top-level defaults
  CHECK(c.quant_bits == 4);

  // A module without an override resolves to the defaults.
  CHECK(c.quant_for("model.layers.0.self_attn.q_proj").group_size == 64);
  CHECK(c.quant_for("model.layers.0.self_attn.q_proj").bits == 4);

  // The per-module object override wins.
  CHECK(c.quant_for("model.layers.0.mlp.down_proj").group_size == 32);
  CHECK(c.quant_for("model.layers.0.mlp.down_proj").bits == 8);

  // A bare `false` is not recorded as an override (the module is left dense and
  // simply has no ".scales" tensor); it resolves to defaults if queried.
  CHECK(c.quant_overrides.count("model.layers.1.self_attn.q_proj") == 0);
}
