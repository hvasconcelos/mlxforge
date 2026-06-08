// Self-contained GGUF loader: config + tokenizer material + quantized weights
// parsed from a single .gguf, and a forward pass that matches the golden first
// token. Self-skips when no local GGUF model is present (MLXFORGE_GGUF_MODEL).
#include <doctest/doctest.h>

#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>

#include "core/gguf.h"
#include "model/llama.h"
#include "runtime/single_stream.h"
#include "support/reference.h"

namespace {
std::string gguf_path() { return MLXFORGE_GGUF_MODEL; }
bool gguf_available() {
  return !gguf_path().empty() && std::ifstream(gguf_path()).good();
}

// Append a little-endian scalar to a byte buffer.
template <typename T>
void put(std::string& b, T v) {
  b.append(reinterpret_cast<const char*>(&v), sizeof(T));
}
// Append a GGUF string (u64 length + bytes).
void put_str(std::string& b, const std::string& s) {
  put<uint64_t>(b, s.size());
  b += s;
}
}  // namespace

TEST_CASE("GGUF loader parses llama config from metadata") {
  if (!gguf_available()) {
    MESSAGE("MLXFORGE_GGUF_MODEL not present; skipping");
    return;
  }
  mlxforge::GgufModel g = mlxforge::load_gguf_model(gguf_path());
  const auto& c = g.config;

  CHECK(c.model_type == "llama");
  CHECK(c.n_layers == 16);
  CHECK(c.hidden == 2048);
  CHECK(c.n_heads == 32);
  CHECK(c.n_kv_heads == 8);
  CHECK(c.head_dim == 64);
  CHECK(c.intermediate_size == 8192);
  CHECK(c.vocab == 128256);
  CHECK(c.rope_theta == doctest::Approx(500000.0f));
  CHECK(c.rms_eps == doctest::Approx(1e-5f));
  CHECK(c.bos_token_id == 128000);

  // tied embeddings (no separate output.weight in this checkpoint)
  CHECK(c.tie_word_embeddings);

  // llama3 rope rescaling is baked into rope_freqs.weight -> head_dim/2 factors
  REQUIRE(c.rope_freq_factors.has_value());
  CHECK(c.rope_freq_factors->size() == 32);
}

TEST_CASE("GGUF loader remaps weights and tags quantization per-weight") {
  if (!gguf_available()) {
    MESSAGE("MLXFORGE_GGUF_MODEL not present; skipping");
    return;
  }
  mlxforge::GgufModel g = mlxforge::load_gguf_model(gguf_path());
  const auto& w = g.weights;

  // canonical HF keys exist
  CHECK(w.has("model.embed_tokens.weight"));
  CHECK(w.has("model.norm.weight"));
  CHECK(w.has("model.layers.0.self_attn.q_proj.weight"));
  CHECK(w.has("model.layers.0.mlp.down_proj.weight"));
  CHECK(w.has("model.layers.15.post_attention_layernorm.weight"));
  // ggml-only names are gone
  CHECK_FALSE(w.has("blk.0.attn_q.weight"));
  CHECK_FALSE(w.has("rope_freqs.weight"));

  // Quantization is per-weight and depends on the file's quant type: legacy
  // Q4_0/Q4_1/Q8_0 projections stay quantized at group_size 32; K-quant files
  // (Q4_K_M, …) arrive dequantized to dense fp16. Whichever it is, an existing
  // ".scales" sibling must agree with is_quantized.
  mlxforge::QuantParams qp;
  if (w.has("model.layers.0.self_attn.q_proj.scales")) {
    CHECK(w.is_quantized("model.layers.0.self_attn.q_proj.weight", qp));
    CHECK(qp.group_size == 32);
    CHECK((qp.bits == 4 || qp.bits == 8));
  } else {
    CHECK_FALSE(w.is_quantized("model.layers.0.self_attn.q_proj.weight", qp));
  }

  // tokenizer raw material is present (rebuilt in Stage 2)
  CHECK(g.tokens.size() == 128256);
  CHECK(g.merges.size() > 0);
  CHECK(g.pre == "llama-bpe");
  CHECK(g.bos_id == 128000);
}

TEST_CASE("GGUF loader rejects a non-llama architecture") {
  // Hand-craft a minimal GGUF: magic, version, 0 tensors, 1 metadata KV
  // (general.architecture = "gpt2"). No model needed.
  std::string b;
  put<uint32_t>(b, 0x46554747);  // "GGUF"
  put<uint32_t>(b, 3);           // version
  put<uint64_t>(b, 0);           // tensor count
  put<uint64_t>(b, 1);           // metadata kv count
  put_str(b, "general.architecture");
  put<uint32_t>(b, 8);  // value type: STRING
  put_str(b, "gpt2");

  const std::string path =
      (std::filesystem::temp_directory_path() / "mlxforge_nonllama.gguf").string();
  { std::ofstream(path, std::ios::binary).write(b.data(), b.size()); }

  CHECK_THROWS_AS(mlxforge::load_gguf_config_and_tokenizer(path), std::runtime_error);
  std::remove(path.c_str());
}

TEST_CASE("GGUF loader maps Qwen3-MoE config and stacks experts into switch_mlp") {
  // Hand-craft a tiny qwen3moe GGUF (1 layer, 2 experts, hidden=4, moe_inter=3)
  // with only the MoE tensors present, and validate the structural remap +
  // config without needing a multi-GB model. The expert tensors are F32 here;
  // the quant layouts reuse the same (E, out, in) orientation this asserts.
  constexpr uint32_t kF32 = 0;
  constexpr uint64_t kAlign = 32;

  std::string kv;
  uint64_t nkv = 0;
  auto kv_str = [&](const std::string& k, const std::string& v) {
    put_str(kv, k); put<uint32_t>(kv, 8); put_str(kv, v); ++nkv;
  };
  auto kv_u32 = [&](const std::string& k, uint32_t v) {
    put_str(kv, k); put<uint32_t>(kv, 4); put<uint32_t>(kv, v); ++nkv;
  };
  auto kv_f32 = [&](const std::string& k, float v) {
    put_str(kv, k); put<uint32_t>(kv, 6); put<float>(kv, v); ++nkv;
  };
  kv_str("general.architecture", "qwen3moe");
  kv_u32("qwen3moe.block_count", 1);
  kv_u32("qwen3moe.embedding_length", 4);
  kv_u32("qwen3moe.attention.head_count", 2);
  kv_u32("qwen3moe.attention.head_count_kv", 1);
  kv_f32("qwen3moe.attention.layer_norm_rms_epsilon", 1e-6f);
  kv_u32("qwen3moe.expert_count", 2);
  kv_u32("qwen3moe.expert_used_count", 2);
  kv_u32("qwen3moe.expert_feed_forward_length", 3);

  // ggml stores dims innermost-first (ne); the loader reverses them, so e.g.
  // ffn_gate_exps ne {in=4, out=3, expert=2} lands as shape (2, 3, 4).
  struct Tensor { std::string name; std::vector<uint64_t> ne; };
  const std::vector<Tensor> tensors = {
      {"blk.0.ffn_gate_inp.weight", {4, 2}},     // router -> (E=2, in=4)
      {"blk.0.ffn_gate_exps.weight", {4, 3, 2}}, // -> (E=2, out=3, in=4)
      {"blk.0.ffn_up_exps.weight", {4, 3, 2}},   // -> (E=2, out=3, in=4)
      {"blk.0.ffn_down_exps.weight", {3, 4, 2}}, // -> (E=2, out=4, in=3)
  };

  std::string tinfo, data;
  uint64_t offset = 0;
  for (const auto& t : tensors) {
    put_str(tinfo, t.name);
    put<uint32_t>(tinfo, static_cast<uint32_t>(t.ne.size()));
    for (uint64_t d : t.ne) put<uint64_t>(tinfo, d);
    put<uint32_t>(tinfo, kF32);
    put<uint64_t>(tinfo, offset);
    uint64_t n = 1;
    for (uint64_t d : t.ne) n *= d;
    for (uint64_t i = 0; i < n; ++i) put<float>(data, 0.0f);
    const uint64_t bytes = n * 4;
    const uint64_t pad = (kAlign - (bytes % kAlign)) % kAlign;
    data.append(pad, '\0');
    offset += bytes + pad;
  }

  std::string b;
  put<uint32_t>(b, 0x46554747);  // "GGUF"
  put<uint32_t>(b, 3);           // version
  put<uint64_t>(b, tensors.size());
  put<uint64_t>(b, nkv);
  b += kv;
  b += tinfo;
  b.append((kAlign - (b.size() % kAlign)) % kAlign, '\0');  // align to data_start
  b += data;

  const std::string path =
      (std::filesystem::temp_directory_path() / "mlxforge_qwen3moe.gguf").string();
  { std::ofstream(path, std::ios::binary).write(b.data(), b.size()); }

  mlxforge::GgufModel g = mlxforge::load_gguf_model(path);
  std::remove(path.c_str());

  CHECK(g.config.model_type == "qwen3_moe");  // matches the safetensors canonical type
  CHECK(g.config.num_experts == 2);
  CHECK(g.config.num_experts_per_tok == 2);
  CHECK(g.config.moe_intermediate_size == 3);
  CHECK(g.config.intermediate_size == 3);  // falls back to the expert width

  auto dims = [&](const std::string& k) {
    const auto& s = g.weights.tensors.at(k).shape();
    return std::vector<int>(s.begin(), s.end());
  };
  REQUIRE(g.weights.has("model.layers.0.mlp.switch_mlp.gate_proj.weight"));
  CHECK(g.weights.has("model.layers.0.mlp.gate.weight"));
  CHECK(dims("model.layers.0.mlp.gate.weight") == std::vector<int>{2, 4});
  CHECK(dims("model.layers.0.mlp.switch_mlp.gate_proj.weight") == std::vector<int>{2, 3, 4});
  CHECK(dims("model.layers.0.mlp.switch_mlp.up_proj.weight") == std::vector<int>{2, 3, 4});
  CHECK(dims("model.layers.0.mlp.switch_mlp.down_proj.weight") == std::vector<int>{2, 4, 3});
  // ggml-only names are gone.
  CHECK_FALSE(g.weights.has("blk.0.ffn_gate_exps.weight"));
}

TEST_CASE("GGUF loader rejects Qwen3.5/qwen3next as unsupported") {
  std::string b;
  put<uint32_t>(b, 0x46554747);  // "GGUF"
  put<uint32_t>(b, 3);           // version
  put<uint64_t>(b, 0);           // tensor count
  put<uint64_t>(b, 1);           // metadata kv count
  put_str(b, "general.architecture");
  put<uint32_t>(b, 8);  // STRING
  put_str(b, "qwen3next");

  const std::string path =
      (std::filesystem::temp_directory_path() / "mlxforge_qwen3next.gguf").string();
  { std::ofstream(path, std::ios::binary).write(b.data(), b.size()); }
  CHECK_THROWS_AS(mlxforge::load_gguf_config_and_tokenizer(path), std::runtime_error);
  std::remove(path.c_str());
}

TEST_CASE("GGUF forward pass produces the golden first token") {
  if (!gguf_available()) {
    MESSAGE("MLXFORGE_GGUF_MODEL not present; skipping");
    return;
  }
  using namespace mlxforge::test;
  mlxforge::GgufModel g = mlxforge::load_gguf_model(gguf_path());
  mlxforge::LlamaModel model(g.config, std::move(g.weights));

  // prompt_0 is "The capital of France is"; the golden argmax is the first
  // generated token ("Paris"). A Q4_0 GGUF must reproduce it, validating the
  // remap + group_size-32 quantized_matmul + baked-rope path end to end.
  std::vector<int> prompt = load_token_ids("prompt_0_ids.npy");
  std::vector<int> golden = load_token_ids("argmax.npy");

  mlxforge::GenerateResult r =
      mlxforge::greedy_generate(model, prompt, /*max_tokens=*/1, /*eos_ids=*/{});
  REQUIRE(r.tokens.size() == 1);
  CHECK(r.tokens[0] == golden[0]);
}
