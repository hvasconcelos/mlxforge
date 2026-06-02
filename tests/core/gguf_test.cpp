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
