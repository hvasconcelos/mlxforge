// optional 4-bit quantization — quantized greedy output stays coherent
// and the resident footprint drops well below the fp16 model.
#include <doctest/doctest.h>

#include <fstream>
#include <string>

#include "core/config.h"
#include "core/weights.h"
#include "model/llama.h"
#include "runtime/single_stream.h"
#include "tokenizer/tokenizer.h"

#include "mlx/memory.h"

namespace mx = mlx::core;

namespace {
std::string dir4() { return MLXFORGE_MODEL_DIR_4BIT; }
bool available() {
  return !dir4().empty() && std::ifstream(dir4() + "/config.json").good();
}
}  // namespace

TEST_CASE("4-bit quantization is detected from config") {
  if (!available()) {
    MESSAGE("4-bit model not present; skipping");
    return;
  }
  mlxforge::ModelConfig cfg = mlxforge::ModelConfig::from_file(dir4() + "/config.json");
  CHECK(cfg.quantized);
  CHECK(cfg.quant_bits == 4);
  CHECK(cfg.quant_group_size == 64);
}

TEST_CASE("quantized greedy output stays coherent; footprint ~0.7 GiB") {
  if (!available()) {
    MESSAGE("4-bit model not present; skipping");
    return;
  }
  mlxforge::ModelConfig cfg = mlxforge::ModelConfig::from_file(dir4() + "/config.json");
  mlxforge::Tokenizer tok = mlxforge::Tokenizer::from_file(dir4() + "/tokenizer.json");

  mx::reset_peak_memory();
  mlxforge::LlamaModel model(cfg, mlxforge::load_weights(dir4(), cfg));

  std::vector<int> prompt = tok.encode("The capital of France is");
  mlxforge::GenerateResult r = mlxforge::greedy_generate(model, prompt, /*max_tokens=*/12, cfg.eos_token_ids);
  std::string text = tok.decode(r.tokens);
  INFO("generated: " << text);
  CHECK(text.find("Paris") != std::string::npos);  // coherent vs the fp16 model

  const double gib = static_cast<double>(mx::get_peak_memory()) / (1024.0 * 1024.0 * 1024.0);
  INFO("peak memory GiB: " << gib);
  CHECK(gib < 1.5);  // far below the fp16 model's ~2.3 GiB
}
