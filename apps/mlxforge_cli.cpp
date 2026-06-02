// mlxforge-cli — command-line entry point.
//
//   mlxforge-cli                      build smoke test: add two arrays, eval,
//                                 print the sum.
//   mlxforge-cli dump-weights <dir>   load a model dir's weights: print every
//                                 key -> shape -> dtype, assert fp16, and report
//                                 peak resident memory.
//   mlxforge-cli generate <dir> <prompt> [max_tokens]
//                                 greedy single-stream generation: prefill the
//                                 prompt (raw text via the chat template, or a
//                                 pre-tokenized .npy of ids) and stream the
//                                 detokenized text to stdout until EOS or
//                                 max_tokens.

#include <cstdio>
#include <string>
#include <vector>

#include "mlx/mlx.h"

#include "core/config.h"
#include "core/logging.h"
#include "core/weights.h"
#include "model/llama.h"
#include "runtime/single_stream.h"
#include "tokenizer/tokenizer.h"

namespace mx = mlx::core;

namespace {
bool ends_with(const std::string& s, const std::string& suffix) {
  return s.size() >= suffix.size() && s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}
}  // namespace

namespace {

int run_smoke() {
  // MLX is lazy: ops build a graph; nothing runs until eval() is called.
  if (!mx::metal::is_available()) {
    mlxforge::log::error("Metal GPU is not available on this machine");
    return 1;
  }
  mlxforge::log::info("Metal available: yes");

  mx::array a({1.0f, 2.0f, 3.0f, 4.0f});
  mx::array b({10.0f, 20.0f, 30.0f, 40.0f});
  mx::array c = mx::add(a, b);

  mx::eval(c);  // force the lazy graph to actually compute

  const float* data = c.data<float>();
  std::printf("a + b = [");
  for (int i = 0; i < c.size(); ++i) {
    std::printf("%g%s", data[i], i + 1 < c.size() ? ", " : "");
  }
  std::printf("]\n");
  return 0;
}

int run_dump_weights(const std::string& dir) {
  mx::reset_peak_memory();
  mlxforge::Weights w = mlxforge::load_weights(dir);

  // Materialize so the fp16 cast and resident-memory figure are real.
  std::vector<mx::array> all;
  all.reserve(w.tensors.size());
  for (auto& [_, a] : w.tensors) all.push_back(a);
  mx::eval(all);

  std::printf("%s", w.summary().c_str());

  size_t non_fp16 = 0;
  for (const auto& [_, a] : w.tensors) {
    if (a.dtype() != mx::float16) ++non_fp16;
  }
  const double gib = static_cast<double>(mx::get_peak_memory()) / (1024.0 * 1024.0 * 1024.0);
  mlxforge::log::info("{} tensors loaded; {} non-fp16; peak memory {:.2f} GiB", w.size(), non_fp16,
                      gib);
  if (non_fp16 > 0) mlxforge::log::warn("{} tensors are not fp16", non_fp16);
  return non_fp16 == 0 ? 0 : 1;
}

int run_generate(const std::string& dir, const std::string& prompt_arg, int max_tokens) {
  mlxforge::ModelConfig cfg = mlxforge::ModelConfig::from_file(dir + "/config.json");
  mlxforge::LlamaModel model(cfg, mlxforge::load_weights(dir));
  mlxforge::Tokenizer tok = mlxforge::Tokenizer::from_file(
      dir + "/tokenizer.json", cfg.bos_token_id,
      mlxforge::chat_format_from_model_type(cfg.model_type));

  // A .npy argument is a pre-tokenized prompt; anything else is raw text run
  // through the chat template.
  std::vector<int> prompt;
  if (ends_with(prompt_arg, ".npy")) {
    mx::array ids = mx::contiguous(mx::astype(mx::load(prompt_arg), mx::int32));
    mx::eval(ids);
    prompt.assign(ids.data<int32_t>(), ids.data<int32_t>() + ids.size());
  } else {
    prompt = tok.apply_chat_template({{"user", prompt_arg}});
  }

  // Stream real detokenized text as tokens are produced.
  mlxforge::StreamingDetokenizer detok(tok);
  mlxforge::GenerateResult r =
      mlxforge::greedy_generate(model, prompt, max_tokens, cfg.eos_token_ids, [&](int id) {
        std::string piece = detok.add(id);
        std::fwrite(piece.data(), 1, piece.size(), stdout);
        std::fflush(stdout);
      });
  std::string tail = detok.finish();
  std::fwrite(tail.data(), 1, tail.size(), stdout);
  std::fputc('\n', stdout);
  mlxforge::log::info("generated {} tokens{}", r.tokens.size(),
                      r.hit_eos ? " (stopped at EOS)" : "");
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  mlxforge::log::init();

  const std::string cmd = argc >= 2 ? argv[1] : "";
  if (cmd == "dump-weights") {
    if (argc < 3) {
      std::fprintf(stderr, "usage: mlxforge-cli dump-weights <model_dir>\n");
      return 2;
    }
    return run_dump_weights(argv[2]);
  }
  if (cmd == "generate") {
    if (argc < 4) {
      std::fprintf(stderr, "usage: mlxforge-cli generate <model_dir> <prompt_ids.npy> [max_tokens]\n");
      return 2;
    }
    const int max_tokens = argc >= 5 ? std::stoi(argv[4]) : 64;
    return run_generate(argv[2], argv[3], max_tokens);
  }
  return run_smoke();
}
