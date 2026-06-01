// xllm-cli — command-line entry point.
//
//   xllm-cli                      build smoke test (XLLM-001): add two arrays,
//                                 eval, print the sum.
//   xllm-cli dump-weights <dir>   load a model dir's weights (XLLM-004): print
//                                 every key -> shape -> dtype, assert fp16, and
//                                 report peak resident memory.
//   xllm-cli generate <dir> <prompt_ids.npy> [max_tokens]
//                                 greedy single-stream generation (XLLM-015):
//                                 prefill the pre-tokenized prompt and stream
//                                 generated token ids to stdout until EOS or
//                                 max_tokens (real text awaits the tokenizer,
//                                 XLLM-021).

#include <cstdio>
#include <string>
#include <vector>

#include "mlx/mlx.h"

#include "core/config.h"
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
    std::fprintf(stderr, "error: Metal GPU is not available on this machine\n");
    return 1;
  }
  std::printf("Metal available: yes\n");

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
  xllm::Weights w = xllm::load_weights(dir);

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
  std::printf("\n%zu tensors loaded; %zu non-fp16; peak memory %.2f GiB\n", w.size(), non_fp16,
              gib);
  return non_fp16 == 0 ? 0 : 1;
}

int run_generate(const std::string& dir, const std::string& prompt_arg, int max_tokens) {
  xllm::ModelConfig cfg = xllm::ModelConfig::from_file(dir + "/config.json");
  xllm::LlamaModel model(cfg, xllm::load_weights(dir));
  xllm::Tokenizer tok = xllm::Tokenizer::from_file(dir + "/tokenizer.json");

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
  xllm::StreamingDetokenizer detok(tok);
  xllm::GenerateResult r =
      xllm::greedy_generate(model, prompt, max_tokens, cfg.eos_token_ids, [&](int id) {
        std::string piece = detok.add(id);
        std::fwrite(piece.data(), 1, piece.size(), stdout);
        std::fflush(stdout);
      });
  std::string tail = detok.finish();
  std::fwrite(tail.data(), 1, tail.size(), stdout);
  std::printf("\n[%zu tokens%s]\n", r.tokens.size(), r.hit_eos ? ", stopped at EOS" : "");
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  const std::string cmd = argc >= 2 ? argv[1] : "";
  if (cmd == "dump-weights") {
    if (argc < 3) {
      std::fprintf(stderr, "usage: xllm-cli dump-weights <model_dir>\n");
      return 2;
    }
    return run_dump_weights(argv[2]);
  }
  if (cmd == "generate") {
    if (argc < 4) {
      std::fprintf(stderr, "usage: xllm-cli generate <model_dir> <prompt_ids.npy> [max_tokens]\n");
      return 2;
    }
    const int max_tokens = argc >= 5 ? std::stoi(argv[4]) : 64;
    return run_generate(argv[2], argv[3], max_tokens);
  }
  return run_smoke();
}
