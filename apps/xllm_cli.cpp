// xllm-cli — command-line entry point.
//
//   xllm-cli                      build smoke test (XLLM-001): add two arrays,
//                                 eval, print the sum.
//   xllm-cli dump-weights <dir>   load a model dir's weights (XLLM-004): print
//                                 every key -> shape -> dtype, assert fp16, and
//                                 report peak resident memory.
//
// The real single-stream generation loop arrives in XLLM-015.

#include <cstdio>
#include <string>
#include <vector>

#include "mlx/mlx.h"

#include "core/weights.h"

namespace mx = mlx::core;

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

}  // namespace

int main(int argc, char** argv) {
  if (argc >= 2 && std::string(argv[1]) == "dump-weights") {
    if (argc < 3) {
      std::fprintf(stderr, "usage: xllm-cli dump-weights <model_dir>\n");
      return 2;
    }
    return run_dump_weights(argv[2]);
  }
  return run_smoke();
}
