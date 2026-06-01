// xllm-cli — for XLLM-001 this is the build smoke test and the MLX lazy-eval
// lesson: build two arrays, add them, force evaluation, print the result.
// The real single-stream generation loop arrives in XLLM-015.

#include <cstdio>

#include "mlx/mlx.h"

namespace mx = mlx::core;

int main() {
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
