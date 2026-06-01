// XLLM-001: trivial test proving the doctest + ctest wiring works, plus a
// minimal MLX add/eval check so the GPU path is exercised under ctest.
#include <doctest/doctest.h>

#include "mlx/mlx.h"

namespace mx = mlx::core;

TEST_CASE("doctest framework is wired into ctest") {
  CHECK(1 + 1 == 2);
}

TEST_CASE("mlx add + eval produces the correct sum") {
  mx::array a({1.0f, 2.0f, 3.0f});
  mx::array b({4.0f, 5.0f, 6.0f});
  mx::array c = mx::add(a, b);
  mx::eval(c);
  CHECK(c.data<float>()[0] == doctest::Approx(5.0f));
  CHECK(c.data<float>()[1] == doctest::Approx(7.0f));
  CHECK(c.data<float>()[2] == doctest::Approx(9.0f));
}
