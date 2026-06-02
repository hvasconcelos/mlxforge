// BatchKVCache layout, update_and_fetch, offset/left-padding
// bookkeeping (tiny tensors, no model/GPU eval required beyond a single step).
#include <doctest/doctest.h>

#include <vector>

#include "cache/batch_kv_cache.h"

#include "mlx/ops.h"
#include "mlx/transforms.h"

using namespace mlxforge;
namespace mx = mlx::core;

namespace {
std::vector<int> read_ints(const mx::array& a) {
  mx::array c = mx::contiguous(mx::astype(a, mx::int32));
  mx::eval(c);
  const int32_t* p = c.data<int32_t>();
  return std::vector<int>(p, p + c.size());
}
std::vector<float> read_floats(const mx::array& a) {
  mx::array c = mx::contiguous(mx::astype(a, mx::float32));
  mx::eval(c);
  const float* p = c.data<float>();
  return std::vector<float>(p, p + c.size());
}
// A (B, H, L, D) tensor filled with `value`.
mx::array filled(int B, int H, int L, int D, float value) {
  return mx::full({B, H, L, D}, value, mx::float16);
}
}  // namespace

TEST_CASE("offset/left_padding/idx track per-row values across writes") {
  BatchKVCache cache(/*n_layers=*/1, /*left_padding=*/{1, 0});  // B=2
  CHECK(cache.idx() == 0);
  CHECK(cache.s_cap() == 0);
  CHECK(read_ints(cache.offset()) == std::vector<int>{-1, 0});
  CHECK(read_ints(cache.left_padding()) == std::vector<int>{1, 0});

  // Prefill 3 tokens.
  auto [k, v] = cache.update_and_fetch(0, filled(2, 1, 3, 2, 1.0f), filled(2, 1, 3, 2, 1.0f));
  cache.advance(3);
  CHECK(k.shape() == mx::Shape{2, 1, 3, 2});  // returns the populated slice
  CHECK(cache.idx() == 3);
  CHECK(cache.s_cap() == 256);  // one 256 block
  CHECK(read_ints(cache.offset()) == std::vector<int>{2, 3});  // -lp + 3
  CHECK(read_ints(cache.left_padding()) == std::vector<int>{1, 0});

  // Decode one more token.
  cache.update_and_fetch(0, filled(2, 1, 1, 2, 1.0f), filled(2, 1, 1, 2, 1.0f));
  cache.advance(1);
  CHECK(cache.idx() == 4);
  CHECK(read_ints(cache.offset()) == std::vector<int>{3, 4});
}

TEST_CASE("crossing a 256 boundary grows once and preserves contents") {
  BatchKVCache cache(/*n_layers=*/1, /*left_padding=*/{0});  // B=1, H=1, D=1

  cache.update_and_fetch(0, filled(1, 1, 200, 1, 1.0f), filled(1, 1, 200, 1, 1.0f));
  cache.advance(200);
  CHECK(cache.s_cap() == 256);  // first allocation

  auto [k, v] = cache.update_and_fetch(0, filled(1, 1, 100, 1, 2.0f), filled(1, 1, 100, 1, 2.0f));
  cache.advance(100);
  CHECK(cache.s_cap() == 456);  // exactly one growth: 200 (trimmed) + 256
  CHECK(k.shape() == mx::Shape{1, 1, 300, 1});

  std::vector<float> vals = read_floats(k);  // 300 sequence values
  REQUIRE(vals.size() == 300);
  CHECK(vals[0] == doctest::Approx(1.0f));
  CHECK(vals[199] == doctest::Approx(1.0f));   // first write preserved
  CHECK(vals[200] == doctest::Approx(2.0f));   // second write
  CHECK(vals[299] == doctest::Approx(2.0f));
}
