// BatchKVCache filter (eviction) and merge (admission).
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
// (B, 1, L, 1) where row b is filled with the value b (to track rows).
mx::array row_tagged(int B, int L) {
  mx::array rows = mx::reshape(mx::arange(B, mx::float16), {B, 1, 1, 1});
  return mx::broadcast_to(rows, {B, 1, L, 1});
}
}  // namespace

TEST_CASE("filter drops the right rows (including first and last)") {
  BatchKVCache cache(/*n_layers=*/1, /*left_padding=*/{0, 0, 0, 0});  // B=4
  mx::array kv = row_tagged(4, 2);  // rows tagged 0,1,2,3
  cache.update_and_fetch(0, kv, kv);
  cache.advance(2);

  cache.filter({1, 2});  // drop row 0 and row 3
  CHECK(cache.batch_size() == 2);
  CHECK(read_ints(cache.offset()) == std::vector<int>{2, 2});
  CHECK(read_ints(cache.left_padding()) == std::vector<int>{0, 0});

  auto [k, v] = cache.fetch(0);
  CHECK(k.shape() == mx::Shape{2, 1, 2, 1});
  std::vector<float> vals = read_floats(k);  // 2 rows * 2 positions
  CHECK(vals[0] == doctest::Approx(1.0f));   // surviving row 1
  CHECK(vals[2] == doctest::Approx(2.0f));   // surviving row 2
}

TEST_CASE("filter's common-left-padding shift reduces idx") {
  BatchKVCache cache(/*n_layers=*/1, /*left_padding=*/{2, 5, 3});  // B=3
  mx::array kv = row_tagged(3, 4);
  cache.update_and_fetch(0, kv, kv);
  cache.advance(4);  // idx=4

  cache.filter({1, 2});  // keep rows with left_padding 5 and 3 -> common min 3
  CHECK(cache.batch_size() == 2);
  CHECK(cache.idx() == 1);  // 4 - min_left_pad(3)
  CHECK(read_ints(cache.left_padding()) == std::vector<int>{2, 0});  // 5-3, 3-3
}

TEST_CASE("merge of two caches with different S_cap") {
  BatchKVCache a(/*n_layers=*/1, /*left_padding=*/{0, 0});  // B=2
  a.update_and_fetch(0, row_tagged(2, 300), row_tagged(2, 300));
  a.advance(300);
  CHECK(a.s_cap() == 512);  // ceil(300/256)*256

  BatchKVCache b(/*n_layers=*/1, /*left_padding=*/{0});  // B=1
  b.update_and_fetch(0, mx::full({1, 1, 100, 1}, 9.0f, mx::float16),
                     mx::full({1, 1, 100, 1}, 9.0f, mx::float16));
  b.advance(100);
  CHECK(b.s_cap() == 256);

  a.merge(b);
  CHECK(a.batch_size() == 3);
  CHECK(a.idx() == 300);  // max(300, 100)
  // offset unchanged per row, concatenated: a=[300,300], b=[100].
  CHECK(read_ints(a.offset()) == std::vector<int>{300, 300, 100});
  // b was right-justified: it gained left padding of max_idx - 100 = 200.
  CHECK(read_ints(a.left_padding()) == std::vector<int>{0, 0, 200});

  auto [k, v] = a.fetch(0);
  CHECK(k.shape() == mx::Shape{3, 1, 300, 1});
}
