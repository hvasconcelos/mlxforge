// Quantized (triplet) KV storage: growth, round-trip, and batch surgery
// (filter/merge/pad_dummies) on tiny tensors — no model required. The gate in
// every case: the quantized cache dequantizes to exactly what quantizing the
// same dense values would give (quantization is per position, so storage
// surgery and quantization commute).
#include <doctest/doctest.h>

#include <vector>

#include "cache/batch_kv_cache.h"
#include "cache/kv_cache.h"

#include "mlx/ops.h"
#include "mlx/transforms.h"

using namespace mlxforge;
namespace mx = mlx::core;

namespace {

// Deterministic varied values (constants would quantize trivially).
mx::array varied(int B, int H, int L, int D, float phase) {
  mx::array a = mx::arange(static_cast<float>(B * H * L * D));
  a = mx::sin(mx::add(mx::multiply(a, mx::array(0.37f)), mx::array(phase)));
  return mx::astype(mx::reshape(a, {B, H, L, D}), mx::float16);
}

// Quantize-then-dequantize of dense values: the exact fp16 content a quantized
// cache must reproduce for the same written positions.
mx::array qdq(const mx::array& x, const KVQuantConfig& qc) {
  std::vector<mx::array> t = mx::quantize(x, qc.group_size, qc.bits);
  return mx::dequantize(t[0], t[1], t[2], qc.group_size, qc.bits);
}

bool same(const mx::array& a, const mx::array& b) {
  mx::array eq = mx::allclose(a, b, /*rtol=*/0.0, /*atol=*/1e-6);
  mx::eval(eq);
  return eq.item<bool>();
}

std::vector<int> read_ints(const mx::array& a) {
  mx::array c = mx::contiguous(mx::astype(a, mx::int32));
  mx::eval(c);
  const int32_t* p = c.data<int32_t>();
  return std::vector<int>(p, p + c.size());
}

constexpr int kD = 64;  // head_dim must be a multiple of group_size

}  // namespace

TEST_CASE("quantized KVCache round-trips appended K/V") {
  for (int bits : {8, 4}) {
    CAPTURE(bits);
    const KVQuantConfig qc{bits, 64};
    KVCache cache(/*n_layers=*/1, qc);
    CHECK(cache.quantized());

    mx::array k1 = varied(1, 2, 5, kD, 0.1f), v1 = varied(1, 2, 5, kD, 0.2f);
    mx::array k2 = varied(1, 2, 1, kD, 0.3f), v2 = varied(1, 2, 1, kD, 0.4f);
    cache.update_and_fetch_quantized(0, k1, v1);
    cache.advance(5);
    QuantizedKVSlice s = cache.update_and_fetch_quantized(0, k2, v2);
    cache.advance(1);

    mx::array got_k = mx::dequantize(s.k.w, s.k.scales, s.k.biases, qc.group_size, qc.bits);
    mx::array got_v = mx::dequantize(s.v.w, s.v.scales, s.v.biases, qc.group_size, qc.bits);
    CHECK(same(got_k, qdq(mx::concatenate({k1, k2}, 2), qc)));
    CHECK(same(got_v, qdq(mx::concatenate({v1, v2}, 2), qc)));
  }
}

TEST_CASE("dense and quantized cache APIs are mutually exclusive") {
  mx::array k = varied(1, 1, 1, kD, 0.0f);
  KVCache dense(1);
  CHECK_THROWS(dense.update_and_fetch_quantized(0, k, k));
  KVCache quant(1, KVQuantConfig{8, 64});
  CHECK_THROWS(quant.update_and_fetch(0, k, k));
  CHECK_THROWS(quant.fetch(0));

  BatchKVCache bdense(1, {0});
  CHECK_THROWS(bdense.update_and_fetch_quantized(0, k, k));
  BatchKVCache bquant(1, {0}, KVQuantConfig{8, 64});
  CHECK_THROWS(bquant.update_and_fetch(0, k, k));
}

TEST_CASE("quantized BatchKVCache grows across a 256 boundary and preserves contents") {
  const KVQuantConfig qc{8, 64};
  BatchKVCache cache(/*n_layers=*/1, /*left_padding=*/{0}, qc);

  mx::array k1 = varied(1, 1, 200, kD, 0.1f), v1 = varied(1, 1, 200, kD, 0.2f);
  cache.update_and_fetch_quantized(0, k1, v1);
  cache.advance(200);
  CHECK(cache.s_cap() == 256);

  mx::array k2 = varied(1, 1, 100, kD, 0.3f), v2 = varied(1, 1, 100, kD, 0.4f);
  cache.update_and_fetch_quantized(0, k2, v2);
  cache.advance(100);
  CHECK(cache.s_cap() == 456);  // exactly one growth: 200 (trimmed) + 256

  auto [k, v] = cache.fetch_dequantized(0);
  CHECK(k.shape() == mx::Shape{1, 1, 300, kD});
  CHECK(same(k, qdq(mx::concatenate({k1, k2}, 2), qc)));
  CHECK(same(v, qdq(mx::concatenate({v1, v2}, 2), qc)));
}

TEST_CASE("quantized batch surgery (pad_dummies/filter/merge) matches dense + quantize") {
  const KVQuantConfig qc{8, 64};

  // Mirror every write and surgery op on a dense cache and a quantized one.
  BatchKVCache dense(1, {1, 0});
  BatchKVCache quant(1, {1, 0}, qc);
  mx::array k = varied(2, 1, 4, kD, 0.1f), v = varied(2, 1, 4, kD, 0.2f);
  dense.update_and_fetch(0, k, v);
  quant.update_and_fetch_quantized(0, k, v);
  dense.advance(4);
  quant.advance(4);

  auto check_equal = [&](const char* what) {
    CAPTURE(what);
    auto [dk, dv] = dense.fetch(0);
    auto [qk, qv] = quant.fetch_dequantized(0);
    CHECK(same(qk, qdq(dk, qc)));
    CHECK(same(qv, qdq(dv, qc)));
    CHECK(quant.idx() == dense.idx());
    CHECK(quant.batch_size() == dense.batch_size());
    CHECK(read_ints(quant.offset()) == read_ints(dense.offset()));
    CHECK(read_ints(quant.left_padding()) == read_ints(dense.left_padding()));
  };

  dense.pad_dummies(2);
  quant.pad_dummies(2);
  check_equal("pad_dummies");

  dense.filter({0, 1});
  quant.filter({0, 1});
  check_equal("filter trims dummies");

  // Admit a freshly prefilled pair of rows with a different length.
  BatchKVCache dense_in(1, {0, 2});
  BatchKVCache quant_in(1, {0, 2}, qc);
  mx::array k2 = varied(2, 1, 7, kD, 0.3f), v2 = varied(2, 1, 7, kD, 0.4f);
  dense_in.update_and_fetch(0, k2, v2);
  quant_in.update_and_fetch_quantized(0, k2, v2);
  dense_in.advance(7);
  quant_in.advance(7);

  dense.merge(dense_in);
  quant.merge(quant_in);
  check_equal("merge right-justifies and concatenates");

  dense.filter({1, 2});
  quant.filter({1, 2});
  check_equal("filter drops the common left padding");

  // eval_state must materialize every triplet component without error.
  quant.eval_state();
}

TEST_CASE("merge rejects mismatched KV quantization configs") {
  BatchKVCache a(1, {0}, KVQuantConfig{8, 64});
  BatchKVCache b(1, {0});
  CHECK_THROWS(a.merge(b));
}
