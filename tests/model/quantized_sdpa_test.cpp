// quantized_sdpa vs fast::scaled_dot_product_attention over the dequantized
// K/V — the same math modulo accumulation order, so fp16-tolerance agreement
// gates the hand-rolled quantized path (and proves this MLX pin broadcasts
// quantized_matmul over the 5-D GQA shapes). No model required.
#include <doctest/doctest.h>

#include <cmath>
#include <limits>
#include <optional>
#include <vector>

#include "model/sdpa.h"

#include "mlx/fast.h"
#include "mlx/ops.h"
#include "mlx/transforms.h"

using namespace mlxforge;
namespace mx = mlx::core;

namespace {

constexpr int kD = 64;  // head_dim (a multiple of the group size)

// Deterministic varied fp16 values.
mx::array varied(int B, int H, int L, int D, float phase) {
  mx::array a = mx::arange(static_cast<float>(B * H * L * D));
  a = mx::sin(mx::add(mx::multiply(a, mx::array(0.61f)), mx::array(phase)));
  return mx::astype(mx::reshape(a, {B, H, L, D}), mx::float16);
}

QuantizedKV quantize_kv(const mx::array& x, const KVQuantConfig& qc) {
  std::vector<mx::array> t = mx::quantize(x, qc.group_size, qc.bits);
  return {t[0], t[1], t[2]};
}

mx::array dequant(const QuantizedKV& t, const KVQuantConfig& qc) {
  return mx::dequantize(t.w, t.scales, t.biases, qc.group_size, qc.bits);
}

void check_close(const mx::array& actual, const mx::array& expected) {
  REQUIRE(actual.shape() == expected.shape());
  mx::array diff = mx::max(mx::abs(mx::subtract(mx::astype(actual, mx::float32),
                                                mx::astype(expected, mx::float32))));
  mx::eval(diff);
  CHECK(diff.item<float>() < 2e-2f);
}

// One comparison: quantized_sdpa over triplets vs fast SDPA over the
// dequantized K/V (so quantization error itself cancels out).
void compare(int B, int n_q_heads, int n_kv_heads, int L, int S, int bits,
             const std::string& mask_mode, const std::optional<mx::array>& mask) {
  CAPTURE(B);
  CAPTURE(n_q_heads);
  CAPTURE(n_kv_heads);
  CAPTURE(L);
  CAPTURE(S);
  CAPTURE(bits);
  const KVQuantConfig qc{bits, 64};
  const float scale = 1.0f / std::sqrt(static_cast<float>(kD));

  mx::array q = varied(B, n_q_heads, L, kD, 0.1f);
  QuantizedKV k = quantize_kv(varied(B, n_kv_heads, S, kD, 0.2f), qc);
  QuantizedKV v = quantize_kv(varied(B, n_kv_heads, S, kD, 0.3f), qc);

  mx::array got = quantized_sdpa(q, k, v, scale, mask_mode, mask, qc.group_size, qc.bits);
  mx::array want = mx::fast::scaled_dot_product_attention(q, dequant(k, qc), dequant(v, qc),
                                                          scale, mask_mode, mask);
  check_close(got, want);
}

// Additive fp16 left-padding mask (B, 1, L, S), as batch_mask() builds it.
mx::array left_pad_mask(const std::vector<int>& left_padding, int L, int S) {
  const int B = static_cast<int>(left_padding.size());
  mx::array lp(left_padding.data(), {B}, mx::int32);
  mx::array kpos = mx::arange(0, S, 1, mx::int32);
  mx::array valid = mx::less_equal(mx::reshape(lp, {B, 1, 1, 1}), mx::reshape(kpos, {1, 1, 1, S}));
  valid = mx::broadcast_to(valid, {B, 1, L, S});
  const float ninf = -std::numeric_limits<float>::infinity();
  return mx::where(valid, mx::array(0.0f, mx::float16), mx::array(ninf, mx::float16));
}

}  // namespace

TEST_CASE("quantized_sdpa matches fast SDPA over dequantized K/V") {
  for (int bits : {8, 4}) {
    // No GQA: causal prefill and unmasked decode.
    compare(1, 2, 2, 8, 8, bits, "causal", std::nullopt);
    compare(1, 2, 2, 1, 16, bits, "", std::nullopt);
    // GQA (n_repeats=4): the 5-D quantized_matmul broadcast path.
    compare(1, 8, 2, 8, 8, bits, "causal", std::nullopt);
    compare(1, 8, 2, 1, 24, bits, "", std::nullopt);
    // Bottom-right-aligned causal over a longer cached history.
    compare(1, 8, 2, 4, 12, bits, "causal", std::nullopt);
  }
}

TEST_CASE("quantized_sdpa applies the batched additive mask under GQA") {
  // B=2 with n_repeats=4 specifically exercises the (B,1,1,L,S) mask reshape:
  // without it, trailing-axis broadcasting would align B against n_repeats.
  for (int bits : {8, 4}) {
    const int B = 2, L = 1, S = 16;
    mx::array mask = left_pad_mask({3, 0}, L, S);
    compare(B, 8, 2, L, S, bits, "", mask);
    compare(B, 4, 4, L, S, bits, "", mask);  // and the no-GQA masked path
  }
}
