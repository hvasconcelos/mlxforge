// The multi-row GEMV decode kernels must agree with mx::matmul at fp16-noise
// tolerance across the whole M range and all three kernel variants
// (one-column for M <= 4, two-column for 5..16 on small weights including an
// odd O exercising its tail guard, and the simdgroup-matrix MMA path for
// 5..32 on a big weight — M not a multiple of 8 exercises its zero-fill row
// guard, D 128 / 1024 cover both CHUNK template paths), and the shape gate
// must reject everything outside the batched-decode shape.
// Pure GPU test — no model weights needed.
#include <doctest/doctest.h>

#include "model/skinny_matmul.h"

#include "mlx/ops.h"
#include "mlx/random.h"
#include "mlx/transforms.h"

namespace {

float max_abs_diff(const mx::array& a, const mx::array& b) {
  mx::array d = mx::max(mx::abs(mx::subtract(mx::astype(a, mx::float32),
                                             mx::astype(b, mx::float32))));
  mx::eval(d);
  return d.item<float>();
}

}  // namespace

TEST_CASE("skinny_matmul matches mx::matmul across M, D, and both variants") {
  for (int d : {128, 1024}) {
    for (int o : {17, 1536}) {  // odd O exercises the two-column tail guard
      mx::array w = mx::astype(mx::random::normal({o, d}), mx::float16);
      for (int m : {2, 3, 4, 5, 8, 12, 16}) {
        CAPTURE(m);
        CAPTURE(d);
        CAPTURE(o);
        mx::array x = mx::astype(
            mx::multiply(mx::random::normal({m, d}), mx::array(0.05f)), mx::float16);
        REQUIRE(mlxforge::skinny_matmul_applies(x, w));
        mx::array ref = mx::matmul(x, mx::transpose(w));
        mx::array got = mlxforge::skinny_matmul(x, w);
        CHECK(got.shape() == ref.shape());
        // fp32 accumulation vs the GEMM's accumulation order: fp16-noise scale.
        CHECK(max_abs_diff(got, ref) < 5e-3f);

        // The 3-D decode shape (B, 1, D) round-trips its leading shape.
        mx::array x3 = mx::reshape(x, {m, 1, d});
        REQUIRE(mlxforge::skinny_matmul_applies(x3, w));
        mx::array got3 = mlxforge::skinny_matmul(x3, w);
        CHECK(got3.shape() == mx::Shape{m, 1, o});
        CHECK(max_abs_diff(mx::reshape(got3, {m, o}), ref) < 5e-3f);
      }
    }
  }
}

TEST_CASE("skinny_matmul MMA path matches mx::matmul on a big weight, M 5..32") {
  // 32768 x 1024 crosses the big-weight threshold (>= 32M elements), routing
  // M in [5, 32] through the simdgroup-matrix kernel. 32775 columns make the
  // last tile partial, exercising the clamped overlap-recompute path.
  for (int o : {32775, 32768}) {
    mx::array w = mx::astype(mx::random::normal({o, 1024}), mx::float16);
    for (int m : {5, 8, 16, 23, 32}) {
      CAPTURE(m);
      CAPTURE(o);
      mx::array x = mx::astype(
          mx::multiply(mx::random::normal({m, 1024}), mx::array(0.05f)), mx::float16);
      REQUIRE(mlxforge::skinny_matmul_applies(x, w));
      mx::array ref = mx::matmul(x, mx::transpose(w));
      mx::array got = mlxforge::skinny_matmul(x, w);
      CHECK(got.shape() == ref.shape());
      CHECK(max_abs_diff(got, ref) < 5e-3f);
    }
  }
}

TEST_CASE("skinny_matmul_applies rejects everything outside the decode shape") {
  mx::array w = mx::astype(mx::random::normal({64, 1024}), mx::float16);
  auto x = [&](mx::Shape s, mx::Dtype t = mx::float16) {
    return mx::astype(mx::random::normal(std::move(s)), t);
  };
  CHECK_FALSE(mlxforge::skinny_matmul_applies(x({1, 1024}), w));     // M=1: GEMV is faster
  CHECK_FALSE(mlxforge::skinny_matmul_applies(x({17, 1024}), w));    // small w: GEMM past 16
  CHECK(mlxforge::skinny_matmul_applies(x({16, 1024}), w));
  CHECK_FALSE(mlxforge::skinny_matmul_applies(x({4, 2, 1024}), w));  // prefill (L > 1)
  CHECK(mlxforge::skinny_matmul_applies(x({4, 1, 1024}), w));
  CHECK_FALSE(mlxforge::skinny_matmul_applies(x({4, 1024}, mx::float32), w));  // dtype
  mx::array w_odd = mx::astype(mx::random::normal({64, 1000}), mx::float16);
  CHECK_FALSE(mlxforge::skinny_matmul_applies(x({4, 1000}), w_odd));  // D % 128 != 0
  // Big weights extend the range to 32 via the MMA path — but no further.
  mx::array w_big = mx::astype(mx::random::normal({32768, 1024}), mx::float16);
  CHECK(mlxforge::skinny_matmul_applies(x({17, 1024}), w_big));
  CHECK(mlxforge::skinny_matmul_applies(x({32, 1024}), w_big));
  CHECK_FALSE(mlxforge::skinny_matmul_applies(x({33, 1024}), w_big));
}
