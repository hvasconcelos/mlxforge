#include "model/skinny_matmul.h"

#include <optional>
#include <vector>

#include "mlx/fast.h"
#include "mlx/ops.h"

namespace mlxforge {

namespace {

// One simdgroup per output column o: 32 lanes split D with half4 loads, so the
// weight row is read once (coalesced) for all M activation rows; per-lane fp32
// accumulators reduce via simd_sum. M is a compile-time template arg so the
// accumulators stay in registers. Best for M in [2, 4].
constexpr const char* kSourceOneCol = R"(
    const uint lane = thread_position_in_grid.x;   // 0..31
    const uint o    = thread_position_in_grid.y;   // output column (w row)
    const int  D    = w_shape[1];
    const int  O    = w_shape[0];

    const device half4* wrow = (const device half4*)(w + (size_t)o * D);
    float acc[M];
    for (int m = 0; m < M; ++m) acc[m] = 0.0f;

    for (int k = lane; k < D / 4; k += 32) {
        half4 wv = wrow[k];
        for (int m = 0; m < M; ++m) {
            half4 xv = ((const device half4*)(x + (size_t)m * D))[k];
            acc[m] += (float)wv.x * (float)xv.x + (float)wv.y * (float)xv.y +
                      (float)wv.z * (float)xv.z + (float)wv.w * (float)xv.w;
        }
    }
    for (int m = 0; m < M; ++m) {
        float r = simd_sum(acc[m]);
        if (lane == 0) y[(size_t)m * O + o] = (half)r;
    }
)";

// Two output columns per simdgroup: each activation load feeds two weight
// rows, halving the redundant x traffic that degrades the one-column variant
// past M ~ 8. Best for M in [5, 16] on the small per-layer weights, where its
// barrier-free independent simdgroups ride out the latency of short
// back-to-back ops better than the MMA tile kernel below.
constexpr const char* kSourceTwoCol = R"(
    const uint lane = thread_position_in_grid.x;   // 0..31
    const uint pair = thread_position_in_grid.y;   // output column pair
    const int  D    = w_shape[1];
    const int  O    = w_shape[0];
    const uint o0   = pair * 2;
    const uint o1   = o0 + 1;
    const bool has1 = o1 < (uint)O;

    const device half4* w0 = (const device half4*)(w + (size_t)o0 * D);
    const device half4* w1 = (const device half4*)(w + (size_t)(has1 ? o1 : o0) * D);
    float acc0[M];
    float acc1[M];
    for (int m = 0; m < M; ++m) { acc0[m] = 0.0f; acc1[m] = 0.0f; }

    for (int k = lane; k < D / 4; k += 32) {
        half4 wv0 = w0[k];
        half4 wv1 = w1[k];
        for (int m = 0; m < M; ++m) {
            half4 xv = ((const device half4*)(x + (size_t)m * D))[k];
            acc0[m] += (float)wv0.x * (float)xv.x + (float)wv0.y * (float)xv.y +
                       (float)wv0.z * (float)xv.z + (float)wv0.w * (float)xv.w;
            acc1[m] += (float)wv1.x * (float)xv.x + (float)wv1.y * (float)xv.y +
                       (float)wv1.z * (float)xv.z + (float)wv1.w * (float)xv.w;
        }
    }
    for (int m = 0; m < M; ++m) {
        float r0 = simd_sum(acc0[m]);
        float r1 = simd_sum(acc1[m]);
        if (lane == 0) {
            y[(size_t)m * O + o0] = (half)r0;
            if (has1) y[(size_t)m * O + o1] = (half)r1;
        }
    }
)";

// simdgroup-matrix MMA tile kernel for M in [5, 32] on *large* weights (the
// vocab head): each simdgroup owns an 8-output-column tile across the full D,
// accumulating y^T = w_tile * x_tile^T in hardware simdgroup_float8x8 ops, so
// the per-element FMA/issue cost that throttles the scalar kernels past M ~ 4
// disappears. Memory hierarchy:
//  - weights stream from device exactly once grid-wide (plain, non-transposed
//    simdgroup_loads; the transpose lands on x instead);
//  - the x chunk is staged in threadgroup memory once per threadgroup and the
//    transposed tile loads hit that on-chip copy, not device/L1 — SG=8
//    simdgroups (64 output columns) share each staged chunk;
//  - the staging loop zero-fills rows past M, so no host-side padding of x.
// Tail tiles clamp o0 to O-8 and overlap-recompute (duplicate stores of
// identical values are benign); applies() guarantees O >= 8 and CHUNK | D.
constexpr const char* kSourceMma = R"(
    const uint lane   = thread_position_in_threadgroup.x;  // 0..31
    const uint sg     = thread_position_in_threadgroup.y;  // 0..SG-1
    const uint o_tile = thread_position_in_grid.y;
    const int  D = w_shape[1];
    const int  O = w_shape[0];
    const size_t o0 = (size_t)min((int)(o_tile * 8), O - 8);

    threadgroup half xs[MT * 8 * CHUNK];
    simdgroup_float8x8 acc[MT];
    for (int t = 0; t < MT; ++t)
        acc[t] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_half8x8 xt, wt[UN];
    const uint tid = sg * 32 + lane;
    for (int kc = 0; kc < D; kc += CHUNK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = tid; i < (uint)(MT * 8 * CHUNK / 4); i += 32 * SG) {
            uint m = i / (CHUNK / 4), j = i % (CHUNK / 4);
            ((threadgroup half4*)xs)[i] = (m < (uint)M)
                ? ((const device half4*)(x + (size_t)m * D + kc))[j]
                : half4(0.0h);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int k = 0; k < CHUNK; k += 8 * UN) {
            for (int u = 0; u < UN; ++u)
                simdgroup_load(wt[u], w + o0 * D + kc + k + u * 8, D);
            for (int t = 0; t < MT; ++t) {
                for (int u = 0; u < UN; ++u) {
                    simdgroup_load(xt, (threadgroup half*)xs +
                                   (size_t)(t * 8) * CHUNK + k + u * 8, CHUNK, 0, true);
                    simdgroup_multiply_accumulate(acc[t], wt[u], xt, acc[t]);
                }
            }
        }
    }
    threadgroup float scratch[SG][64];
    for (int t = 0; t < MT; ++t) {
        simdgroup_store(acc[t], scratch[sg], 8);
        simdgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = lane; i < 64; i += 32) {
            int m = t * 8 + (int)(i % 8);
            if (m < M) y[(size_t)m * O + o0 + (i / 8)] = (half)scratch[sg][i];
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }
)";

constexpr int kOneColMaxM = 4;
constexpr int kTwoColMaxM = 16;
constexpr int kMmaMaxM = 32;
constexpr int kSimdgroupsPerTg = 8;  // SG: o-tiles sharing one staged x chunk
constexpr int kUnroll = 4;           // UN: weight tiles in flight per k step

// The MMA kernel only beats the alternatives on big single matmuls (the vocab
// head, where it sustains GEMV-class bandwidth out to M=32). On the small
// per-layer weights the engine runs as short dependent back-to-back ops, and
// there the barrier-free scalar kernels (M <= 16) or the stock GEMM (M > 16)
// win — measured on chained-dependent shapes, not single dispatches.
constexpr int64_t kMmaMinWeightElems = 32 * 1024 * 1024;  // 64 MB of fp16

bool is_big_weight(const mx::array& w) {
  return (int64_t)w.shape()[0] * w.shape()[1] >= kMmaMinWeightElems;
}

}  // namespace

bool skinny_matmul_applies(const mx::array& x, const mx::array& w) {
  if (x.dtype() != mx::float16 || w.dtype() != mx::float16) return false;
  if (w.ndim() != 2 || w.shape()[1] % 128 != 0) return false;
  const int nd = x.ndim();
  if (nd == 3 && x.shape()[1] != 1) return false;  // decode shape only, never prefill
  if (nd != 2 && nd != 3) return false;
  const int m = x.shape()[0];
  if (m < 2 || x.shape()[nd - 1] != w.shape()[1]) return false;
  if (m <= kTwoColMaxM) return true;
  // 17..32 pays off only on big weights (and the MMA kernel needs a full
  // 8-column tile); on small weights the stock GEMM wins — fall back.
  return m <= kMmaMaxM && w.shape()[0] >= 8 && is_big_weight(w);
}

mx::array skinny_matmul(const mx::array& x, const mx::array& w) {
  static const auto one_col = mx::fast::metal_kernel(
      "mlxforge_gemv_multirow", {"x", "w"}, {"y"}, kSourceOneCol);
  static const auto two_col = mx::fast::metal_kernel(
      "mlxforge_gemv_multirow2", {"x", "w"}, {"y"}, kSourceTwoCol);
  static const auto mma = mx::fast::metal_kernel(
      "mlxforge_gemv_mma", {"x", "w"}, {"y"}, kSourceMma);

  const int m = x.shape()[0];
  const int o = w.shape()[0];
  const int d = w.shape()[1];
  mx::array x2 = x.ndim() == 3 ? mx::reshape(x, {m, x.shape()[2]}) : x;
  std::vector<mx::array> out;
  if (m <= kOneColMaxM) {
    out = one_col({x2, w}, {mx::Shape{m, o}}, {mx::float16},
                  /*grid=*/{32, o, 1}, /*threadgroup=*/{32, 1, 1},
                  /*template_args=*/{{"M", m}},
                  /*init_value=*/std::nullopt, /*verbose=*/false, {});
  } else if (o >= 8 && is_big_weight(w)) {
    const int chunk = d % 256 == 0 ? 256 : 128;
    int tiles = (o + 7) / 8;
    tiles = (tiles + kSimdgroupsPerTg - 1) / kSimdgroupsPerTg * kSimdgroupsPerTg;
    out = mma({x2, w}, {mx::Shape{m, o}}, {mx::float16},
              /*grid=*/{32, tiles, 1}, /*threadgroup=*/{32, kSimdgroupsPerTg, 1},
              /*template_args=*/
              {{"M", m}, {"MT", (m + 7) / 8}, {"CHUNK", chunk}, {"UN", kUnroll},
               {"SG", kSimdgroupsPerTg}},
              /*init_value=*/std::nullopt, /*verbose=*/false, {});
  } else {
    out = two_col({x2, w}, {mx::Shape{m, o}}, {mx::float16},
                  /*grid=*/{32, (o + 1) / 2, 1}, /*threadgroup=*/{32, 1, 1},
                  /*template_args=*/{{"M", m}},
                  /*init_value=*/std::nullopt, /*verbose=*/false, {});
  }
  return x.ndim() == 3 ? mx::reshape(out[0], {m, 1, o}) : out[0];
}

}  // namespace mlxforge
