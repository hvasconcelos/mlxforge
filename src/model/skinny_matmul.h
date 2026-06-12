// Multi-row GEMV kernels for the batched-decode regime.
//
// MLX's Metal matmul drops from ~161 GB/s (GEMV, M=1) to ~56 GB/s (tiled GEMM)
// the moment M reaches 2, and every M in [2, 64] pays the same tile cost
// (ml-explore/mlx#3661) — exactly the continuous-batching decode shape, where
// each per-token step is weight-bandwidth-bound. Three custom
// fast::metal_kernel kernels recover GEMV-class bandwidth, dispatched by
// (M, weight size):
//  - M in [2, 4]: one output column per simdgroup, scalar fp32 register
//    accumulators (~161 GB/s — bandwidth-saturated, nothing faster exists);
//  - M in [5, 16], small weights: two columns per simdgroup — barrier-free
//    independent simdgroups, which ride out the latency of the short
//    back-to-back per-layer matmuls better than anything tiled;
//  - M in [5, 32], big weights (>= 64 MB, in practice the vocab head):
//    simdgroup-matrix MMA tiles. The scalar approach dies past M ~ 4 on
//    instruction issue (M loads + 4M FMAs per weight half4), so this variant
//    moves the arithmetic into hardware simdgroup_half8x8
//    multiply-accumulates (8 output columns x full D per simdgroup, fp32
//    accumulator tiles), stages the x chunk in threadgroup memory once per
//    8-simdgroup threadgroup, and streams each weight element from device
//    exactly once grid-wide: ~134 GB/s at M=8 and still ~68 at M=32 on the
//    vocab head, vs the two-column kernel's 106/12 and the GEMM's flat ~55.
// The split is empirical: on *chained dependent* small matmuls (the engine's
// per-layer regime) the scalar kernels and the stock GEMM (M > 16) win, while
// the MMA kernel wins on big single matmuls at every M in [5, 32] — single-op
// microbenchmarks rank these kernels differently than the engine does.
//
// Accumulation is fp32 but in a different order than mx::matmul, so logits can
// differ at fp16-noise scale — the same class as the decode-vs-recompute gap.
// Gated by EngineConfig::skinny_mm (default on) via DecoderModel::set_skinny_mm
// and token-equality tests against the stock-matmul stream.
#pragma once

#include "mlx/array.h"

namespace mx = mlx::core;

namespace mlxforge {

// True when a kernel path applies to the shapes: x is (B, 1, D) or (B, D)
// fp16, w is a dense fp16 (O, D) weight with D a multiple of 128 (half4
// loads / chunked staging); B in [2, 16] always qualifies, B in [17, 32] only
// on big weights with O >= 8 (the MMA path — small weights past 16 are the
// GEMM's). Enablement is the caller's flag (DecoderModel::skinny_mm_); this
// checks shapes only.
bool skinny_matmul_applies(const mx::array& x, const mx::array& w);

// x @ w.T via the multi-row GEMV kernels. Preserves x's leading shape:
// (B, 1, D) -> (B, 1, O), (B, D) -> (B, O).
mx::array skinny_matmul(const mx::array& x, const mx::array& w);

}  // namespace mlxforge
