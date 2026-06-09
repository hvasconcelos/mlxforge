#include "runtime/embedding.h"

#include "mlx/ops.h"
#include "mlx/transforms.h"

namespace mlxforge {

std::vector<float> embed_pooled(const DecoderModel& model, const std::vector<int>& ids,
                                Pooling pooling) {
  if (ids.empty()) return {};

  const int L = static_cast<int>(ids.size());
  mx::array tokens(ids.data(), {1, L}, mx::int32);
  mx::array hidden = model.forward_hidden(tokens);  // (1, L, H)
  const int H = hidden.shape()[2];

  // Pool over the sequence axis -> (1, H).
  mx::array pooled =
      (pooling == Pooling::Last)
          ? mx::reshape(mx::slice(hidden, {0, L - 1, 0}, {1, L, H}), {1, H})
          : mx::mean(hidden, /*axis=*/1, /*keepdims=*/false);

  pooled = mx::astype(pooled, mx::float32);

  // L2-normalize so cosine similarity is a plain dot product.
  mx::array norm = mx::sqrt(mx::sum(mx::multiply(pooled, pooled)));
  pooled = mx::divide(pooled, mx::maximum(norm, mx::array(1e-12f)));

  mx::array out = mx::contiguous(mx::reshape(pooled, {H}));
  mx::eval(out);
  return std::vector<float>(out.data<float>(), out.data<float>() + out.size());
}

}  // namespace mlxforge
