// golden-reference compare harness.
//
// Loads the .npy fixtures dumped by reference/dump_ref.py into MLX arrays and
// asserts closeness (fp16 rel ~1e-2) or exact equality for token streams. The
// comparison core (compare_close / compare_tokens) is pure and unit-tested; the
// assert_* wrappers fold the result into doctest with a readable failure message
// that names the first divergent index and its magnitude.
#pragma once

#include <cmath>
#include <cstdint>
#include <sstream>
#include <string>
#include <vector>

#include <doctest/doctest.h>

#include "mlx/mlx.h"

namespace mlxforge::test {

namespace mx = mlx::core;

// Path to a committed reference fixture (reference/fixtures/<name>).
inline std::string ref_path(const std::string& name) {
  return std::string(MLXFORGE_REF_FIXTURES_DIR) + "/" + name;
}

// Load a .npy fixture as an MLX array (dtype preserved).
inline mx::array load_npy_at(const std::string& dir, const std::string& name) {
  return mx::load(dir + "/" + name);
}
inline mx::array load_npy(const std::string& name) { return mx::load(ref_path(name)); }

// Load an int32 .npy token-id fixture as a flat vector<int>.
inline std::vector<int> load_token_ids_at(const std::string& dir, const std::string& name) {
  mx::array a = mx::contiguous(mx::astype(load_npy_at(dir, name), mx::int32));
  mx::eval(a);
  const int32_t* p = a.data<int32_t>();
  return std::vector<int>(p, p + a.size());
}
inline std::vector<int> load_token_ids(const std::string& name) {
  return load_token_ids_at(MLXFORGE_REF_FIXTURES_DIR, name);
}

// Same accessors against the Qwen3 fixture set (reference/fixtures_qwen3).
inline std::string qwen3_ref_path(const std::string& name) {
  return std::string(MLXFORGE_REF_FIXTURES_DIR_QWEN3) + "/" + name;
}
inline mx::array load_qwen3_npy(const std::string& name) { return mx::load(qwen3_ref_path(name)); }
inline std::vector<int> load_qwen3_token_ids(const std::string& name) {
  return load_token_ids_at(MLXFORGE_REF_FIXTURES_DIR_QWEN3, name);
}

struct CompareResult {
  bool ok = true;
  std::string message;
};

// Render a shape as "[d0, d1, ...]" for readable reports.
inline std::string shape_str(const mx::Shape& shape) {
  std::ostringstream os;
  os << "[";
  for (size_t d = 0; d < shape.size(); ++d) os << shape[d] << (d + 1 < shape.size() ? ", " : "");
  os << "]";
  return os.str();
}

// Render a flat index into multi-dimensional coordinates for readable reports.
inline std::string coords_of(int64_t flat, const mx::Shape& shape) {
  std::vector<int64_t> idx(shape.size());
  for (int d = static_cast<int>(shape.size()) - 1; d >= 0; --d) {
    idx[d] = flat % shape[d];
    flat /= shape[d];
  }
  std::ostringstream os;
  os << "(";
  for (size_t d = 0; d < idx.size(); ++d) os << idx[d] << (d + 1 < idx.size() ? ", " : "");
  os << ")";
  return os.str();
}

// allclose-style elementwise check: |a-b| <= atol + rtol*|b|. On the first
// failure, reports the divergent index and magnitudes. Both inputs are compared
// in float32 to avoid fp16 rounding in the comparison itself.
inline CompareResult compare_close(const mx::array& actual, const mx::array& expected,
                                   float rtol = 1e-2f, float atol = 1e-2f) {
  if (actual.shape() != expected.shape()) {
    return {false, "shape mismatch: actual " + shape_str(actual.shape()) + " vs expected " +
                       shape_str(expected.shape())};
  }
  // data<float>() reads the raw row-major buffer, so non-contiguous inputs
  // (e.g. a transposed view) must be materialized contiguous first.
  mx::array a = mx::contiguous(mx::astype(actual, mx::float32));
  mx::array b = mx::contiguous(mx::astype(expected, mx::float32));
  mx::eval(a, b);
  const float* pa = a.data<float>();
  const float* pb = b.data<float>();

  double worst_rel = 0.0;
  int64_t first_bad = -1;
  for (int64_t i = 0; i < static_cast<int64_t>(a.size()); ++i) {
    const double av = static_cast<double>(pa[i]);
    const double bv = static_cast<double>(pb[i]);
    const double diff = std::abs(av - bv);
    const double tol = atol + static_cast<double>(rtol) * std::abs(bv);
    if (diff > tol && first_bad < 0) first_bad = i;
    const double denom = std::max(std::abs(bv), 1e-6);
    worst_rel = std::max(worst_rel, diff / denom);
  }
  if (first_bad < 0) return {true, ""};

  std::ostringstream os;
  os << "first divergence at index " << coords_of(first_bad, a.shape()) << ": actual="
     << pa[first_bad] << " expected=" << pb[first_bad]
     << " absdiff=" << std::abs(pa[first_bad] - pb[first_bad]) << " (rtol=" << rtol
     << ", atol=" << atol << ", worst_rel=" << worst_rel << ")";
  return {false, os.str()};
}

// Exact token-stream comparison. Returns the index of the first mismatch, or -1.
inline int first_token_mismatch(const std::vector<int>& actual, const std::vector<int>& expected) {
  if (actual.size() != expected.size()) {
    return static_cast<int>(std::min(actual.size(), expected.size()));
  }
  for (size_t i = 0; i < actual.size(); ++i) {
    if (actual[i] != expected[i]) return static_cast<int>(i);
  }
  return -1;
}

// doctest wrappers --------------------------------------------------------

inline void assert_close(const mx::array& actual, const mx::array& expected, float rtol = 1e-2f,
                         float atol = 1e-2f) {
  CompareResult r = compare_close(actual, expected, rtol, atol);
  INFO(r.message);
  CHECK(r.ok);
}

inline void assert_tokens_equal(const std::vector<int>& actual, const std::vector<int>& expected) {
  int bad = first_token_mismatch(actual, expected);
  if (bad >= 0) {
    std::ostringstream os;
    os << "token streams differ at index " << bad << ": actual="
       << (bad < static_cast<int>(actual.size()) ? std::to_string(actual[bad]) : "<end>")
       << " expected="
       << (bad < static_cast<int>(expected.size()) ? std::to_string(expected[bad]) : "<end>")
       << " (sizes " << actual.size() << " vs " << expected.size() << ")";
    INFO(os.str());
  }
  CHECK(bad == -1);
}

}  // namespace mlxforge::test
