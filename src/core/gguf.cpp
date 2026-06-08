#include "core/gguf.h"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <optional>
#include <stdexcept>
#include <unordered_map>

#include "core/logging.h"
#include "mlx/ops.h"
#include "mlx/transforms.h"

namespace mx = mlx::core;

namespace mlxforge {

namespace {

bool ends_with(const std::string& s, const std::string& suffix) {
  return s.size() >= suffix.size() &&
         s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

mx::Shape to_shape(const std::vector<int>& v) { return mx::Shape(v.begin(), v.end()); }

// Total element count of a tensor shape (kept int64 so large weights don't overflow).
int64_t num_elements(const std::vector<int>& shape) {
  int64_t n = 1;
  for (int d : shape) n *= d;
  return n;
}

// ----- Own GGUF tensor reader ---------------------------------------------
// We do not use mx::load_gguf at all: MLX v0.31.2's bundled gguflib unpacks Q4_1
// from the wrong block offset (it assumes Q4_0's 2-byte header, but Q4_1's is 4
// bytes: scale + min), mis-dequantizes the Q4_K/Q5_K K-quants, and fails outright
// on some K-quants — all silent or fatal. So we parse the container ourselves and
// read every tensor here: legacy Q4_0/Q4_1/Q8_0 stay quantized (MLX affine layout,
// group_size 32), Q4_K/Q5_K/Q6_K dequantize to dense fp16, F32/F16 read directly.

constexpr uint32_t kGgmlF32 = 0;
constexpr uint32_t kGgmlF16 = 1;
constexpr uint32_t kGgmlQ4_0 = 2;
constexpr uint32_t kGgmlQ4_1 = 3;
constexpr uint32_t kGgmlQ8_0 = 8;
constexpr uint32_t kGgmlQ4_K = 12;
constexpr uint32_t kGgmlQ5_K = 13;
constexpr uint32_t kGgmlQ6_K = 14;
constexpr uint32_t kGgufMagic = 0x46554747;  // "GGUF" little-endian

bool is_legacy_quant(uint32_t t) { return t == kGgmlQ4_0 || t == kGgmlQ4_1 || t == kGgmlQ8_0; }
bool is_k_quant(uint32_t t) { return t == kGgmlQ4_K || t == kGgmlQ5_K || t == kGgmlQ6_K; }

// IEEE half -> float (Apple Silicon is little-endian; scales/mins are stored f16).
float half_to_float(uint16_t h) {
  const uint32_t sign = static_cast<uint32_t>(h & 0x8000) << 16;
  uint32_t exp = (h >> 10) & 0x1F;
  uint32_t mant = h & 0x3FF;
  uint32_t f;
  if (exp == 0) {
    if (mant == 0) {
      f = sign;
    } else {
      exp = 127 - 15 + 1;
      while ((mant & 0x400) == 0) {
        mant <<= 1;
        --exp;
      }
      mant &= 0x3FF;
      f = sign | (exp << 23) | (mant << 13);
    }
  } else if (exp == 0x1F) {
    f = sign | 0x7F800000u | (mant << 13);
  } else {
    f = sign | ((exp - 15 + 127) << 23) | (mant << 13);
  }
  float out;
  std::memcpy(&out, &f, 4);
  return out;
}

template <typename T>
T read_scalar(std::istream& f) {
  T v;
  f.read(reinterpret_cast<char*>(&v), sizeof(T));
  return v;
}

std::string read_gguf_string(std::istream& f) {
  const uint64_t n = read_scalar<uint64_t>(f);
  std::string s(n, '\0');
  if (n) f.read(&s[0], static_cast<std::streamsize>(n));
  return s;
}

void skip_metadata_value(std::istream& f, uint32_t type);

// Skip `cnt` array elements of value type `et`. Fixed-size scalar elements are
// skipped in one jump (-1 marks the variable-width types: string/array); those
// need per-element handling.
void skip_array_elements(std::istream& f, uint32_t et, uint64_t cnt) {
  static const int sz[] = {1, 1, 2, 2, 4, 4, 4, 1, -1, -1, 8, 8, 8};
  if (et <= 12 && sz[et] > 0) {
    f.seekg(static_cast<std::streamoff>(cnt) * sz[et], std::ios::cur);
  } else {
    for (uint64_t i = 0; i < cnt; ++i) skip_metadata_value(f, et);
  }
}

// Advance the stream past one metadata value of the given GGUF value type, so
// we can reach the tensor-info section without interpreting the values (MLX
// already parsed them for us).
void skip_metadata_value(std::istream& f, uint32_t type) {
  switch (type) {
    case 0: case 1: case 7: f.seekg(1, std::ios::cur); break;            // u8,i8,bool
    case 2: case 3: f.seekg(2, std::ios::cur); break;                    // u16,i16
    case 4: case 5: case 6: f.seekg(4, std::ios::cur); break;            // u32,i32,f32
    case 10: case 11: case 12: f.seekg(8, std::ios::cur); break;         // u64,i64,f64
    case 8: { const uint64_t n = read_scalar<uint64_t>(f); f.seekg(static_cast<std::streamoff>(n), std::ios::cur); break; }
    case 9: {  // array: elem_type, count, elements
      const uint32_t et = read_scalar<uint32_t>(f);
      const uint64_t cnt = read_scalar<uint64_t>(f);
      skip_array_elements(f, et, cnt);
      break;
    }
    default: throw std::runtime_error("gguf: unknown metadata value type " + std::to_string(type));
  }
}

struct GgufTensorInfo {
  std::string name;
  uint32_t ggml_type;
  std::vector<uint64_t> dims;  // ggml order (reverse of MLX)
  uint64_t offset;             // relative to the tensor-data section start
};

struct GgufDirectory {
  std::vector<GgufTensorInfo> tensors;
  uint64_t data_start = 0;  // absolute file offset where tensor data begins
};

// Parse the GGUF header/tensor-info section to recover each tensor's type,
// dims and data offset (MLX does not expose these). `alignment` comes from the
// already-parsed metadata (general.alignment, default 32).
GgufDirectory parse_gguf_directory(const std::string& path, uint64_t alignment) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("gguf: cannot open '" + path + "'");
  if (read_scalar<uint32_t>(f) != kGgufMagic) throw std::runtime_error("gguf: bad magic in '" + path + "'");
  read_scalar<uint32_t>(f);  // version
  const uint64_t n_tensors = read_scalar<uint64_t>(f);
  const uint64_t n_kv = read_scalar<uint64_t>(f);
  for (uint64_t i = 0; i < n_kv; ++i) {
    read_gguf_string(f);  // key
    skip_metadata_value(f, read_scalar<uint32_t>(f));
  }
  GgufDirectory dir;
  dir.tensors.reserve(n_tensors);
  for (uint64_t i = 0; i < n_tensors; ++i) {
    GgufTensorInfo t;
    t.name = read_gguf_string(f);
    const uint32_t nd = read_scalar<uint32_t>(f);
    for (uint32_t d = 0; d < nd; ++d) t.dims.push_back(read_scalar<uint64_t>(f));
    t.ggml_type = read_scalar<uint32_t>(f);
    t.offset = read_scalar<uint64_t>(f);
    dir.tensors.push_back(std::move(t));
  }
  const uint64_t pos = static_cast<uint64_t>(f.tellg());
  dir.data_start = ((pos + alignment - 1) / alignment) * alignment;
  return dir;
}

// (weight, scales, biases) for one legacy-quant tensor, in MLX's affine layout
// (group_size 32). The nibble repacking mirrors MLX's own (correct) Q4_0 path;
// Q4_1 differs only by a 4-byte header and a stored min as the bias.
struct QuantArrays {
  mx::array weight;
  mx::array scales;
  mx::array biases;
};

QuantArrays extract_legacy_quant(std::ifstream& f, uint64_t data_pos,
                                 const std::vector<int>& mlx_shape, uint32_t type) {
  const int in = mlx_shape.back();
  int64_t n = 1;
  for (int d : mlx_shape) n *= d;
  const int64_t nblocks = n / 32;
  const int bits = (type == kGgmlQ8_0) ? 8 : 4;

  std::vector<uint8_t> wbytes;
  wbytes.reserve(static_cast<size_t>(n) * bits / 8);
  std::vector<float> scales(nblocks), biases(nblocks);

  f.seekg(static_cast<std::streamoff>(data_pos));
  for (int64_t b = 0; b < nblocks; ++b) {
    const float scale = half_to_float(read_scalar<uint16_t>(f));
    scales[b] = scale;
    if (type == kGgmlQ4_1) {
      biases[b] = half_to_float(read_scalar<uint16_t>(f));  // stored min
    } else if (type == kGgmlQ4_0) {
      biases[b] = -8.0f * scale;
    } else {  // Q8_0
      biases[b] = -128.0f * scale;
    }

    if (bits == 4) {
      uint8_t rb[16];
      f.read(reinterpret_cast<char*>(rb), 16);
      // low nibbles -> quants 0..15, high nibbles -> quants 16..31, packed two
      // per byte in that order (matches MLX's unpack_32_4 output).
      for (int k = 0; k < 8; ++k) wbytes.push_back((rb[2 * k] & 0x0F) | ((rb[2 * k + 1] & 0x0F) << 4));
      for (int k = 0; k < 8; ++k) wbytes.push_back((rb[2 * k] >> 4) | ((rb[2 * k + 1] >> 4) << 4));
    } else {
      uint8_t rb[32];
      f.read(reinterpret_cast<char*>(rb), 32);
      for (int j = 0; j < 32; ++j) wbytes.push_back(rb[j] ^ 0x80);  // int8 -> uint8
    }
  }

  std::vector<int> wshape = mlx_shape;
  wshape.back() = in * bits / 32;  // packed uint32 columns
  std::vector<int> sshape = mlx_shape;
  sshape.back() = in / 32;  // groups

  mx::array weight(reinterpret_cast<const uint32_t*>(wbytes.data()), to_shape(wshape), mx::uint32);
  mx::array sc = mx::astype(mx::array(scales.data(), to_shape(sshape), mx::float32), mx::float16);
  mx::array bs = mx::astype(mx::array(biases.data(), to_shape(sshape), mx::float32), mx::float16);
  mx::eval(weight, sc, bs);
  return {weight, sc, bs};
}

uint16_t le16(const uint8_t* p) {
  uint16_t v;
  std::memcpy(&v, p, 2);
  return v;
}

// Q4_K/Q5_K pack 8 sub-block (32-weight) scales+mins into 12 bytes via 6-bit
// fields; this reproduces ggml's get_scale_min_k4 unpacking.
void get_scale_min_k4(int j, const uint8_t* q, uint8_t& d, uint8_t& m) {
  if (j < 4) {
    d = q[j] & 63;
    m = q[j + 4] & 63;
  } else {
    d = (q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4);
    m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
  }
}

// Dequantize a K-quant tensor (Q4_K/Q5_K/Q6_K) to dense fp32, mirroring ggml's
// dequantize_row_q*_K. K-quants are 256-weight super-blocks; MLX v0.31.2's own
// K-quant dequant is wrong for Q4_K/Q5_K, so we do all three ourselves. Returns
// `n` row-major values (the tensor's quantized dim is always a multiple of 256).
std::vector<float> dequantize_k_quant(std::ifstream& f, uint64_t data_pos, int64_t n,
                                      uint32_t type) {
  const int64_t nsb = n / 256;
  const int bpsb = type == kGgmlQ4_K ? 144 : type == kGgmlQ5_K ? 176 : 210;
  std::vector<uint8_t> raw(static_cast<size_t>(nsb) * bpsb);
  f.seekg(static_cast<std::streamoff>(data_pos));
  f.read(reinterpret_cast<char*>(raw.data()), static_cast<std::streamsize>(raw.size()));

  std::vector<float> y(n);
  float* yp = y.data();
  for (int64_t s = 0; s < nsb; ++s) {
    const uint8_t* blk = raw.data() + s * bpsb;
    if (type == kGgmlQ4_K) {
      const float d = half_to_float(le16(blk));
      const float dmin = half_to_float(le16(blk + 2));
      const uint8_t* scales = blk + 4;  // 12 bytes
      const uint8_t* q = blk + 16;      // 128 bytes
      for (int j = 0, is = 0; j < 256; j += 64, is += 2, q += 32) {
        uint8_t sc, m;
        get_scale_min_k4(is, scales, sc, m);
        const float d1 = d * sc, m1 = dmin * m;
        get_scale_min_k4(is + 1, scales, sc, m);
        const float d2 = d * sc, m2 = dmin * m;
        for (int l = 0; l < 32; ++l) *yp++ = d1 * (q[l] & 0x0F) - m1;
        for (int l = 0; l < 32; ++l) *yp++ = d2 * (q[l] >> 4) - m2;
      }
    } else if (type == kGgmlQ5_K) {
      const float d = half_to_float(le16(blk));
      const float dmin = half_to_float(le16(blk + 2));
      const uint8_t* scales = blk + 4;  // 12 bytes
      const uint8_t* qh = blk + 16;     // 32 bytes (high bit)
      const uint8_t* ql = blk + 48;     // 128 bytes (low 4 bits)
      uint8_t u1 = 1, u2 = 2;
      for (int j = 0, is = 0; j < 256; j += 64, is += 2, ql += 32, u1 <<= 2, u2 <<= 2) {
        uint8_t sc, m;
        get_scale_min_k4(is, scales, sc, m);
        const float d1 = d * sc, m1 = dmin * m;
        get_scale_min_k4(is + 1, scales, sc, m);
        const float d2 = d * sc, m2 = dmin * m;
        for (int l = 0; l < 32; ++l) *yp++ = d1 * ((ql[l] & 0x0F) + ((qh[l] & u1) ? 16 : 0)) - m1;
        for (int l = 0; l < 32; ++l) *yp++ = d2 * ((ql[l] >> 4) + ((qh[l] & u2) ? 16 : 0)) - m2;
      }
    } else {  // Q6_K
      const uint8_t* ql = blk;             // 128 bytes (low 4 bits)
      const uint8_t* qh = blk + 128;       // 64 bytes (high 2 bits)
      const int8_t* sc = reinterpret_cast<const int8_t*>(blk + 192);  // 16 int8 scales
      const float d = half_to_float(le16(blk + 208));
      for (int n2 = 0; n2 < 256; n2 += 128, ql += 64, qh += 32, sc += 8, yp += 128) {
        for (int l = 0; l < 32; ++l) {
          const int is = l / 16;
          const int q1 = static_cast<int8_t>((ql[l] & 0x0F) | (((qh[l] >> 0) & 3) << 4)) - 32;
          const int q2 = static_cast<int8_t>((ql[l + 32] & 0x0F) | (((qh[l] >> 2) & 3) << 4)) - 32;
          const int q3 = static_cast<int8_t>((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
          const int q4 = static_cast<int8_t>((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
          yp[l + 0] = d * sc[is + 0] * q1;
          yp[l + 32] = d * sc[is + 2] * q2;
          yp[l + 64] = d * sc[is + 4] * q3;
          yp[l + 96] = d * sc[is + 6] * q4;
        }
      }
    }
  }
  return y;
}

// Read a dense F32/F16 tensor straight from the file into an fp32/fp16 array.
mx::array read_dense_tensor(std::ifstream& f, uint64_t pos, const std::vector<int>& shape,
                            uint32_t type) {
  const int64_t n = num_elements(shape);
  f.seekg(static_cast<std::streamoff>(pos));
  if (type == kGgmlF32) {
    std::vector<float> buf(n);
    f.read(reinterpret_cast<char*>(buf.data()), static_cast<std::streamsize>(n * 4));
    mx::array a(buf.data(), to_shape(shape), mx::float32);
    mx::eval(a);
    return a;
  }
  // F16: read raw halves, widen to fp32, then build an fp16 array (lossless).
  std::vector<uint16_t> raw(n);
  f.read(reinterpret_cast<char*>(raw.data()), static_cast<std::streamsize>(n * 2));
  std::vector<float> buf(n);
  for (int64_t i = 0; i < n; ++i) buf[i] = half_to_float(raw[i]);
  mx::array a = mx::astype(mx::array(buf.data(), to_shape(shape), mx::float32), mx::float16);
  mx::eval(a);
  return a;
}

// Load every weight tensor from the GGUF file ourselves (we do NOT use
// mx::load_gguf): its bundled gguflib mis-unpacks Q4_1 and mis-dequantizes
// Q4_K/Q5_K, and outright fails to load some K-quants. We handle the dtypes we
// support and reject the rest with a clear error rather than risk silent
// garbage. Legacy Q4_0/Q4_1/Q8_0 stay quantized (group_size 32); the K-quants
// Q4_K/Q5_K/Q6_K are dequantized to dense fp16; F32/F16 are read directly.
std::unordered_map<std::string, mx::array> load_gguf_tensors(const std::string& path,
                                                             uint64_t alignment) {
  const GgufDirectory dir = parse_gguf_directory(path, std::max<uint64_t>(1, alignment));
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("gguf: cannot open '" + path + "'");

  std::unordered_map<std::string, mx::array> out;
  for (const auto& t : dir.tensors) {
    std::vector<int> shape;
    for (auto it = t.dims.rbegin(); it != t.dims.rend(); ++it) shape.push_back(static_cast<int>(*it));
    const uint64_t pos = dir.data_start + t.offset;

    if (t.ggml_type == kGgmlF32 || t.ggml_type == kGgmlF16) {
      out.emplace(t.name, read_dense_tensor(f, pos, shape, t.ggml_type));
    } else if (is_legacy_quant(t.ggml_type)) {
      QuantArrays qa = extract_legacy_quant(f, pos, shape, t.ggml_type);
      const std::string base = t.name.substr(0, t.name.size() - std::strlen(".weight"));
      out.emplace(t.name, qa.weight);
      out.emplace(base + ".scales", qa.scales);
      out.emplace(base + ".biases", qa.biases);
    } else if (is_k_quant(t.ggml_type)) {
      std::vector<float> vals = dequantize_k_quant(f, pos, num_elements(shape), t.ggml_type);
      out.emplace(t.name, mx::astype(mx::array(vals.data(), to_shape(shape), mx::float32),
                                     mx::float16));
    } else {
      throw std::runtime_error("gguf: unsupported tensor quant type " +
                               std::to_string(t.ggml_type) + " for '" + t.name +
                               "' (supported: F32/F16, Q4_0/Q4_1/Q8_0, Q4_K/Q5_K/Q6_K)");
    }
  }
  return out;
}

// ----- Metadata: parsed from the GGUF KV section ourselves ----------------
// Parsing the metadata directly (rather than via mx::load_gguf) lets the server
// read config + tokenizer on the main thread WITHOUT loading any weight arrays
// (MLX arrays are thread-bound; the worker must create the weights itself).

struct GgufMetadata {
  std::unordered_map<std::string, double> nums;  // numeric scalars by key
  std::unordered_map<std::string, std::string> strs;
  std::vector<std::string> tokens, merges;  // tokenizer.ggml.tokens / .merges
  std::vector<int> token_types;             // tokenizer.ggml.token_type
};

bool is_numeric_type(uint32_t t) { return t <= 7 || (t >= 10 && t <= 12); }

double read_numeric(std::istream& f, uint32_t type) {
  switch (type) {
    case 0: return read_scalar<uint8_t>(f);
    case 1: return read_scalar<int8_t>(f);
    case 2: return read_scalar<uint16_t>(f);
    case 3: return read_scalar<int16_t>(f);
    case 4: return read_scalar<uint32_t>(f);
    case 5: return read_scalar<int32_t>(f);
    case 6: return read_scalar<float>(f);
    case 7: return read_scalar<uint8_t>(f);  // bool
    case 10: return static_cast<double>(read_scalar<uint64_t>(f));
    case 11: return static_cast<double>(read_scalar<int64_t>(f));
    case 12: return read_scalar<double>(f);
    default: throw std::runtime_error("gguf: non-numeric metadata type " + std::to_string(type));
  }
}

GgufMetadata parse_gguf_metadata(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("gguf: cannot open '" + path + "'");
  if (read_scalar<uint32_t>(f) != kGgufMagic) throw std::runtime_error("gguf: bad magic in '" + path + "'");
  read_scalar<uint32_t>(f);  // version
  read_scalar<uint64_t>(f);  // tensor count (ignored here)
  const uint64_t n_kv = read_scalar<uint64_t>(f);

  GgufMetadata m;
  for (uint64_t i = 0; i < n_kv; ++i) {
    const std::string key = read_gguf_string(f);
    const uint32_t vt = read_scalar<uint32_t>(f);
    if (vt == 8) {
      m.strs[key] = read_gguf_string(f);
    } else if (is_numeric_type(vt)) {
      m.nums[key] = read_numeric(f, vt);
    } else if (vt == 9) {  // array
      const uint32_t et = read_scalar<uint32_t>(f);
      const uint64_t cnt = read_scalar<uint64_t>(f);
      if (key == "tokenizer.ggml.tokens" && et == 8) {
        m.tokens.resize(cnt);
        for (uint64_t j = 0; j < cnt; ++j) m.tokens[j] = read_gguf_string(f);
      } else if (key == "tokenizer.ggml.merges" && et == 8) {
        m.merges.resize(cnt);
        for (uint64_t j = 0; j < cnt; ++j) m.merges[j] = read_gguf_string(f);
      } else if (key == "tokenizer.ggml.token_type" && is_numeric_type(et)) {
        m.token_types.resize(cnt);
        for (uint64_t j = 0; j < cnt; ++j) m.token_types[j] = static_cast<int>(read_numeric(f, et));
      } else {  // an array we don't need (scores, languages, ...) — skip its elements
        skip_array_elements(f, et, cnt);
      }
    } else {
      throw std::runtime_error("gguf: unknown metadata value type " + std::to_string(vt));
    }
  }
  return m;
}

int gm_int(const GgufMetadata& m, const std::string& k) {
  auto it = m.nums.find(k);
  if (it == m.nums.end()) throw std::runtime_error("gguf: missing metadata '" + k + "'");
  return static_cast<int>(it->second);
}
int gm_int_or(const GgufMetadata& m, const std::string& k, int def) {
  auto it = m.nums.find(k);
  return it == m.nums.end() ? def : static_cast<int>(it->second);
}
float gm_float(const GgufMetadata& m, const std::string& k) {
  auto it = m.nums.find(k);
  if (it == m.nums.end()) throw std::runtime_error("gguf: missing metadata '" + k + "'");
  return static_cast<float>(it->second);
}
float gm_float_or(const GgufMetadata& m, const std::string& k, float def) {
  auto it = m.nums.find(k);
  return it == m.nums.end() ? def : static_cast<float>(it->second);
}
std::string gm_str(const GgufMetadata& m, const std::string& k, const std::string& def = "") {
  auto it = m.strs.find(k);
  return it == m.strs.end() ? def : it->second;
}

// ----- Tensor name remap (ggml -> canonical HF) ---------------------------

// Map a "blk.{i}." per-layer suffix to its canonical HF module path. Returns
// "" for an unrecognized suffix.
std::string remap_block_module(const std::string& rest) {
  static const std::unordered_map<std::string, std::string> kTable = {
      {"attn_q", "self_attn.q_proj"},        {"attn_k", "self_attn.k_proj"},
      {"attn_v", "self_attn.v_proj"},        {"attn_output", "self_attn.o_proj"},
      {"attn_norm", "input_layernorm"},      {"ffn_gate", "mlp.gate_proj"},
      {"ffn_up", "mlp.up_proj"},             {"ffn_down", "mlp.down_proj"},
      {"ffn_norm", "post_attention_layernorm"},
      {"attn_q_norm", "self_attn.q_norm"},   {"attn_k_norm", "self_attn.k_norm"},
      // Qwen3-MoE: the router and the per-layer stacked experts. ggml stores the
      // expert tensors as one 3-D tensor whose reversed dims land as
      // (num_experts, out, in) — exactly the stacked switch_mlp form the forward
      // pass (gather_mm / gather_qmm) consumes, so the dense, K-quant (dequantized
      // to fp16) and legacy-quant layouts all map through here unchanged.
      {"ffn_gate_inp", "mlp.gate"},          {"ffn_gate_exps", "mlp.switch_mlp.gate_proj"},
      {"ffn_up_exps", "mlp.switch_mlp.up_proj"},
      {"ffn_down_exps", "mlp.switch_mlp.down_proj"},
  };
  auto it = kTable.find(rest);
  return it == kTable.end() ? std::string{} : it->second;
}

// Translate a ggml tensor name to the canonical HF key, preserving the
// ".weight"/".scales"/".biases" suffix. Returns nullopt for tensors we drop
// (rope_freqs — lifted into the config — and anything unrecognized).
std::optional<std::string> remap_gguf_key(const std::string& name) {
  std::string suffix, base = name;
  for (const char* s : {".weight", ".scales", ".biases"}) {
    if (ends_with(name, s)) {
      suffix = s;
      base = name.substr(0, name.size() - std::strlen(s));
      break;
    }
  }

  if (base == "token_embd") return "model.embed_tokens" + suffix;
  if (base == "output_norm") return "model.norm" + suffix;
  if (base == "output") return "lm_head" + suffix;
  if (base == "rope_freqs") return std::nullopt;  // baked rope scaling, handled separately

  if (base.rfind("blk.", 0) == 0) {
    const size_t dot = base.find('.', 4);
    if (dot != std::string::npos) {
      const std::string idx = base.substr(4, dot - 4);
      const std::string mapped = remap_block_module(base.substr(dot + 1));
      if (!mapped.empty()) return "model.layers." + idx + "." + mapped + suffix;
    }
  }
  return std::nullopt;
}

// Tag every quantized weight (a base with a ".scales" sibling) in `w.quant`.
// GGUF emits group_size 32; the bit-width follows from the packed-weight vs
// scales column counts: weight cols = in*bits/32, scales cols = in/32, so
// bits = weight_cols / scales_cols (4 for Q4_0/Q4_1, 8 for Q8_0).
void index_gguf_quant(Weights& w) {
  for (const auto& [key, scales] : w.tensors) {
    if (!ends_with(key, ".scales")) continue;
    const std::string base = key.substr(0, key.size() - std::strlen(".scales"));
    const mx::array& weight = w.tensors.at(base + ".weight");
    const int wcols = weight.shape().empty() ? 0 : weight.shape().back();
    const int scols = scales.shape().empty() ? 0 : scales.shape().back();
    const int bits = scols > 0 ? wcols / scols : 0;
    if (bits < 2 || bits > 8 || bits == 7) {
      log::warn("gguf: unexpected derived bits={} for '{}' (w_cols={} s_cols={})", bits, base,
                wcols, scols);
    }
    w.quant.emplace(base, QuantParams{/*group_size=*/32, bits});
  }
}

// llama.cpp permutes the q/k projection rows on HF->GGUF conversion so RoPE can
// be applied in its interleaved layout. We apply HF-style (NeoX) RoPE, so we
// must invert that permutation. It reorders output rows only, so it applies
// cleanly to the packed quantized weight and its per-row scales/biases alike.
// Inverse map: HF row (h*hd + b*hd/2 + a) <- GGUF row (h*hd + a*2 + b).
mx::array unpermute_qk_rows(const mx::array& a, int n_head, int head_dim) {
  const int out = n_head * head_dim;
  const int half = head_dim / 2;
  std::vector<int> idx(out);
  for (int hf = 0; hf < out; ++hf) {
    const int h = hf / head_dim, rem = hf % head_dim;
    const int b = rem / half, aa = rem % half;
    idx[hf] = h * head_dim + aa * 2 + b;
  }
  mx::array index(idx.data(), {out}, mx::int32);
  mx::array out_arr = mx::take(a, index, /*axis=*/0);
  mx::eval(out_arr);  // materialize once at load, not per forward pass
  return out_arr;
}

void unpermute_qk(Weights& w, const ModelConfig& c) {
  const std::pair<std::string, int> projs[] = {{"q_proj", c.n_heads},
                                               {"k_proj", c.n_kv_heads}};
  for (int i = 0; i < c.n_layers; ++i) {
    const std::string p = "model.layers." + std::to_string(i) + ".self_attn.";
    for (const auto& [proj, heads] : projs) {
      for (const char* suf : {".weight", ".scales", ".biases"}) {
        auto it = w.tensors.find(p + proj + suf);
        if (it != w.tensors.end()) it->second = unpermute_qk_rows(it->second, heads, c.head_dim);
      }
    }
  }
}

ModelConfig config_from_gguf(const GgufMetadata& m) {
  const std::string arch = gm_str(m, "general.architecture");
  // Qwen3.5 / Qwen3-Next (hybrid gated-DeltaNet) has no settled, validated GGUF
  // tensor layout; reject it explicitly rather than risk silent garbage from an
  // unverified linear-attention mapping.
  if (arch == "qwen3next") {
    throw std::runtime_error(
        "gguf: Qwen3.5/qwen3next GGUF is not yet supported (its gated-DeltaNet "
        "linear-attention layers have no validated GGUF tensor mapping)");
  }
  if (arch != "llama" && arch != "qwen2" && arch != "qwen3" && arch != "qwen3moe") {
    throw std::runtime_error("gguf: unsupported architecture '" + arch +
                             "' (expected llama/qwen2/qwen3/qwen3moe)");
  }
  // Metadata keys are namespaced by architecture (e.g. "qwen3.block_count").
  const std::string p = arch + ".";
  ModelConfig c;
  // Match the safetensors model_type so the chat format and any downstream
  // dispatch behave identically; the metadata prefix `p` stays the ggml arch.
  c.model_type = (arch == "qwen3moe") ? "qwen3_moe" : arch;
  c.n_layers = gm_int(m, p + "block_count");
  c.hidden = gm_int(m, p + "embedding_length");
  c.n_heads = gm_int(m, p + "attention.head_count");
  c.n_kv_heads = gm_int_or(m, p + "attention.head_count_kv", c.n_heads);
  // MoE (Qwen3-MoE): every layer routes `num_experts_per_tok` of `num_experts`
  // experts, each of width `moe_intermediate_size`. The dense feed_forward_length
  // is absent for an all-MoE model, so fall back to the expert width.
  c.num_experts = gm_int_or(m, p + "expert_count", 0);
  c.num_experts_per_tok = gm_int_or(m, p + "expert_used_count", 0);
  c.moe_intermediate_size = gm_int_or(m, p + "expert_feed_forward_length", 0);
  c.intermediate_size = gm_int_or(m, p + "feed_forward_length", c.moe_intermediate_size);
  // Qwen3 sets an explicit head dim (attention.key_length) that need not equal
  // hidden/n_heads; llama derives it from the rope dimension count.
  c.head_dim = gm_int_or(m, p + "attention.key_length",
                         gm_int_or(m, p + "rope.dimension_count", c.hidden / std::max(1, c.n_heads)));
  c.rms_eps = gm_float(m, p + "attention.layer_norm_rms_epsilon");
  c.rope_theta = gm_float_or(m, p + "rope.freq_base", 10000.0f);
  c.max_position_embeddings = gm_int_or(m, p + "context_length", 0);
  c.vocab = gm_int_or(m, p + "vocab_size", static_cast<int>(m.tokens.size()));
  c.bos_token_id = gm_int_or(m, "tokenizer.ggml.bos_token_id", -1);
  const int eos = gm_int_or(m, "tokenizer.ggml.eos_token_id", -1);
  if (eos >= 0) c.eos_token_ids = {eos};
  return c;
}

// Build the config + tokenizer half of a GgufModel from the parsed metadata
// (no weights). Shared by the full loader and the metadata-only entry point.
GgufModel model_head_from_metadata(const GgufMetadata& m) {
  GgufModel g;
  g.config = config_from_gguf(m);
  g.tokens = m.tokens;
  g.merges = m.merges;
  g.token_types = m.token_types;
  // Some families (Qwen3) define a bos_token_id but do not prepend it on encode;
  // `tokenizer.ggml.add_bos_token` (a GGUF bool, default true) is the source of
  // truth, mirroring the safetensors tokenizer_config.json's add_bos_token.
  g.bos_id = gm_int_or(m, "tokenizer.ggml.add_bos_token", 1) ? g.config.bos_token_id : -1;
  g.eos_id = gm_int_or(m, "tokenizer.ggml.eos_token_id", -1);
  g.pre = gm_str(m, "tokenizer.ggml.pre");
  return g;
}

}  // namespace

bool is_gguf_path(const std::string& spec) {
  if (spec.size() < 5) return false;
  std::string tail = spec.substr(spec.size() - 5);
  std::transform(tail.begin(), tail.end(), tail.begin(),
                 [](unsigned char ch) { return std::tolower(ch); });
  return tail == ".gguf";
}

GgufModel load_gguf_config_and_tokenizer(const std::string& gguf_path) {
  // Parse only the metadata KV section — no weight tensors. Safe to call on any
  // thread (creates no MLX arrays), which lets the server build the config +
  // tokenizer on the main thread while the worker loads the weights.
  GgufModel g = model_head_from_metadata(parse_gguf_metadata(gguf_path));
  log::info("gguf: header '{}' {} layers={} hidden={} vocab={} tokens={} pre='{}'", gguf_path,
            g.config.model_type, g.config.n_layers, g.config.hidden, g.config.vocab,
            g.tokens.size(), g.pre);
  return g;
}

GgufModel load_gguf_model(const std::string& gguf_path) {
  log::info("gguf: loading '{}'", gguf_path);
  const GgufMetadata meta = parse_gguf_metadata(gguf_path);
  GgufModel g = model_head_from_metadata(meta);

  // Load every weight tensor ourselves (MLX's gguflib mis-handles Q4_1/Q4_K/Q5_K
  // and fails outright on some K-quants).
  std::unordered_map<std::string, mx::array> tensors =
      load_gguf_tensors(gguf_path, static_cast<uint64_t>(gm_int_or(meta, "general.alignment", 32)));

  // Lift the baked llama3 rope rescaling out of rope_freqs.weight (the scaling
  // params are absent from the metadata) into the config.
  if (auto it = tensors.find("rope_freqs.weight"); it != tensors.end()) {
    mx::array rf = mx::contiguous(mx::astype(it->second, mx::float32));
    mx::eval(rf);
    const float* p = rf.data<float>();
    g.config.rope_freq_factors = std::vector<float>(p, p + rf.size());
  }

  // Remap tensors to canonical keys; cast floating tensors to fp16, keep packed
  // quantized weights (uint32) as-is.
  for (const auto& [name, arr] : tensors) {
    std::optional<std::string> key = remap_gguf_key(name);
    if (!key) {
      if (name != "rope_freqs.weight") log::debug("gguf: dropping tensor '{}'", name);
      continue;
    }
    mx::array value =
        mx::issubdtype(arr.dtype(), mx::floating) ? mx::astype(arr, mx::float16) : arr;
    g.weights.tensors.emplace(*key, value);
  }
  g.config.tie_word_embeddings = (g.weights.tensors.count("lm_head.weight") == 0);
  unpermute_qk(g.weights, g.config);  // undo llama.cpp's q/k rope permutation
  index_gguf_quant(g.weights);

  log::info("gguf: {} layers={} hidden={} heads={}/{} head_dim={} vocab={} quant_weights={} "
            "tokens={} merges={} pre='{}'",
            g.config.model_type, g.config.n_layers, g.config.hidden, g.config.n_heads,
            g.config.n_kv_heads,
            g.config.head_dim, g.config.vocab, g.weights.quant.size(), g.tokens.size(),
            g.merges.size(), g.pre);
  return g;
}

}  // namespace mlxforge
