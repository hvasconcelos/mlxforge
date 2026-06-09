// Minimal base64 decoder for image content in the chat APIs (OpenAI `image_url`
// data: URIs and Anthropic `image` base64 sources). Header-only so both endpoint
// translators can share it without a new TU.
#pragma once

#include <stdexcept>
#include <string>

namespace mlxforge {

// Decode standard base64 to raw bytes. Whitespace is skipped and missing padding
// is tolerated; an out-of-alphabet character throws std::runtime_error.
inline std::string base64_decode(const std::string& in) {
  auto sextet = [](char c) -> int {
    if (c >= 'A' && c <= 'Z') return c - 'A';
    if (c >= 'a' && c <= 'z') return c - 'a' + 26;
    if (c >= '0' && c <= '9') return c - '0' + 52;
    if (c == '+') return 62;
    if (c == '/') return 63;
    return -1;
  };
  std::string out;
  out.reserve(in.size() * 3 / 4);
  int buf = 0, bits = 0;
  for (char c : in) {
    if (c == '=') break;
    if (c == '\n' || c == '\r' || c == ' ' || c == '\t') continue;
    const int v = sextet(c);
    if (v < 0) throw std::runtime_error("invalid base64 character");
    buf = (buf << 6) | v;
    bits += 6;
    if (bits >= 8) {
      bits -= 8;
      out.push_back(static_cast<char>((buf >> bits) & 0xFF));
    }
  }
  return out;
}

}  // namespace mlxforge
