#include "vision/image_decode.h"

#include <fstream>
#include <stdexcept>
#include <vector>

#define STB_IMAGE_IMPLEMENTATION
#define STBI_NO_STDIO  // we read files ourselves; only the from-memory path is used
#include "stb_image.h"

namespace mlxforge {

mx::array decode_image(const uint8_t* data, std::size_t len) {
  int w = 0, h = 0, channels = 0;
  // Force 3 channels (RGB), dropping alpha / expanding grayscale.
  unsigned char* px =
      stbi_load_from_memory(data, static_cast<int>(len), &w, &h, &channels, /*desired=*/3);
  if (px == nullptr) {
    throw std::runtime_error(std::string("decode_image: ") + stbi_failure_reason());
  }
  // Copy out of stb's buffer into an MLX array, then release stb's allocation.
  std::vector<uint8_t> buf(px, px + static_cast<std::size_t>(h) * w * 3);
  stbi_image_free(px);
  return mx::array(buf.data(), {h, w, 3}, mx::uint8);
}

mx::array decode_image_file(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("decode_image_file: cannot open '" + path + "'");
  std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(f)),
                             std::istreambuf_iterator<char>());
  return decode_image(bytes.data(), bytes.size());
}

}  // namespace mlxforge
