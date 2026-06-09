// Image decode for the ViT front-door: compressed bytes (JPEG/PNG/…) -> RGB.
//
// Wraps the vendored stb_image loader. Output is an (H, W, 3) uint8 MLX array —
// exactly the input patchify_image() expects. stb is included only in the .cpp,
// so it never leaks into the public headers.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>

#include "mlx/array.h"

namespace mlxforge {

namespace mx = mlx::core;

// Decode an in-memory image to (H, W, 3) uint8 RGB. Throws std::runtime_error
// (with stb's reason) if the bytes are not a decodable image.
mx::array decode_image(const uint8_t* data, std::size_t len);

// Read just an encoded image's pixel dimensions, without decoding it (CPU only,
// no MLX) -> {height, width}. Lets a non-worker thread compute the image-token
// count for prompt templating. Throws if the bytes are not a recognizable image.
std::array<int, 2> image_info(const uint8_t* data, std::size_t len);

// Read and decode an image file to (H, W, 3) uint8 RGB. Throws if the file
// cannot be read or decoded.
mx::array decode_image_file(const std::string& path);

}  // namespace mlxforge
