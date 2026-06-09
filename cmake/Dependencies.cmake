# Third-party dependencies, each pinned to an exact tag (recorded here so the
# build is reproducible). Resolved via FetchContent. Bump deliberately.
#
#   MLX C++ core   v0.31.2  68cf2fddd8de5edd8ab3d926391772b2e2cedad8
#   cpp-httplib    v0.46.1  44215e23e92c473a3553d24ae634aed6eefc7dd0
#   doctest        v2.5.2   6804767ee637789db8a5cb281381cae98dc36906
#   spdlog         v1.15.3  3335c380a08c5e0f5117a66622df6afdb3d74959
#   stb_image      v2.30    31c1ad37456438565541f4919958214b6e762fb4
#   nlohmann/json           (vendored transitively by MLX; reused, not re-fetched)

include(FetchContent)

# --- MLX (Metal GPU backend; the heavy one) ---------------------------------
set(MLX_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(MLX_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(MLX_BUILD_BENCHMARKS OFF CACHE BOOL "" FORCE)
set(MLX_BUILD_PYTHON_BINDINGS OFF CACHE BOOL "" FORCE)
set(MLX_BUILD_METAL ON CACHE BOOL "" FORCE)
FetchContent_Declare(
  mlx
  GIT_REPOSITORY https://github.com/ml-explore/mlx.git
  GIT_TAG v0.31.2
  GIT_SHALLOW TRUE
)

# nlohmann/json is NOT fetched here: MLX already vendors it and exposes the
# nlohmann_json::nlohmann_json target, which we reuse to avoid a duplicate-target
# clash. (Pinned transitively by MLX v0.31.2.)

# --- cpp-httplib (header-only) ----------------------------------------------
FetchContent_Declare(
  httplib
  GIT_REPOSITORY https://github.com/yhirose/cpp-httplib.git
  GIT_TAG v0.46.1
  GIT_SHALLOW TRUE
)

# tokenizers-cpp (the Rust HF tokenizer) has been removed: the engine now uses
# its own from-scratch byte-level BPE (src/tokenizer/bpe.cpp), validated against
# committed mlx-lm golden ids. This also drops the cargo/Rust build requirement.
# A SentencePiece tokenizer is not yet reimplemented.

# --- doctest (unit-test framework, header-only) -----------------------------
set(DOCTEST_WITH_TESTS OFF CACHE BOOL "" FORCE)
set(DOCTEST_NO_INSTALL ON CACHE BOOL "" FORCE)
FetchContent_Declare(
  doctest
  GIT_REPOSITORY https://github.com/doctest/doctest.git
  GIT_TAG v2.5.2
  GIT_SHALLOW TRUE
)

# --- spdlog (fast C++ logging; uses its bundled fmt) ------------------------
set(SPDLOG_BUILD_SHARED OFF CACHE BOOL "" FORCE)
set(SPDLOG_FMT_EXTERNAL OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
  spdlog
  GIT_REPOSITORY https://github.com/gabime/spdlog.git
  GIT_TAG v1.15.3
  GIT_SHALLOW TRUE
)

# --- stb_image (single-header, public-domain image loader) ------------------
# Decodes JPEG/PNG/etc. to RGB for the ViT. No CMakeLists upstream, so it is only
# populated here (header-only); mlxforge_core adds ${stb_SOURCE_DIR} to its
# private include path and one TU defines STB_IMAGE_IMPLEMENTATION.
FetchContent_Declare(
  stb
  GIT_REPOSITORY https://github.com/nothings/stb.git
  GIT_TAG 31c1ad37456438565541f4919958214b6e762fb4  # stb_image v2.30
  GIT_SHALLOW TRUE
)

FetchContent_MakeAvailable(mlx httplib doctest spdlog)
FetchContent_MakeAvailable(stb)

# Provide doctest's CMake helpers (doctest_discover_tests) on the module path.
list(APPEND CMAKE_MODULE_PATH ${doctest_SOURCE_DIR}/scripts/cmake)
