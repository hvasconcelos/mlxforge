cmake_minimum_required(VERSION 3.24)
project(mlxforge LANGUAGES CXX)

# --- Global build settings ---------------------------------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

if(NOT APPLE)
  message(FATAL_ERROR "mlxforge targets Apple Silicon (MLX Metal backend) only.")
endif()

include(cmake/Dependencies.cmake)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

find_package(Threads REQUIRED)

# --- Build options -----------------------------------------------------------
# The product is libmlxforge (the C ABI). The HTTP server and CLI are QA
# harnesses on the same engine, built by default for development/CI but excluded
# from the released library artifact.
option(MLXFORGE_BUILD_SERVER "Build the OpenAI HTTP server harness" ON)
option(MLXFORGE_BUILD_CLI "Build the CLI harness" ON)
option(MLXFORGE_BUILD_SHARED "Build the libmlxforge shared library (the product)" ON)
option(MLXFORGE_ENABLE_HF_DOWNLOAD "Enable HuggingFace model download via libcurl" ON)
# ASan + UBSan for the engine core and test binary, to back the fuzz/endurance
# suite with crash/use-after-free/double-free/UB detection. LeakSanitizer is NOT
# available on Apple Silicon, so leaks are quantified by the RSS-growth assertions
# in tests/capi/capi_endurance_test.cpp instead. Off by default (it slows builds
# and is not wanted in the released artifact); turn on for a hardening run:
#   cmake -S . -B build-asan -DMLXFORGE_ENABLE_SANITIZERS=ON
option(MLXFORGE_ENABLE_SANITIZERS "Build core + tests with AddressSanitizer + UBSan" OFF)
if(MLXFORGE_ENABLE_SANITIZERS)
  add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer)
  add_link_options(-fsanitize=address,undefined)
endif()

# --- Engine core (the library; lean — no HTTP) -------------------------------
# Static library holding the engine modules (core/, model/, cache/, sample/,
# scheduler/, runtime/, tokenizer/). The C ABI (src/capi) is compiled into the
# shared library and the test binary directly. mlxforge_core does NOT include
# the server (cpp-httplib) and only links libcurl when HF download is enabled,
# so a build with the harnesses off produces a lean dylib with neither.
add_library(mlxforge_core STATIC
  src/core/config.cpp
  src/core/env.cpp
  src/core/logging.cpp
  src/core/weights.cpp
  src/core/gguf.cpp
  src/core/model_source.cpp
  src/model/decoder_model.cpp
  src/model/sdpa.cpp
  src/model/qwen3.cpp
  src/model/qwen3_moe.cpp
  src/model/qwen3_5.cpp
  src/model/vision/vit.cpp
  src/model/qwen3_vl.cpp
  src/vision/preprocess.cpp
  src/vision/image_decode.cpp
  src/model/model_factory.cpp
  src/cache/kv_cache.cpp
  src/cache/batch_kv_cache.cpp
  src/cache/kv_quant.cpp
  src/cache/kv_budget.cpp
  src/sample/sampler.cpp
  src/sample/json_grammar.cpp
  src/runtime/single_stream.cpp
  src/runtime/multimodal_stream.cpp
  src/runtime/embedding.cpp
  src/runtime/engine.cpp
  src/scheduler/scheduler.cpp
  src/runtime/worker.cpp
  src/runtime/batching.cpp
  src/tokenizer/bpe.cpp
  src/tokenizer/spm.cpp
  src/tokenizer/tokenizer.cpp
)
target_include_directories(mlxforge_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
# stb_image is header-only and used only inside src/vision/image_decode.cpp, so it
# stays a PRIVATE include (never part of the public surface).
target_include_directories(mlxforge_core PRIVATE ${stb_SOURCE_DIR})
target_link_libraries(mlxforge_core PUBLIC mlx nlohmann_json::nlohmann_json Threads::Threads spdlog::spdlog)
target_compile_features(mlxforge_core PUBLIC cxx_std_17)
# Position-independent so it can be folded into the shared library / a Node addon.
set_target_properties(mlxforge_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

if(MLXFORGE_ENABLE_HF_DOWNLOAD)
  find_package(CURL REQUIRED)  # system libcurl (macOS SDK): HuggingFace model downloads
  target_sources(mlxforge_core PRIVATE src/core/hf_download.cpp)
  target_link_libraries(mlxforge_core PRIVATE CURL::libcurl)  # client-only; no curl types in public headers
  target_compile_definitions(mlxforge_core PRIVATE MLXFORGE_ENABLE_HF_DOWNLOAD=1)
endif()

# --- libmlxforge: the product (shared library exposing only the C ABI) -------
if(MLXFORGE_BUILD_SHARED)
  add_library(mlxforge_shared SHARED src/capi/mlxforge.cpp)
  target_link_libraries(mlxforge_shared PRIVATE mlxforge_core)
  set_target_properties(mlxforge_shared PROPERTIES
    OUTPUT_NAME mlxforge
    VERSION 0.1.0
    SOVERSION 0)
  target_include_directories(mlxforge_shared PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/capi>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

  # Install the dylib, the single public header, and a pkg-config file.
  install(TARGETS mlxforge_shared LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
  install(FILES src/capi/mlxforge.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
  configure_file(cmake/mlxforge.pc.in
    ${CMAKE_CURRENT_BINARY_DIR}/mlxforge.pc @ONLY)
  install(FILES ${CMAKE_CURRENT_BINARY_DIR}/mlxforge.pc
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/pkgconfig)
endif()

# --- Server harness (cpp-httplib; not part of the product library) -----------
# The OpenAI/Anthropic serialization + HTTP routing live in their own static lib
# so the test suite can exercise them; both the lib and the server binary are
# gated on MLXFORGE_BUILD_SERVER.
if(MLXFORGE_BUILD_SERVER)
  add_library(mlxforge_server STATIC
    src/server/openai.cpp
    src/server/anthropic.cpp
    src/server/http_server.cpp
    src/server/config.cpp
  )
  target_link_libraries(mlxforge_server PUBLIC mlxforge_core httplib::httplib)

  add_executable(mlxforge apps/mlxforge.cpp)
  target_link_libraries(mlxforge PRIVATE mlxforge_server)
endif()

# --- CLI harness (weight dump + single-stream generation) --------------------
if(MLXFORGE_BUILD_CLI)
  add_executable(mlxforge-cli apps/mlxforge_cli.cpp)
  target_link_libraries(mlxforge-cli PRIVATE mlxforge_core)
endif()

# --- Tests -------------------------------------------------------------------
enable_testing()
add_subdirectory(tests)
