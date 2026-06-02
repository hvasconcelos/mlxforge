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

find_package(Threads REQUIRED)

# --- Engine library ----------------------------------------------------------
# One static library holding every engine module (core/, model/, cache/,
# sample/, scheduler/, runtime/, server/, tokenizer/); the binaries and the test
# suite all link against it.
add_library(mlxforge_core STATIC
  src/core/config.cpp
  src/core/logging.cpp
  src/core/weights.cpp
  src/model/llama.cpp
  src/cache/kv_cache.cpp
  src/cache/batch_kv_cache.cpp
  src/cache/kv_budget.cpp
  src/sample/sampler.cpp
  src/runtime/single_stream.cpp
  src/scheduler/scheduler.cpp
  src/runtime/worker.cpp
  src/runtime/batching.cpp
  src/tokenizer/tokenizer.cpp
  src/server/openai.cpp
  src/server/http_server.cpp
  src/server/config.cpp
)
target_include_directories(mlxforge_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(mlxforge_core PUBLIC mlx nlohmann_json::nlohmann_json Threads::Threads tokenizers_cpp httplib::httplib spdlog::spdlog)
target_compile_features(mlxforge_core PUBLIC cxx_std_17)

# --- CLI binary (weight dump + single-stream generation) --------------------
add_executable(mlxforge-cli apps/mlxforge_cli.cpp)
target_link_libraries(mlxforge-cli PRIVATE mlxforge_core)

add_executable(mlxforge apps/mlxforge.cpp)
target_link_libraries(mlxforge PRIVATE mlxforge_core)

# --- Tests -------------------------------------------------------------------
enable_testing()
add_subdirectory(tests)
