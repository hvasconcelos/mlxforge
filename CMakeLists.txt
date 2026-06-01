cmake_minimum_required(VERSION 3.24)
project(xllm LANGUAGES CXX)

# --- Global build settings ---------------------------------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

if(NOT APPLE)
  message(FATAL_ERROR "xllm targets Apple Silicon (MLX Metal backend) only.")
endif()

include(cmake/Dependencies.cmake)

# --- Engine library ----------------------------------------------------------
# Per-module sources are added here as each story lands. Stays header/stub-light
# until XLLM-003+ introduce core/, model/, cache/, ... modules.
add_library(xllm_core INTERFACE)
target_include_directories(xllm_core INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(xllm_core INTERFACE mlx nlohmann_json::nlohmann_json)
target_compile_features(xllm_core INTERFACE cxx_std_17)

# --- CLI binary (smoke test for XLLM-001; real generation loop in XLLM-015) ---
add_executable(xllm-cli apps/xllm_cli.cpp)
target_link_libraries(xllm-cli PRIVATE xllm_core)

# --- Tests -------------------------------------------------------------------
enable_testing()
add_subdirectory(tests)
