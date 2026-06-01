# Aggregated test binary. Each new story adds its *_test.cpp source here; every
# TEST_CASE is registered with ctest via doctest_discover_tests.
include(doctest)

add_executable(xllm_tests
  doctest_main.cpp
  smoke_test.cpp
  core/config_test.cpp
  core/weights_test.cpp
  support/reference_test.cpp
  model/llama_front_test.cpp
  model/llama_block_test.cpp
)
target_include_directories(xllm_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(xllm_tests PRIVATE xllm_core doctest::doctest)

# Resolve the local HF snapshot dir for integration tests (empty -> tests skip).
file(GLOB _xllm_model_snapshots
  "$ENV{HOME}/.cache/huggingface/hub/models--mlx-community--Llama-3.2-1B-Instruct-bf16/snapshots/*")
set(XLLM_MODEL_DIR "")
if(_xllm_model_snapshots)
  list(GET _xllm_model_snapshots 0 XLLM_MODEL_DIR)
endif()

target_compile_definitions(xllm_tests PRIVATE
  XLLM_TEST_FIXTURES_DIR="${CMAKE_CURRENT_SOURCE_DIR}/fixtures"
  XLLM_REF_FIXTURES_DIR="${CMAKE_SOURCE_DIR}/reference/fixtures"
  XLLM_MODEL_DIR="${XLLM_MODEL_DIR}")

doctest_discover_tests(xllm_tests)
