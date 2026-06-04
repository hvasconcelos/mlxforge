# Aggregated test binary. Each module's *_test.cpp source is listed here; every
# TEST_CASE is registered with ctest via doctest_discover_tests.
include(doctest)

add_executable(mlxforge_tests
  doctest_main.cpp
  smoke_test.cpp
  core/config_test.cpp
  core/weights_test.cpp
  core/gguf_test.cpp
  core/model_source_test.cpp
  support/reference_test.cpp
  model/llama_front_test.cpp
  model/llama_block_test.cpp
  model/llama_forward_test.cpp
  model/batched_decode_test.cpp
  cache/kv_cache_test.cpp
  cache/batch_kv_cache_test.cpp
  cache/batch_kv_filter_merge_test.cpp
  cache/kv_budget_test.cpp
  sample/sampler_test.cpp
  runtime/single_stream_test.cpp
  scheduler/scheduler_test.cpp
  scheduler/worker_test.cpp
  runtime/prefill_test.cpp
  scheduler/continuous_batch_test.cpp
  runtime/bucketing_test.cpp
  scheduler/validation_test.cpp
  model/qwen3_test.cpp
  tokenizer/tokenizer_test.cpp
  tokenizer/bpe_test.cpp
  server/openai_test.cpp
  server/sse_test.cpp
  server/hardening_test.cpp
  model/quantized_test.cpp
)
target_include_directories(mlxforge_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(mlxforge_tests PRIVATE mlxforge_core doctest::doctest)

# Resolve the local HF snapshot dir for integration tests (empty -> tests skip).
file(GLOB _mlxforge_model_snapshots
  "$ENV{HOME}/.cache/huggingface/hub/models--mlx-community--Llama-3.2-1B-Instruct-bf16/snapshots/*")
file(GLOB _mlxforge_model_snapshots_4bit
  "$ENV{HOME}/.cache/huggingface/hub/models--mlx-community--Llama-3.2-1B-Instruct-4bit/snapshots/*")
set(MLXFORGE_MODEL_DIR_4BIT "")
if(_mlxforge_model_snapshots_4bit)
  list(GET _mlxforge_model_snapshots_4bit 0 MLXFORGE_MODEL_DIR_4BIT)
endif()
set(MLXFORGE_MODEL_DIR "")
if(_mlxforge_model_snapshots)
  list(GET _mlxforge_model_snapshots 0 MLXFORGE_MODEL_DIR)
endif()

# Qwen3 dense model snapshot for the QK-Norm / ChatML integration tests (empty
# -> those tests self-skip).
file(GLOB _mlxforge_model_snapshots_qwen3
  "$ENV{HOME}/.cache/huggingface/hub/models--mlx-community--Qwen3-0.6B-bf16/snapshots/*")
set(MLXFORGE_MODEL_DIR_QWEN3 "")
if(_mlxforge_model_snapshots_qwen3)
  list(GET _mlxforge_model_snapshots_qwen3 0 MLXFORGE_MODEL_DIR_QWEN3)
endif()

# A local Llama GGUF file for the self-contained-loader integration tests
# (empty -> the GGUF tests self-skip). Any *.gguf under the cached bartowski
# Llama-3.2-1B-Instruct-GGUF repo is used.
file(GLOB _mlxforge_gguf_files
  "$ENV{HOME}/.cache/huggingface/hub/models--bartowski--Llama-3.2-1B-Instruct-GGUF/snapshots/*/*.gguf")
set(MLXFORGE_GGUF_MODEL "")
if(_mlxforge_gguf_files)
  list(GET _mlxforge_gguf_files 0 MLXFORGE_GGUF_MODEL)
endif()

target_compile_definitions(mlxforge_tests PRIVATE
  MLXFORGE_TEST_FIXTURES_DIR="${CMAKE_CURRENT_SOURCE_DIR}/fixtures"
  MLXFORGE_REF_FIXTURES_DIR="${CMAKE_SOURCE_DIR}/reference/fixtures"
  MLXFORGE_REF_FIXTURES_DIR_QWEN3="${CMAKE_SOURCE_DIR}/reference/fixtures_qwen3"
  MLXFORGE_MODEL_DIR="${MLXFORGE_MODEL_DIR}"
  MLXFORGE_MODEL_DIR_4BIT="${MLXFORGE_MODEL_DIR_4BIT}"
  MLXFORGE_MODEL_DIR_QWEN3="${MLXFORGE_MODEL_DIR_QWEN3}"
  MLXFORGE_GGUF_MODEL="${MLXFORGE_GGUF_MODEL}")

doctest_discover_tests(mlxforge_tests)
