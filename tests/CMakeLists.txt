# Aggregated test binary. Each new story adds its *_test.cpp source here; every
# TEST_CASE is registered with ctest via doctest_discover_tests.
include(doctest)

add_executable(xllm_tests
  doctest_main.cpp
  smoke_test.cpp
)
target_link_libraries(xllm_tests PRIVATE xllm_core doctest::doctest)

doctest_discover_tests(xllm_tests)
