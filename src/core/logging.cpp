#include "core/logging.h"

#include <cstdlib>
#include <memory>
#include <mutex>
#include <vector>

#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>

namespace mlxforge {
namespace log {

namespace {
// Environment override: returns the env value or `fallback` if unset/empty.
std::string env_or(const char* key, const std::string& fallback) {
  const char* v = std::getenv(key);
  return (v && *v) ? std::string(v) : fallback;
}

std::once_flag g_once;
}  // namespace

void init() {
  std::call_once(g_once, [] {
    // Diagnostics go to stderr so they never pollute stdout (the CLI streams
    // generated text and weight dumps there).
    std::vector<spdlog::sink_ptr> sinks;
    sinks.push_back(std::make_shared<spdlog::sinks::stderr_color_sink_mt>());

    const std::string file = env_or("MLXFORGE_LOG_FILE", "");
    if (!file.empty()) {
      // Append (don't truncate) so logs across runs accumulate.
      sinks.push_back(std::make_shared<spdlog::sinks::basic_file_sink_mt>(file, /*truncate=*/false));
    }

    auto logger = std::make_shared<spdlog::logger>("mlxforge", sinks.begin(), sinks.end());

    // from_str returns `off` for an unrecognized string; fall back to info so a
    // typo doesn't silence the engine.
    const std::string level_str = env_or("MLXFORGE_LOG_LEVEL", "info");
    auto level = spdlog::level::from_str(level_str);
    if (level == spdlog::level::off && level_str != "off") level = spdlog::level::info;
    logger->set_level(level);
    logger->flush_on(spdlog::level::warn);

    logger->set_pattern(env_or("MLXFORGE_LOG_PATTERN", "[%H:%M:%S.%e] [%^%l%$] %v"));

    spdlog::set_default_logger(logger);
  });
}

}  // namespace log
}  // namespace mlxforge
