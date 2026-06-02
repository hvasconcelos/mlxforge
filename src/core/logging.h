#pragma once

// Thin wrapper over spdlog so call sites stay terse and initialization lives in
// one place. Use the `mlxforge::log::{debug,info,warn,error}` free functions with
// fmt-style `{}` placeholders, e.g. `log::info("serving on :{}", port)`.
//
// Verbosity and output are controlled by environment variables (read once by
// init(), matching the MLXFORGE_* convention used elsewhere):
//
//   MLXFORGE_LOG_LEVEL    trace|debug|info|warn|error|critical|off  (default: info)
//   MLXFORGE_LOG_FILE     path; if set, also append logs to this file (console stays on)
//   MLXFORGE_LOG_PATTERN  spdlog pattern string (default: "[%H:%M:%S.%e] [%^%l%$] %v")

#include <spdlog/spdlog.h>

namespace mlxforge {
namespace log {

/// @brief Configure the default logger from the MLXFORGE_LOG_* environment
///        variables. Idempotent and thread-safe: the first call installs the
///        sinks; later calls are no-ops. Logging before init() still works via
///        spdlog's built-in default logger.
void init();

// Re-export spdlog's fmt-style free functions under mlxforge::log so call sites
// read `log::info("msg {}", x)`. These route to the default logger set by init().
using spdlog::debug;
using spdlog::error;
using spdlog::info;
using spdlog::warn;

}  // namespace log
}  // namespace mlxforge
