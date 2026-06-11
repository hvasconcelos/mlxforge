// Server configuration from CLI flags / environment.
#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace mlxforge {

// Holds all configuration parameters for the HTTP server, settable via CLI flags or env vars.
struct ServerConfig {
  // Directory containing the model weights and configuration.
  std::string model_dir;

  // Host/IP address to bind the HTTP server. Default: "0.0.0.0" (all interfaces).
  std::string host = "0.0.0.0";

  // TCP port to listen on. Default: 8080.
  int port = 8080;

  // Maximum allowed prompt length in tokens. Prompts longer than this will be rejected with HTTP 400.
  int max_ctx = 8192;

  // Maximum number of requests allowed to wait in the queue. If full, HTTP 429 is returned.
  int max_waiting = 256;

  // Memory budget for the KV cache in bytes. 0 = unbounded (all requests admitted).
  std::size_t kv_budget_bytes = 0;

  // KV-cache quantization bits: 0 = dense fp16 (default), 8 or 4 store the
  // cache quantized (engine-wide; group size fixed at 64 for the server).
  int kv_bits = 0;

  // Prefix cache: reuse pooled KV across requests sharing a token prefix.
  bool prefix_cache = false;
  // Prefix-pool block granularity in tokens (power of two; validated by the engine).
  int kv_block = 256;
  // Prefix-pool RAM budget in bytes. 0 = unbounded.
  std::size_t kv_pool_bytes = 1ull << 30;

  // SSD spill tier for the prefix pool (requires prefix_cache): evicted blocks
  // persist under this directory and survive restarts. Empty = no spill.
  std::string kv_spill_dir;
  // Disk budget for spilled blocks in bytes. 0 = unbounded.
  std::size_t kv_spill_bytes = 0;

  // Parses command line arguments (the model via -m/--model, plus optional flags as --flag value or --flag=value),
  // layering configuration sources by precedence (lowest to highest):
  //   struct defaults < config file (-c/--config) < environment variables < CLI flags.
  // The config file is a JSON object (see from_file); CLI flags always override it.
  // Env vars: MLXFORGE_HOST, MLXFORGE_PORT, MLXFORGE_MAX_CTX, MLXFORGE_MAX_WAITING,
  // MLXFORGE_KV_BUDGET, MLXFORGE_KV_BITS, MLXFORGE_PREFIX_CACHE, MLXFORGE_KV_BLOCK,
  // MLXFORGE_KV_POOL, MLXFORGE_KV_SPILL_DIR, MLXFORGE_KV_SPILL_BYTES.
  // Throws std::runtime_error if an unknown or malformed flag is encountered.
  static ServerConfig parse(const std::vector<std::string>& args);

  // Loads and validates a JSON config file into a fully-populated ServerConfig,
  // with struct defaults filling any keys the file omits. Recognized keys
  // (snake_case): "model", "host", "port", "max_ctx", "max_waiting", "kv_budget",
  // "kv_bits", "prefix_cache", "kv_block", "kv_pool", "kv_spill_dir",
  // "kv_spill_bytes".
  // Validates before applying: rejects unknown keys, wrong types, and out-of-range
  // values. Throws std::runtime_error (with the file path / offending key) on any
  // failure to open, parse, or validate.
  static ServerConfig from_file(const std::string& path);
};

}  // namespace mlxforge
