// mlxforge — the OpenAI-compatible server binary.
//
// Usage: mlxforge <model> [--host H] [--port P] [--max-ctx N] [--max-waiting N]
//   <model> is either a local model directory or a HuggingFace repo id (downloaded on first use).
//   Loads the tokenizer/config, starts the GPU worker, and serves the HTTP API.
//   Command line flags can also be set via environment variables (see server/config.h for details).
//   Receives SIGINT/SIGTERM for graceful shutdown.
//
// Main responsibilities of this file:
//   - Parse command-line and environment config for server and model
//   - Resolve/prepare the model directory (with optional HuggingFace fetch)
//   - Load tokenizer and model configs
//   - Instantiate and start the worker and request scheduler
//   - Run the batching HTTP server (OpenAI API-compatible)
//   - Log critical state changes; print decorative banner on startup
//   - Handle signals for graceful shutdown, draining in-flight requests

#include <csignal>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

#include "core/config.h"
#include "core/logging.h"
#include "core/model_source.h"
#include "core/weights.h"
#include "model/llama.h"
#include "runtime/worker.h"
#include "scheduler/scheduler.h"
#include "server/config.h"
#include "server/http_server.h"
#include "tokenizer/tokenizer.h"

// Internal namespace for non-exported globals and helpers.
namespace {

// Pointer to the active server instance (global for use in the signal handler).
mlxforge::HttpServer* g_server = nullptr;

// Signal handler: Called on SIGINT/SIGTERM.
// If the server is running, calls stop() to unblock listen() and begin shutdown.
void on_signal(int) {
  if (g_server)
    g_server->stop();  // interrupts listen(); main() then drains outstanding requests.
}

// Prints a decorative ASCII art startup banner to stdout.
// This is program output, not a log line.
void print_banner() {
  std::puts(R"(
▗▖  ▗▖▗▖   ▗▖  ▗▖▗▄▄▄▖ ▗▄▖ ▗▄▄▖  ▗▄▄▖▗▄▄▄▖
▐▛▚▞▜▌▐▌    ▝▚▞▘ ▐▌   ▐▌ ▐▌▐▌ ▐▌▐▌   ▐▌
▐▌  ▐▌▐▌     ▐▌  ▐▛▀▀▘▐▌ ▐▌▐▛▀▚▖▐▌▝▜▌▐▛▀▀▘
▐▌  ▐▌▐▙▄▄▖▗▞▘▝▚▖▐▌   ▝▚▄▞▘▐▌ ▐▌▝▚▄▞▘▐▙▄▄▖ ⚒️

   LLaMA inference on Apple MLX · OpenAI-compatible API
)");
  std::fflush(stdout);
}

}  // namespace

int main(int argc, char** argv) {
  // Initialize the logger as early as possible (stderr; picks up env config)
  mlxforge::log::init();

  // Print banner to stdout (decorative, not for logging)
  print_banner();

  // Parse server config from command-line arguments and (optionally) environment.
  mlxforge::ServerConfig sc;
  try {
    // Convert argv to a std::vector<std::string> (excluding program name)
    sc = mlxforge::ServerConfig::parse(std::vector<std::string>(argv + 1, argv + argc));
  } catch (const std::exception& e) {
    mlxforge::log::error("config error: {}", e.what());
    return 2;
  }
  // If required positional argument is missing, print usage and exit.
  if (sc.model_dir.empty()) {
    std::fprintf(stderr, "usage: mlxforge <model> [--host H] [--port P] [--max-ctx N]\n");
    return 2;
  }

  // Resolve the model directory:
  //   - Accepts a local dir or a HuggingFace repo id.
  //   - If necessary, downloads or finds in cache (never on worker thread).
  std::string dir;
  try {
    dir = mlxforge::resolve_model_dir(sc.model_dir);
  } catch (const std::exception& e) {
    mlxforge::log::error("model error: {}", e.what());
    return 2;
  }

  // Load model and tokenizer configuration from the resolved directory.
  mlxforge::ModelConfig cfg = mlxforge::ModelConfig::from_file(dir + "/config.json");
  mlxforge::Tokenizer tok = mlxforge::Tokenizer::from_file(
      dir + "/tokenizer.json",
      cfg.bos_token_id,
      mlxforge::chat_format_from_model_type(cfg.model_type)
  );

  // Create inference scheduler and set waiting queue bound.
  mlxforge::Scheduler scheduler;
  scheduler.set_max_waiting(sc.max_waiting);

  // Construct and start the GPU worker thread (loads model; performs evals)
  mlxforge::Worker worker(
      [dir] {
        // Loads weights and constructs model on the worker thread.
        return std::make_unique<mlxforge::LlamaModel>(
            mlxforge::ModelConfig::from_file(dir + "/config.json"),
            mlxforge::load_weights(dir));
      },
      &scheduler);
  worker.start();

  // Instantiate HTTP server (OpenAI API-compatible).
  //   - Scheduler handles queuing/batching of inference requests.
  //   - Tokenizer used for prompt/input processing and token streaming.
  //   - cfg names the currently loaded model.
  //   - The model spec the user passed is the served model name: it is echoed in
  //     responses and /v1/models, and requests naming a different model are
  //     rejected (an OpenAI client must target the loaded model).
  //   - Readiness/metrics checks are lambda-captured from worker.
  mlxforge::HttpServer server(
      &scheduler,
      &tok,
      cfg,
      sc.model_dir,
      [&worker] { return worker.ready(); },
      sc.max_ctx,
      [&worker] { return worker.metrics(); }
  );

  // Set global server pointer for signal handler shutdown.
  g_server = &server;

  // Attach signal handlers for graceful shutdown (Ctrl+C, kill).
  std::signal(SIGINT, on_signal);
  std::signal(SIGTERM, on_signal);

  // Info log: server has started, print bind details and config bounds.
  mlxforge::log::info("mlxforge serving on http://{}:{} (max_ctx={} max_waiting={})",
                      sc.host, sc.port, sc.max_ctx, sc.max_waiting);

  // Run the server's request loop (blocks until stop() is called).
  server.listen(sc.host, sc.port);

  // After listen() returns, begin draining worker batch before exit.
  mlxforge::log::info("draining in-flight requests...");
  worker.stop();  // drains active batch and stops worker thread

  return 0;
}
