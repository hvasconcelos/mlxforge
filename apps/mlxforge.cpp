// mlxforge — the OpenAI-compatible server binary.
//
// Usage: mlxforge -m <model> [-c <conffile>] [--host H] [--port P] [--max-ctx N] [--max-waiting N]
//   -m/--model is either a local model directory or a HuggingFace repo id (downloaded on first use).
//   The model may instead be set via the config file's "model" key (-m/--model overrides it);
//   it is optional on the command line, but if neither source provides one the server exits.
//   Loads the tokenizer/config, starts the GPU worker, and serves the HTTP API.
//   -c/--config loads defaults from a JSON file; env vars then CLI flags override it.
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
#include "core/gguf.h"
#include "core/logging.h"
#include "core/model_source.h"
#include "core/weights.h"
#include "model/model_factory.h"
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

   Local Inference on Apple MLX · OpenAI- & Anthropic-compatible API
)");
  std::fflush(stdout);
}

// Prints the command-line help to stdout (program output, not a log line),
// listing every flag, the model/config precedence, and the env-var overrides.
void print_help() {
  std::puts(
      "mlxforge — OpenAI- & Anthropic-compatible LLM inference server on Apple MLX\n"
      "\n"
      "usage: mlxforge [-m <model>] [-c <conffile>] [options]\n"
      "\n"
      "options:\n"
      "  -m, --model <spec>     model directory or HuggingFace repo id\n"
      "  -c, --config <file>    load configuration from a JSON file\n"
      "      --host <H>         bind address (default 0.0.0.0)\n"
      "      --port <P>         listen port (default 8080)\n"
      "      --max-ctx <N>      max prompt length in tokens (default 8192)\n"
      "      --max-waiting <N>  max queued requests (default 256)\n"
      "      --kv-budget <B>    KV cache budget in bytes, 0 = unbounded (default 0)\n"
      "  -h, --help             show this help and exit\n"
      "\n"
      "The model may be given via -m or the config file's \"model\" key.\n"
      "Config precedence (low to high): defaults < config file < env vars < CLI flags.\n"
      "Env vars: MLXFORGE_HOST, MLXFORGE_PORT, MLXFORGE_MAX_CTX, MLXFORGE_MAX_WAITING, "
      "MLXFORGE_KV_BUDGET.");
  std::fflush(stdout);
}

}  // namespace

int main(int argc, char** argv) {
  // Initialize the logger as early as possible (stderr; picks up env config)
  mlxforge::log::init();

  // Convert argv to a std::vector<std::string> (excluding program name).
  const std::vector<std::string> args(argv + 1, argv + argc);

  // -h/--help short-circuits everything: print the banner + help to stdout and exit 0.
  for (const std::string& a : args) {
    if (a == "-h" || a == "--help") {
      print_banner();
      print_help();
      return 0;
    }
  }

  // Print banner to stdout (decorative, not for logging)
  print_banner();

  // Parse server config from command-line arguments and (optionally) environment.
  mlxforge::ServerConfig sc;
  try {
    sc = mlxforge::ServerConfig::parse(args);
  } catch (const std::exception& e) {
    mlxforge::log::error("config error: {}", e.what());
    return 2;
  }
  // The model is optional on the command line: it may come from -m/--model or
  // the config file's "model" key. If neither supplies one, there's nothing to
  // serve — report it clearly and print usage.
  if (sc.model_dir.empty()) {
    std::fprintf(stderr,
                 "error: no model provided (pass -m <model> or set \"model\" in the config file)\n"
                 "usage: mlxforge -m <model> [-c <conffile>] [--host H] [--port P] [--max-ctx N]\n");
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

  // Load the config + tokenizer on the main thread. For GGUF this parses only
  // the metadata (no weight tensors): MLX arrays are thread-bound, so the worker
  // must create the weights itself.
  const bool is_gguf = mlxforge::is_gguf_path(dir);
  mlxforge::ModelConfig cfg;
  mlxforge::Tokenizer tok;
  if (is_gguf) {
    mlxforge::GgufModel head = mlxforge::load_gguf_config_and_tokenizer(dir);
    cfg = std::move(head.config);
    tok = mlxforge::Tokenizer::from_gguf(head.tokens, head.merges, head.token_types, head.pre,
                                         head.bos_id,
                                         mlxforge::chat_format_from_model_type(cfg.model_type));
  } else {
    cfg = mlxforge::ModelConfig::from_file(dir + "/config.json");
    tok = mlxforge::Tokenizer::from_file(dir + "/tokenizer.json", cfg.bos_token_id,
                                         mlxforge::chat_format_from_model_type(cfg.model_type));
  }

  // Create inference scheduler and set waiting queue bound.
  mlxforge::Scheduler scheduler;
  scheduler.set_max_waiting(sc.max_waiting);

  // Construct and start the GPU worker thread (loads weights + builds the model
  // on the worker thread, where the MLX arrays must live).
  mlxforge::Worker worker(
      [dir, is_gguf] {
        if (is_gguf) {
          mlxforge::GgufModel g = mlxforge::load_gguf_model(dir);
          return mlxforge::create_model(std::move(g.config), std::move(g.weights));
        }
        mlxforge::ModelConfig wcfg = mlxforge::ModelConfig::from_file(dir + "/config.json");
        auto weights = mlxforge::load_weights(dir, wcfg);
        return mlxforge::create_model(std::move(wcfg), std::move(weights));
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
