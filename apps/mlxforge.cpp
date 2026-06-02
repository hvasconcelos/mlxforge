// mlxforge — the OpenAI-compatible server binary.
//   mlxforge <model> [--host H] [--port P] [--max-ctx N] [--max-waiting N]
// <model> is a local model directory or a HuggingFace repo id (downloaded on
// first use). Loads the tokenizer/config, starts the GPU worker, serves the HTTP API.
// Config knobs also read from env (MLXFORGE_HOST, MLXFORGE_PORT, ...). SIGINT/SIGTERM
// trigger a graceful shutdown that drains in-flight requests.
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

namespace {
mlxforge::HttpServer* g_server = nullptr;
void on_signal(int) {
  if (g_server) g_server->stop();  // unblocks listen(); main then drains
}
}  // namespace

int main(int argc, char** argv) {
  mlxforge::log::init();

  mlxforge::ServerConfig sc;
  try {
    sc = mlxforge::ServerConfig::parse(std::vector<std::string>(argv + 1, argv + argc));
  } catch (const std::exception& e) {
    mlxforge::log::error("config error: {}", e.what());
    return 2;
  }
  if (sc.model_dir.empty()) {
    std::fprintf(stderr, "usage: mlxforge <model> [--host H] [--port P] [--max-ctx N]\n");
    return 2;
  }

  // Resolve the spec (local dir or HF repo id) to a concrete local dir once, up
  // front, so the network/cache lookup never happens on the worker thread.
  std::string dir;
  try {
    dir = mlxforge::resolve_model_dir(sc.model_dir);
  } catch (const std::exception& e) {
    mlxforge::log::error("model error: {}", e.what());
    return 2;
  }

  mlxforge::ModelConfig cfg = mlxforge::ModelConfig::from_file(dir + "/config.json");
  mlxforge::Tokenizer tok = mlxforge::Tokenizer::from_file(
      dir + "/tokenizer.json", cfg.bos_token_id,
      mlxforge::chat_format_from_model_type(cfg.model_type));

  mlxforge::Scheduler scheduler;
  scheduler.set_max_waiting(sc.max_waiting);
  mlxforge::Worker worker(
      [dir] {
        return std::make_unique<mlxforge::LlamaModel>(
            mlxforge::ModelConfig::from_file(dir + "/config.json"), mlxforge::load_weights(dir));
      },
      &scheduler);
  worker.start();

  mlxforge::HttpServer server(&scheduler, &tok, cfg, "mlxforge", [&worker] { return worker.ready(); },
                          sc.max_ctx);
  g_server = &server;
  std::signal(SIGINT, on_signal);
  std::signal(SIGTERM, on_signal);

  mlxforge::log::info("mlxforge serving on http://{}:{} (max_ctx={} max_waiting={})", sc.host,
                      sc.port, sc.max_ctx, sc.max_waiting);
  server.listen(sc.host, sc.port);  // blocks until stop()

  mlxforge::log::info("draining in-flight requests...");
  worker.stop();  // drains the active batch before exit
  return 0;
}
