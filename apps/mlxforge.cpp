// mlxforge — the OpenAI-compatible server binary (MLXFORGE-022/023/024).
//   mlxforge <model_dir> [--host H] [--port P] [--max-ctx N] [--max-waiting N]
// Loads the tokenizer/config, starts the GPU worker, and serves the HTTP API.
// Config knobs also read from env (MLXFORGE_HOST, MLXFORGE_PORT, ...). SIGINT/SIGTERM
// trigger a graceful shutdown that drains in-flight requests.
#include <csignal>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

#include "core/config.h"
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
  mlxforge::ServerConfig sc;
  try {
    sc = mlxforge::ServerConfig::parse(std::vector<std::string>(argv + 1, argv + argc));
  } catch (const std::exception& e) {
    std::fprintf(stderr, "config error: %s\n", e.what());
    return 2;
  }
  if (sc.model_dir.empty()) {
    std::fprintf(stderr, "usage: mlxforge <model_dir> [--host H] [--port P] [--max-ctx N]\n");
    return 2;
  }

  mlxforge::ModelConfig cfg = mlxforge::ModelConfig::from_file(sc.model_dir + "/config.json");
  mlxforge::Tokenizer tok = mlxforge::Tokenizer::from_file(sc.model_dir + "/tokenizer.json");

  mlxforge::Scheduler scheduler;
  scheduler.set_max_waiting(sc.max_waiting);
  mlxforge::Worker worker(
      [dir = sc.model_dir] {
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

  std::printf("mlxforge serving on http://%s:%d (max_ctx=%d max_waiting=%d)\n", sc.host.c_str(),
              sc.port, sc.max_ctx, sc.max_waiting);
  server.listen(sc.host, sc.port);  // blocks until stop()

  std::printf("draining in-flight requests...\n");
  worker.stop();  // drains the active batch before exit
  return 0;
}
