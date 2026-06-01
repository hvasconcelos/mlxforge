// xllm — the OpenAI-compatible server binary (XLLM-022).
//   xllm <model_dir> [port]
// Loads the tokenizer/config, starts the GPU worker, and serves the HTTP API.
#include <cstdio>
#include <memory>
#include <string>

#include "core/config.h"
#include "core/weights.h"
#include "model/llama.h"
#include "runtime/worker.h"
#include "scheduler/scheduler.h"
#include "server/http_server.h"
#include "tokenizer/tokenizer.h"

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: xllm <model_dir> [port]\n");
    return 2;
  }
  const std::string dir = argv[1];
  const int port = argc >= 3 ? std::stoi(argv[2]) : 8080;

  xllm::ModelConfig cfg = xllm::ModelConfig::from_file(dir + "/config.json");
  xllm::Tokenizer tok = xllm::Tokenizer::from_file(dir + "/tokenizer.json");

  xllm::Scheduler scheduler;
  xllm::Worker worker(
      [dir] {
        return std::make_unique<xllm::LlamaModel>(
            xllm::ModelConfig::from_file(dir + "/config.json"), xllm::load_weights(dir));
      },
      &scheduler);
  worker.start();

  xllm::HttpServer server(&scheduler, &tok, cfg, "xllm");
  std::printf("xllm serving on http://0.0.0.0:%d\n", port);
  server.listen("0.0.0.0", port);

  worker.stop();
  return 0;
}
