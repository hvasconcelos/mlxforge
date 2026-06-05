#include "runtime/engine.h"

#include <utility>

#include "core/gguf.h"
#include "core/model_source.h"
#include "core/weights.h"
#include "model/model_factory.h"

namespace mlxforge {

// Parse the config + tokenizer on the calling thread. For GGUF this reads only
// the metadata (no weight tensors): MLX arrays are thread-bound, so the worker
// must create the weights itself (see make_factory).
Engine::Loaded Engine::load_head(const std::string& spec) {
  Loaded out;
  out.dir = resolve_model_dir(spec);
  out.is_gguf = is_gguf_path(out.dir);
  if (out.is_gguf) {
    GgufModel head = load_gguf_config_and_tokenizer(out.dir);
    out.config = std::move(head.config);
    out.tokenizer = Tokenizer::from_gguf(head.tokens, head.merges, head.token_types, head.pre,
                                         head.bos_id,
                                         chat_format_from_model_type(out.config.model_type));
  } else {
    out.config = ModelConfig::from_file(out.dir + "/config.json");
    out.tokenizer = Tokenizer::from_file(out.dir + "/tokenizer.json", out.config.bos_token_id,
                                         chat_format_from_model_type(out.config.model_type));
  }
  return out;
}

// Build the lambda the worker runs on its own thread to load the weights and
// construct the model (where the MLX arrays must live).
Worker::ModelFactory Engine::make_factory(std::string dir, bool is_gguf) {
  return [dir = std::move(dir), is_gguf]() -> std::unique_ptr<DecoderModel> {
    if (is_gguf) {
      GgufModel g = load_gguf_model(dir);
      return create_model(std::move(g.config), std::move(g.weights));
    }
    ModelConfig wcfg = ModelConfig::from_file(dir + "/config.json");
    auto weights = load_weights(dir, wcfg);
    return create_model(std::move(wcfg), std::move(weights));
  };
}

Engine::Engine(EngineConfig cfg) : Engine(cfg, load_head(cfg.model_spec)) {}

Engine::Engine(EngineConfig cfg, Loaded loaded)
    : model_name_(std::move(cfg.model_spec)),
      cfg_(std::move(loaded.config)),
      tok_(std::move(loaded.tokenizer)),
      worker_(make_factory(std::move(loaded.dir), loaded.is_gguf), &scheduler_) {
  scheduler_.set_max_waiting(cfg.max_waiting);
  worker_.start();  // loads the weights on the worker thread, then loops
}

Engine::~Engine() = default;  // worker_ destructs first (drains), then scheduler_

}  // namespace mlxforge
