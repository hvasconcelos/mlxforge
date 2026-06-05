#include "runtime/engine.h"

#include <utility>

#include "core/gguf.h"
#include "core/model_source.h"
#include "core/weights.h"
#include "model/model_factory.h"

namespace mlxforge {

// Loads the model directory, config, and tokenizer metadata, but not weights.
// For GGUF, this loads only the config/tokenizer fields from the GGUF header (not tensors).
// For non-GGUF (MLX-style), loads config.json and tokenizer.json from disk.
// This must be called on the main thread (MLX arrays are thread-locked to creating thread).
Engine::Loaded Engine::load_head(const std::string& spec) {
  Loaded out;

  // Resolve the physical directory for this model spec (could be local path or HF repo).
  out.dir = resolve_model_dir(spec);

  // Detect format: GGUF or not (gguf for Llama3, Qwen3, etc., older/other for JSON).
  out.is_gguf = is_gguf_path(out.dir);

  if (out.is_gguf) {
    // Parse GGUF header: config, tokenizer vocab/merges/etc. No heavy tensor weights read here.
    GgufModel head = load_gguf_config_and_tokenizer(out.dir);
    out.config = std::move(head.config);
    out.tokenizer = Tokenizer::from_gguf(
        head.tokens,
        head.merges,
        head.token_types,
        head.pre,
        head.bos_id,
        chat_format_from_model_type(out.config.model_type)  // Derive chat format if any
    );
  } else {
    // Legacy/MLX model: parse config and tokenizer JSON files.
    out.config = ModelConfig::from_file(out.dir + "/config.json");
    out.tokenizer = Tokenizer::from_file(
        out.dir + "/tokenizer.json",
        out.config.bos_token_id,
        chat_format_from_model_type(out.config.model_type)
    );
  }
  return out;
}

// Returns a callable (r-value lambda) that, when invoked on the worker thread,
// loads weight tensors (heavy), constructs the model, and returns it as unique_ptr.
// MLX arrays must be created on the thread that will use them (the worker's).
//   - For GGUF: loads the weights from the GGUF file, then builds model.
//   - For non-GGUF: loads model config and weights, then builds model.
Worker::ModelFactory Engine::make_factory(std::string dir, bool is_gguf) {
  return [dir = std::move(dir), is_gguf]() -> std::unique_ptr<DecoderModel> {
    if (is_gguf) {
      // Parse config and load all weights from GGUF in one shot.
      GgufModel g = load_gguf_model(dir);
      return create_model(std::move(g.config), std::move(g.weights));
    }
    // Load config and weight tensors for legacy (non-GGUF) model format.
    ModelConfig wcfg = ModelConfig::from_file(dir + "/config.json");
    auto weights = load_weights(dir, wcfg);
    return create_model(std::move(wcfg), std::move(weights));
  };
}

// Engine constructor: minimal form — loads config/tokenizer head from the given model spec.
Engine::Engine(EngineConfig cfg)
    : Engine(cfg, load_head(cfg.model_spec)) {}

// Engine constructor: explicit Loaded head (allows passing in preloaded config/tokenizer).
// Sets up model name, config, tokenizer, and spawns worker thread for weights/model load.
Engine::Engine(EngineConfig cfg, Loaded loaded)
    : model_name_(std::move(cfg.model_spec)),
      cfg_(std::move(loaded.config)),
      tok_(std::move(loaded.tokenizer)),
      worker_(make_factory(std::move(loaded.dir), loaded.is_gguf), &scheduler_) {
  // Configure the max waiting requests for the batch scheduler.
  scheduler_.set_max_waiting(cfg.max_waiting);

  // Start the background worker thread: loads heavy weights and enters run loop.
  worker_.start();
}

// Properly destruct the Engine: ensures worker thread is drained before scheduler dies.
Engine::~Engine() = default;  // worker_ is destroyed first (drains); then scheduler_

}  // namespace mlxforge
