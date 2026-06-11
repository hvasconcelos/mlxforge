#include "runtime/engine.h"

#include <fstream>
#include <memory>
#include <stdexcept>
#include <utility>

#include <nlohmann/json.hpp>

#include "core/gguf.h"
#include "core/model_source.h"
#include "core/weights.h"
#include "model/model_factory.h"
#include "scheduler/request.h"

namespace mlxforge {

namespace {

// Best-effort: sniff a sentence-transformers pooling sidecar in the model dir to
// pick embedding defaults. Qwen3-Embedding ships `1_Pooling/config.json` with
// `pooling_mode_lasttoken: true`, which also implies a trailing EOS (its
// last_token_pool reads the sentence-final position). Anything we can't read
// keeps the plain-LLM default (mean pooling, no EOS).
void detect_embedding_defaults(const std::string& dir, int& pooling, bool& add_eos) {
  std::ifstream f(dir + "/1_Pooling/config.json");
  if (!f) return;
  try {
    nlohmann::json j;
    f >> j;
    if (j.value("pooling_mode_lasttoken", false)) {
      pooling = 1;  // Pooling::Last
      add_eos = true;
    }
  } catch (...) {
    // malformed sidecar -> keep defaults
  }
}

// Validate the engine's KV-quantization request against the loaded model and
// return the Worker's KVQuantConfig. Unsupported setups are hard errors —
// never a silent fp16 fallback (the failure mode here is silent numerical
// garbage, so the caller must know exactly what storage they got).
KVQuantConfig validate_kv_quant(const EngineConfig& ec, const ModelConfig& mc) {
  if (ec.kv_bits == 0) return {};
  if (ec.kv_bits != 4 && ec.kv_bits != 8)
    throw std::runtime_error("kv_bits must be 0 (off), 4, or 8; got " +
                             std::to_string(ec.kv_bits));
  if (ec.kv_group_size != 32 && ec.kv_group_size != 64 && ec.kv_group_size != 128)
    throw std::runtime_error("kv_group_size must be 32, 64, or 128; got " +
                             std::to_string(ec.kv_group_size));
  if (mc.head_dim % ec.kv_group_size != 0)
    throw std::runtime_error("kv_group_size " + std::to_string(ec.kv_group_size) +
                             " does not divide head_dim " + std::to_string(mc.head_dim));
  // Golden-gated for the standard attention paths only so far: the vision
  // (mlx-vlm-gated) and Qwen3.5 hybrid streams have no quantized reference yet.
  if (mc.has_vision_tower())
    throw std::runtime_error("KV-cache quantization is not supported for vision-language models");
  if (mc.full_attention_interval > 0)
    throw std::runtime_error("KV-cache quantization is not supported for hybrid (Qwen3.5) models");
  return {ec.kv_bits, ec.kv_group_size};
}

// Validate the prefix-cache request against the loaded model and return the
// Worker's PrefixCacheConfig. Same philosophy as validate_kv_quant: unsupported
// setups are hard errors, never a silent off.
// `model_name` is passed separately: by the time the Worker member initializes,
// EngineConfig::model_spec has already been moved into Engine::model_name_.
PrefixCacheConfig validate_prefix_cache(const EngineConfig& ec, const ModelConfig& mc,
                                        const std::string& model_name) {
  if (!ec.prefix_cache) {
    if (!ec.kv_spill_dir.empty())
      throw std::runtime_error("kv_spill_dir requires prefix_cache to be enabled");
    return {};
  }
  const int bs = ec.kv_block_size;
  if (bs < 16 || bs > 4096 || (bs & (bs - 1)) != 0)
    throw std::runtime_error("kv_block_size must be a power of two in [16, 4096]; got " +
                             std::to_string(bs));
  // Token-id hashing cannot identify image content / 3D positions, and the
  // hybrid linear-attention state cannot be reconstructed at block boundaries.
  if (mc.has_vision_tower())
    throw std::runtime_error("the prefix cache is not supported for vision-language models");
  if (mc.full_attention_interval > 0)
    throw std::runtime_error("the prefix cache is not supported for hybrid (Qwen3.5) models");
  PrefixCacheConfig pc;
  pc.enabled = true;
  pc.block_size = bs;
  pc.pool_bytes = ec.kv_pool_bytes;
  pc.spill_dir = ec.kv_spill_dir;
  pc.spill_bytes = ec.kv_spill_bytes;
  // Salt every block key with the model identity + storage config so pooled
  // (and, later, persisted) blocks can never cross models or settings.
  const std::string fp = model_name + "|" + mc.model_type + "|" + std::to_string(mc.n_layers) +
                         "|" + std::to_string(ec.kv_bits) + "|" + std::to_string(ec.kv_group_size) +
                         "|" + std::to_string(bs);
  pc.salt = fnv1a(fp.data(), fp.size());
  return pc;
}

}  // namespace

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

  // Sniff embedding defaults from the on-disk sentence-transformers sidecar (no
  // such files inside a GGUF, so this is a no-op there).
  detect_embedding_defaults(out.dir, out.embed_pooling_default, out.embed_add_eos_default);
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
      embed_pooling_default_(loaded.embed_pooling_default),
      embed_add_eos_default_(loaded.embed_add_eos_default),
      // Pass the tokenizer so the worker can build per-token byte strings for
      // constrained decoding. tok_ is initialized above and outlives worker_.
      // cfg_ is initialized above, so the KV-quant validation sees the model.
      worker_(make_factory(std::move(loaded.dir), loaded.is_gguf), &scheduler_, &tok_,
              validate_kv_quant(cfg, cfg_), validate_prefix_cache(cfg, cfg_, model_name_)) {
  // Configure the max waiting requests for the batch scheduler.
  scheduler_.set_max_waiting(cfg.max_waiting);

  // Start the background worker thread: loads heavy weights and enters run loop.
  worker_.start();
}

std::vector<float> Engine::embed(const std::string& text, const EmbedOptions& opts) {
  // Resolve tri-state options against the model's detected defaults.
  const int pooling = opts.pooling >= 0 ? opts.pooling : embed_pooling_default_;
  const bool add_eos = opts.add_eos >= 0 ? (opts.add_eos != 0) : embed_add_eos_default_;

  // Qwen3-Embedding wraps retrieval queries; documents are embedded raw. The
  // format matches Qwen's get_detailed_instruct exactly (note: no space after
  // "Query:"), so the tokens match the reference.
  const std::string input =
      opts.instruction.empty() ? text : "Instruct: " + opts.instruction + "\nQuery:" + text;

  auto req = std::make_shared<Request>();
  req->prompt_ids = tok_.encode(input);
  // Last-token pooling needs the sentence-final EOS so it reads the right state.
  if (add_eos && !cfg_.eos_token_ids.empty())
    req->prompt_ids.push_back(cfg_.eos_token_ids.front());
  req->embedding = true;
  req->pooling = pooling;
  req->embedding_normalize = opts.normalize;
  if (!scheduler_.submit(req)) return {};  // queue full

  // Block until the worker runs the forward pass and closes the queue (an
  // embedding request produces no tokens).
  int tok = 0;
  while (req->tokens.pop(tok)) {
  }
  return req->embedding_result;
}

// Properly destruct the Engine: ensures worker thread is drained before scheduler dies.
Engine::~Engine() = default;  // worker_ is destroyed first (drains); then scheduler_

}  // namespace mlxforge
