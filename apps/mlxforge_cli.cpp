// mlxforge-cli — command-line entry point.
//
// Usage patterns:
//   mlxforge-cli
//     - Performs a smoke test: adds two arrays, evaluates the computation, and prints the sum.
//   mlxforge-cli dump-weights <dir>
//     - Loads a model's weights from the supplied directory, prints key/shape/dtype for each tensor,
//       asserts that all tensors are fp16, and reports the peak resident memory used.
//   mlxforge-cli generate <model> <prompt> [max_tokens]
//     - Runs greedy single-stream generation: pre-fills the prompt (as raw text using the chat template
//       or as a pre-tokenized .npy of ids), then streams the detokenized text to stdout until EOS or
//       max_tokens.
//   mlxforge-cli bench <model> [max_tokens] [runs]
//     - Repeatable throughput benchmark over a fixed prompt: one discarded warmup run, then `runs`
//       timed runs (defaults: max_tokens=128, runs=3) reporting time-to-first-token and decode tok/s.
//   mlxforge-cli embed <model> <text> [--last|--mean] [--eos] [--instruct "..."] [--no-normalize]
//     - Embeds text and prints the (by default unit-normalized) vector. With no flags the model
//       self-selects its convention (a Qwen3-Embedding checkpoint uses last-token pooling + a
//       trailing EOS). The embedding smoke/golden-reference harness for the library.
//
// <dir>/<model> is either a local model directory or a HuggingFace repo id (e.g. mlx-community/Llama-3.2-1B-Instruct-4bit),
// which will be downloaded on first use.

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <string>
#include <thread>
#include <vector>

#include "mlx/mlx.h"

#include "core/config.h"
#include "core/gguf.h"
#include "core/logging.h"
#include "core/model_source.h"
#include "core/weights.h"
#include "model/model_factory.h"
#include "runtime/engine.h"
#include "runtime/single_stream.h"
#include "tokenizer/tokenizer.h"

// Alias for convenience
namespace mx = mlx::core;

// Utility function: Checks if string s ends with the given suffix.
namespace {
bool ends_with(const std::string& s, const std::string& suffix) {
  return s.size() >= suffix.size()
    && s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}
}  // namespace

namespace {

// A loaded model + its tokenizer + config, from either a GGUF file or a
// safetensors model directory. The model is heap-owned so it can be moved out
// of the loader without requiring DecoderModel to be movable (and to hold the
// concrete subclass create_model() picks behind the base pointer).
struct LoadedModel {
  std::unique_ptr<mlxforge::DecoderModel> model;
  mlxforge::Tokenizer tok;
  mlxforge::ModelConfig cfg;
};

// Resolve a model spec and load it for single-stream inference, dispatching on
// whether it resolves to a GGUF file or a safetensors directory.
LoadedModel load_for_inference(const std::string& spec) {
  const std::string resolved = mlxforge::resolve_model_dir(spec);
  LoadedModel lm;
  if (mlxforge::is_gguf_path(resolved)) {
    mlxforge::GgufModel g = mlxforge::load_gguf_model(resolved);
    lm.cfg = g.config;
    lm.tok = mlxforge::Tokenizer::from_gguf(g.tokens, g.merges, g.token_types, g.pre, g.bos_id,
                                            mlxforge::chat_format_from_model_type(lm.cfg.model_type));
    lm.model = mlxforge::create_model(std::move(g.config), std::move(g.weights));
  } else {
    lm.cfg = mlxforge::ModelConfig::from_file(resolved + "/config.json");
    lm.model = mlxforge::create_model(lm.cfg, mlxforge::load_weights(resolved, lm.cfg));
    lm.tok = mlxforge::Tokenizer::from_file(resolved + "/tokenizer.json", lm.cfg.bos_token_id,
                                            mlxforge::chat_format_from_model_type(lm.cfg.model_type));
  }
  return lm;
}

// build smoke test: verifies MLX graph building/eval and prints result
int run_smoke() {
  // Check if Metal GPU backend is available and log; return error otherwise
  if (!mx::metal::is_available()) {
    mlxforge::log::error("Metal GPU is not available on this machine");
    return 1;
  }
  mlxforge::log::info("Metal available: yes");

  // Create two arrays with the same shape
  mx::array a({1.0f, 2.0f, 3.0f, 4.0f});
  mx::array b({10.0f, 20.0f, 30.0f, 40.0f});
  // Add the arrays (creates a lazy operation graph)
  mx::array c = mx::add(a, b);

  // Evaluate the computation (forces the lazy graph to run)
  mx::eval(c);

  // Print the resulting array
  const float* data = c.data<float>();
  std::printf("a + b = [");
  for (int i = 0; i < c.size(); ++i) {
    std::printf("%g%s", data[i], i + 1 < c.size() ? ", " : "");
  }
  std::printf("]\n");
  return 0;
}

// Loads all weights from a model directory and checks their dtypes.
int run_dump_weights(const std::string& spec) {
  // Resolve the model spec, possibly downloading it
  const std::string resolved = mlxforge::resolve_model_dir(spec);
  // Reset the recorded peak device memory for accurate measurement
  mx::reset_peak_memory();
  // Load all weight tensors (cfg supplies the quant params used to tag quantized
  // tensors). GGUF carries its config in-file; safetensors reads config.json.
  mlxforge::Weights w;
  if (mlxforge::is_gguf_path(resolved)) {
    w = std::move(mlxforge::load_gguf_model(resolved).weights);
  } else {
    mlxforge::ModelConfig cfg = mlxforge::ModelConfig::from_file(resolved + "/config.json");
    w = mlxforge::load_weights(resolved, cfg);
  }

  // Materialize (evaluate) all tensors so memory usage is valid,
  // and enforce that any fp16 cast is done before measurement
  std::vector<mx::array> all;
  all.reserve(w.tensors.size());
  for (auto& [_, a] : w.tensors)
    all.push_back(a);
  mx::eval(all);

  // Print a summary of all tensors: key, shape, dtype, etc.
  std::printf("%s", w.summary().c_str());

  // Check if all tensors are fp16, and count the ones that are not
  size_t non_fp16 = 0;
  for (const auto& [_, a] : w.tensors) {
    if (a.dtype() != mx::float16)
      ++non_fp16;
  }

  // Calculate resident peak memory in GiB
  const double gib = static_cast<double>(mx::get_peak_memory()) / (1024.0 * 1024.0 * 1024.0);

  // Log statistics on weight tensors and memory usage.
  mlxforge::log::info("{} tensors loaded; {} non-fp16; peak memory {:.2f} GiB",
                      w.size(), non_fp16, gib);
  if (non_fp16 > 0)
    mlxforge::log::warn("{} tensors are not fp16", non_fp16);

  // Return nonzero if some tensors are not fp16, zero otherwise.
  return non_fp16 == 0 ? 0 : 1;
}

// Performs generation using a loaded model, with either raw text or pre-tokenized prompts.
int run_generate(const std::string& spec, const std::string& prompt_arg, int max_tokens) {
  // Resolve and load the model (GGUF file or safetensors dir; downloads if needed)
  LoadedModel lm = load_for_inference(spec);
  mlxforge::DecoderModel& model = *lm.model;
  mlxforge::Tokenizer& tok = lm.tok;
  const mlxforge::ModelConfig& cfg = lm.cfg;

  // Prepare the prompt, either by loading a .npy of pre-tokenized ids
  // or by applying the chat template to the provided raw prompt text.
  std::vector<int> prompt;
  if (ends_with(prompt_arg, ".npy")) {
    // Pre-tokenized prompt: load as int32 and evaluate so CPU data is current
    mx::array ids = mx::contiguous(mx::astype(mx::load(prompt_arg), mx::int32));
    mx::eval(ids);
    prompt.assign(ids.data<int32_t>(), ids.data<int32_t>() + ids.size());
  } else {
    // Raw text: wrap as user message and run through chat template
    prompt = tok.apply_chat_template({{"user", prompt_arg}});
  }

  // Create a streaming detokenizer for outputting human-readable text incrementally
  mlxforge::StreamingDetokenizer detok(tok);

  // Call greedy_generate, providing a lambda which receives each token id
  mlxforge::GenerateResult r =
      mlxforge::greedy_generate(model, prompt, max_tokens, cfg.eos_token_ids,
        [&](int id) {
          // For every generated token: detokenize and stream to stdout
          std::string piece = detok.add(id);
          std::fwrite(piece.data(), 1, piece.size(), stdout);
          std::fflush(stdout);
       });

  // Output any final detokenized tail remaining in the streaming detokenizer
  std::string tail = detok.finish();
  std::fwrite(tail.data(), 1, tail.size(), stdout);
  std::fputc('\n', stdout);

  // Log some generation statistics
  mlxforge::log::info("generated {} tokens{}", r.tokens.size(),
                      r.hit_eos ? " (stopped at EOS)" : "");
  mlxforge::log::info("time to first token {:.1f}ms; decode {:.1f} tok/s ({} tokens in {:.1f}ms)",
                      r.ttft_ms, r.decode_tokens_per_second(), r.decode_tokens, r.decode_ms);
  return 0;
}

// Repeatable throughput benchmark: a discarded warmup run absorbs Metal
// kernel-compilation cost, then `runs` timed runs report TTFT and steady-state
// decode tok/s. EOS is disabled (empty eos_ids) so every run generates exactly
// `max_tokens` — fixed-length runs are what make the numbers comparable across
// invocations and against mlx-lm. No detokenize callback, so decode timing is
// the pure forward pass.
int run_bench(const std::string& spec, int max_tokens, int runs) {
  LoadedModel lm = load_for_inference(spec);
  mlxforge::DecoderModel& model = *lm.model;
  mlxforge::Tokenizer& tok = lm.tok;

  // A fixed prompt keeps prefill length (and therefore TTFT) constant.
  const std::vector<int> prompt =
      tok.apply_chat_template({{"user", "Write a short paragraph about the ocean."}});
  const std::vector<int> no_eos;  // disable EOS so runs are exactly max_tokens long

  mlxforge::log::info("bench: prompt={} tokens, max_tokens={}, warmup=1, runs={}", prompt.size(),
                      max_tokens, runs);

  // Warmup (discarded): triggers Metal kernel compilation and cache warmup.
  mlxforge::greedy_generate(model, prompt, max_tokens, no_eos);

  double ttft_sum = 0.0, tps_sum = 0.0;
  double ttft_min = 1e300, ttft_max = 0.0, tps_min = 1e300, tps_max = 0.0;
  for (int i = 0; i < runs; ++i) {
    mlxforge::GenerateResult r = mlxforge::greedy_generate(model, prompt, max_tokens, no_eos);
    const double tps = r.decode_tokens_per_second();
    ttft_sum += r.ttft_ms;
    tps_sum += tps;
    ttft_min = std::min(ttft_min, r.ttft_ms);
    ttft_max = std::max(ttft_max, r.ttft_ms);
    tps_min = std::min(tps_min, tps);
    tps_max = std::max(tps_max, tps);
    std::printf("  run %d/%d:  ttft %.1f ms   decode %.1f tok/s  (%d tokens)\n", i + 1, runs,
                r.ttft_ms, tps, r.decode_tokens);
    std::fflush(stdout);
  }

  // Summary report goes to stdout (the benchmark's primary output, like
  // dump-weights), so it shows regardless of log level.
  std::printf("\nttft     mean %.1f ms   (min %.1f, max %.1f)\n", ttft_sum / runs, ttft_min,
              ttft_max);
  std::printf("decode   mean %.1f tok/s   (min %.1f, max %.1f)\n", tps_sum / runs, tps_min, tps_max);
  return 0;
}

// Embedding smoke harness: build a real Engine (so this exercises the exact
// library path — detection of embedding defaults, instruction wrap, EOS append,
// pooling, normalize) and print the resulting vector to stdout. This is the
// golden-reference / weight-inspection instrument for the embedding interface.
int run_embed(const std::string& spec, const std::string& text,
              const mlxforge::EmbedOptions& opts) {
  mlxforge::EngineConfig cfg;
  cfg.model_spec = spec;
  mlxforge::Engine engine(cfg);
  // The worker loads weights on its own thread; block until it is ready.
  while (!engine.ready()) std::this_thread::sleep_for(std::chrono::milliseconds(20));

  std::vector<float> v = engine.embed(text, opts);
  if (v.empty()) {
    mlxforge::log::error("embedding failed (empty input or model error)");
    return 1;
  }

  mlxforge::log::info("embedding dim {} (pooling {}, add_eos {}, normalize {})", v.size(),
                      opts.pooling, opts.add_eos, opts.normalize);
  // Vector goes to stdout (the command's primary output, like dump-weights).
  std::printf("[");
  for (size_t i = 0; i < v.size(); ++i)
    std::printf("%.6f%s", static_cast<double>(v[i]), i + 1 < v.size() ? ", " : "");
  std::printf("]\n");
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  // Initialize logging
  mlxforge::log::init();

  // Parse the command (first argument, or empty string if not provided)
  const std::string cmd = argc >= 2 ? argv[1] : "";
  if (cmd == "dump-weights") {
    // Print usage if not enough arguments; otherwise run weights dump logic
    if (argc < 3) {
      std::fprintf(stderr, "usage: mlxforge-cli dump-weights <model_dir>\n");
      return 2;
    }
    return run_dump_weights(argv[2]);
  }
  if (cmd == "generate") {
    // Print usage if not enough arguments; otherwise run generation logic
    if (argc < 4) {
      std::fprintf(stderr, "usage: mlxforge-cli generate <model_dir> <prompt_ids.npy> [max_tokens]\n");
      return 2;
    }
    // Parse max_tokens if provided, otherwise default to 64
    const int max_tokens = argc >= 5 ? std::stoi(argv[4]) : 64;
    return run_generate(argv[2], argv[3], max_tokens);
  }
  if (cmd == "bench") {
    // Repeatable throughput benchmark over a fixed prompt.
    if (argc < 3) {
      std::fprintf(stderr, "usage: mlxforge-cli bench <model_dir> [max_tokens] [runs]\n");
      return 2;
    }
    const int max_tokens = argc >= 4 ? std::stoi(argv[3]) : 128;
    const int runs = argc >= 5 ? std::stoi(argv[4]) : 3;
    return run_bench(argv[2], max_tokens, runs);
  }
  if (cmd == "embed") {
    // Embed text and print the vector. Flags override the model's detected
    // defaults (a Qwen3-Embedding checkpoint self-selects last-token + EOS).
    if (argc < 4) {
      std::fprintf(stderr,
                   "usage: mlxforge-cli embed <model_dir> <text> "
                   "[--last|--mean] [--eos] [--instruct \"...\"] [--no-normalize]\n");
      return 2;
    }
    mlxforge::EmbedOptions opts;  // pooling/add_eos = -1 (detected), normalize on
    for (int i = 4; i < argc; ++i) {
      const std::string a = argv[i];
      if (a == "--last") opts.pooling = 1;
      else if (a == "--mean") opts.pooling = 0;
      else if (a == "--eos") opts.add_eos = 1;
      else if (a == "--no-normalize") opts.normalize = false;
      else if (a == "--instruct" && i + 1 < argc) opts.instruction = argv[++i];
      else {
        std::fprintf(stderr, "embed: unknown argument '%s'\n", a.c_str());
        return 2;
      }
    }
    return run_embed(argv[2], argv[3], opts);
  }

  // No subcommand: run the smoke test by default
  return run_smoke();
}
