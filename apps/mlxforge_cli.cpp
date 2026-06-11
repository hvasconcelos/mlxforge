// mlxforge-cli — command-line entry point.
//
// Usage patterns:
//   mlxforge-cli
//     - Performs a smoke test: adds two arrays, evaluates the computation, and prints the sum.
//   mlxforge-cli dump-weights <dir>
//     - Loads a model's weights from the supplied directory, prints key/shape/dtype for each tensor,
//       asserts that all tensors are fp16, and reports the peak resident memory used.
//   mlxforge-cli generate <model> <prompt> [max_tokens] [--logprobs [N]] [--kv-bits N]
//     - Runs greedy single-stream generation: pre-fills the prompt (as raw text using the chat template
//       or as a pre-tokenized .npy of ids), then streams the detokenized text to stdout until EOS or
//       max_tokens. With --logprobs [N], each emitted token's log-prob (and its N most-likely
//       alternatives) is printed to stderr after generation; stdout stays the generated text.
//       --kv-bits 8|4 stores the KV cache quantized (the manual harness for the quantized path).
//   mlxforge-cli bench <model> [max_tokens] [runs]
//     - Repeatable throughput benchmark over a fixed prompt: one discarded warmup run, then `runs`
//       timed runs (defaults: max_tokens=128, runs=3) reporting time-to-first-token and decode tok/s.
//   mlxforge-cli bench-prefix <model> [prefix_tokens] [runs]
//     - Prefix-cache benchmark (defaults: prefix_tokens=2048, runs=3): builds an Engine with the
//       prefix cache on, then measures TTFT for one COLD request and `runs` WARM requests that share
//       a prefix_tokens-long prompt prefix (distinct tails). Reports the cold/warm speedup and the
//       engine's reuse metrics; warm decode tok/s shows reuse leaves throughput unchanged.
//   mlxforge-cli embed <model> <text> [--last|--mean] [--eos] [--instruct "..."] [--no-normalize]
//     - Embeds text and prints the (by default unit-normalized) vector. With no flags the model
//       self-selects its convention (a Qwen3-Embedding checkpoint uses last-token pooling + a
//       trailing EOS). The embedding smoke/golden-reference harness for the library.
//
// <dir>/<model> is either a local model directory or a HuggingFace repo id (e.g. mlx-community/Llama-3.2-1B-Instruct-4bit),
// which will be downloaded on first use.

#include <algorithm>
#include <cctype>
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
#include "model/qwen3_vl.h"
#include "model/vision/vit.h"
#include "runtime/engine.h"
#include "runtime/multimodal_stream.h"
#include "runtime/single_stream.h"
#include "scheduler/request.h"
#include "tokenizer/tokenizer.h"
#include "vision/image_decode.h"

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

// Render a token's text for the logprobs dump: quoted, with the common control
// characters escaped so whitespace tokens stay legible on one line.
std::string show_token(const std::string& s) {
  std::string out = "\"";
  for (char c : s) {
    if (c == '\n') out += "\\n";
    else if (c == '\t') out += "\\t";
    else if (c == '\r') out += "\\r";
    else out += c;
  }
  return out + "\"";
}

// Performs generation using a loaded model, with either raw text or pre-tokenized prompts.
// `top_logprobs` mirrors the engine knob: -1 = off; 0 = each token's own log-prob;
// N > 0 = also its N most-likely alternatives (printed to stderr after generation).
int run_generate(const std::string& spec, const std::string& prompt_arg, int max_tokens,
                 int top_logprobs = -1, int kv_bits = 0) {
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
       }, top_logprobs, mlxforge::KVQuantConfig{kv_bits, 64});

  // Output any final detokenized tail remaining in the streaming detokenizer
  std::string tail = detok.finish();
  std::fwrite(tail.data(), 1, tail.size(), stdout);
  std::fputc('\n', stdout);

  // Per-token log-probs go to stderr (stdout stays the generated text). One line
  // per emitted token: its text + log-prob, then any requested alternatives.
  if (top_logprobs >= 0) {
    std::fprintf(stderr, "logprobs (%zu tokens):\n", r.token_logprobs.size());
    for (const mlxforge::TokenLogprob& lp : r.token_logprobs) {
      std::fprintf(stderr, "  %-16s logprob=%8.4f", show_token(tok.decode({lp.id})).c_str(),
                   lp.logprob);
      if (!lp.top.empty()) {
        std::fprintf(stderr, "  top:");
        for (const auto& alt : lp.top)
          std::fprintf(stderr, " %s=%.4f", show_token(tok.decode({alt.first})).c_str(), alt.second);
      }
      std::fputc('\n', stderr);
    }
  }

  // Log some generation statistics
  mlxforge::log::info("generated {} tokens{}", r.tokens.size(),
                      r.hit_eos ? " (stopped at EOS)" : "");
  mlxforge::log::info("time to first token {:.1f}ms; decode {:.1f} tok/s ({} tokens in {:.1f}ms)",
                      r.ttft_ms, r.decode_tokens_per_second(), r.decode_tokens, r.decode_ms);
  return 0;
}

// Vision-language generation: decode an image, run it through the ViT, and
// generate a response to `prompt_arg` about it. The model must be a Qwen3-VL
// checkpoint (has a vision tower); the ViT borrows the loaded model's weights.
int run_generate_image(const std::string& spec, const std::string& image_path,
                       const std::string& prompt_arg, int max_tokens) {
  LoadedModel lm = load_for_inference(spec);
  auto* vl = dynamic_cast<mlxforge::Qwen3VLModel*>(lm.model.get());
  if (vl == nullptr || !lm.cfg.has_vision_tower()) {
    mlxforge::log::error("model '{}' is not a vision-language model (no ViT)", spec);
    return 1;
  }
  mlxforge::VitEncoder vit(*lm.cfg.vision, lm.model->weights());
  mx::array image = mlxforge::decode_image_file(image_path);

  mlxforge::StreamingDetokenizer detok(lm.tok);
  mlxforge::GenerateResult r = mlxforge::generate_from_image(
      *vl, vit, lm.tok, prompt_arg, image, max_tokens, lm.cfg.eos_token_ids, [&](int id) {
        std::string piece = detok.add(id);
        std::fwrite(piece.data(), 1, piece.size(), stdout);
        std::fflush(stdout);
      });
  std::string tail = detok.finish();
  std::fwrite(tail.data(), 1, tail.size(), stdout);
  std::fputc('\n', stdout);

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

// Prefix-cache benchmark: build a real Engine (the exact library path, prefix
// cache on) and measure time-to-first-token for prompts sharing a long prefix —
// the shared-system-prompt scenario the cache exists for. One discarded warmup
// absorbs Metal kernel compilation; the COLD run then prefills the shared
// prefix from scratch, and each WARM run reuses its pooled blocks with a unique
// tail (a distinct "user question"). Greedy, EOS disabled, fixed max_tokens, so
// runs are comparable; decode tok/s is reported to show reuse does not change
// steady-state throughput.
int run_bench_prefix(const std::string& spec, int prefix_tokens, int runs) {
  mlxforge::EngineConfig cfg;
  cfg.model_spec = spec;
  cfg.prefix_cache = true;
  mlxforge::Engine engine(cfg);
  while (!engine.ready()) std::this_thread::sleep_for(std::chrono::milliseconds(20));
  const mlxforge::Tokenizer& tok = engine.tokenizer();

  // Build a shared prefix of ~prefix_tokens ids by repeating a paragraph.
  const std::vector<int> para =
      tok.encode("The ocean covers most of the planet, and its slow currents move heat between "
                 "the equator and the poles, shaping weather on every continent. ");
  std::vector<int> prefix;
  while (static_cast<int>(prefix.size()) < prefix_tokens)
    prefix.insert(prefix.end(), para.begin(), para.end());
  prefix.resize(prefix_tokens);

  const int kMaxTokens = 32;
  // Submit a prompt through the scheduler (the continuous-batching path the
  // server and bindings use) and return {ttft_ms, decode tok/s}.
  auto timed_run = [&](std::vector<int> ids) {
    auto req = std::make_shared<mlxforge::Request>();
    req->prompt_ids = std::move(ids);
    req->params.temperature = 0.0f;
    req->max_tokens = kMaxTokens;  // eos_ids stays empty: fixed-length runs
    const auto t0 = std::chrono::steady_clock::now();
    engine.scheduler().submit(req);
    double ttft_ms = 0.0;
    auto t_first = t0;
    int produced = 0, tk = 0;
    while (req->tokens.pop(tk)) {
      if (produced++ == 0) {
        t_first = std::chrono::steady_clock::now();
        ttft_ms = std::chrono::duration<double, std::milli>(t_first - t0).count();
      }
    }
    const double decode_s =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t_first).count();
    const double tps = (produced > 1 && decode_s > 0) ? (produced - 1) / decode_s : 0.0;
    return std::make_pair(ttft_ms, tps);
  };
  auto with_tail = [&](int i) {
    std::vector<int> ids = prefix;
    const std::vector<int> tail =
        tok.encode("Question " + std::to_string(i) + ": summarize the key point briefly.");
    ids.insert(ids.end(), tail.begin(), tail.end());
    return ids;
  };

  mlxforge::log::info("bench-prefix: prefix={} tokens, max_tokens={}, warmup=1, runs={}",
                      prefix.size(), kMaxTokens, runs);

  // Warmup (discarded): an UNRELATED prompt — it absorbs Metal kernel
  // compilation but shares no prefix, so the cold run below stays cold.
  timed_run(tok.encode("A completely unrelated warmup prompt about gardening tools."));
  const auto [cold_ttft, cold_tps] = timed_run(with_tail(0));
  std::printf("  cold:      ttft %8.1f ms   decode %.1f tok/s\n", cold_ttft, cold_tps);
  std::fflush(stdout);

  double warm_sum = 0.0, warm_min = 1e300, warm_max = 0.0, tps_sum = 0.0;
  for (int i = 1; i <= runs; ++i) {
    const auto [ttft, tps] = timed_run(with_tail(i));
    warm_sum += ttft;
    tps_sum += tps;
    warm_min = std::min(warm_min, ttft);
    warm_max = std::max(warm_max, ttft);
    std::printf("  warm %d/%d:  ttft %8.1f ms   decode %.1f tok/s\n", i, runs, ttft, tps);
    std::fflush(stdout);
  }

  const mlxforge::WorkerMetrics m = engine.metrics();
  const double warm_mean = warm_sum / runs;
  std::printf("\nttft       cold %.1f ms   warm mean %.1f ms (min %.1f, max %.1f)   speedup %.1fx\n",
              cold_ttft, warm_mean, warm_min, warm_max,
              warm_mean > 0 ? cold_ttft / warm_mean : 0.0);
  std::printf("decode     cold %.1f tok/s   warm mean %.1f tok/s\n", cold_tps, tps_sum / runs);
  std::printf("reuse      hits %ld   tokens reused %lld   pool %ld blocks / %lld bytes\n",
              m.prefix_hits, m.prefix_tokens_reused, m.prefix_pool_blocks, m.prefix_pool_bytes);
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
      std::fprintf(stderr,
                   "usage: mlxforge-cli generate <model_dir> <prompt_ids.npy> [max_tokens] "
                   "[--logprobs [N]] [--kv-bits N]\n");
      return 2;
    }
    // Positional [max_tokens] (default 64), an optional --logprobs [N] flag (N
    // alternatives, default 0 = the chosen token's own log-prob only), and an
    // optional --kv-bits N (0 = fp16 cache, 8 or 4 = quantized).
    int max_tokens = 64;
    int top_logprobs = -1;
    int kv_bits = 0;
    for (int i = 4; i < argc; ++i) {
      const std::string a = argv[i];
      if (a == "--logprobs") {
        if (i + 1 < argc && std::isdigit(static_cast<unsigned char>(argv[i + 1][0])))
          top_logprobs = std::stoi(argv[++i]);
        else
          top_logprobs = 0;
      } else if (a == "--kv-bits") {
        if (i + 1 >= argc) {
          std::fprintf(stderr, "error: --kv-bits needs a value (0, 4, or 8)\n");
          return 2;
        }
        kv_bits = std::stoi(argv[++i]);
        if (kv_bits != 0 && kv_bits != 4 && kv_bits != 8) {
          std::fprintf(stderr, "error: --kv-bits must be 0, 4, or 8\n");
          return 2;
        }
      } else {
        max_tokens = std::stoi(a);
      }
    }
    return run_generate(argv[2], argv[3], max_tokens, top_logprobs, kv_bits);
  }
  if (cmd == "image") {
    // Vision-language generation: describe / answer about an image.
    if (argc < 5) {
      std::fprintf(stderr,
                   "usage: mlxforge-cli image <model_dir> <image_file> <prompt> [max_tokens]\n");
      return 2;
    }
    const int max_tokens = argc >= 6 ? std::stoi(argv[5]) : 128;
    return run_generate_image(argv[2], argv[3], argv[4], max_tokens);
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
  if (cmd == "bench-prefix") {
    // Prefix-cache benchmark: cold vs warm TTFT over a shared prompt prefix.
    if (argc < 3) {
      std::fprintf(stderr, "usage: mlxforge-cli bench-prefix <model_dir> [prefix_tokens] [runs]\n");
      return 2;
    }
    const int prefix_tokens = argc >= 4 ? std::stoi(argv[3]) : 2048;
    const int runs = argc >= 5 ? std::stoi(argv[4]) : 3;
    return run_bench_prefix(argv[2], prefix_tokens, runs);
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
