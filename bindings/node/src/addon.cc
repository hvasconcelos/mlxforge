// N-API binding over the libmlxforge C ABI (capi/mlxforge.h).
//
// Two object types are exposed: Engine (one batched MLX engine + worker thread)
// and Request (one in-flight generation). Request.next() runs the blocking
// C-ABI poll on a libuv worker thread via an AsyncWorker and resolves a Promise
// with the next decoded UTF-8 chunk (or null at end) — so many requests can be
// driven concurrently from JS and share the engine's continuous batching.
#include <napi.h>

#include <string>
#include <vector>

#include "mlxforge.h"

namespace {

// Translate a JS sampling options object into the flat C struct. Missing fields
// fall back to the C ABI's "disabled" sentinels (a zeroed struct => greedy). A
// `jsonSchema`/`responseFormat` string is copied into `schema_out` (so it
// outlives the submit call) and pointed at by the returned struct.
mlxforge_sampling parse_sampling(const Napi::Object& o, std::string& schema_out) {
  mlxforge_sampling s = {};
  auto num = [&](const char* k, double def) -> double {
    return (o.Has(k) && o.Get(k).IsNumber()) ? o.Get(k).As<Napi::Number>().DoubleValue() : def;
  };
  s.temperature = static_cast<float>(num("temperature", 0.0));
  s.top_k = static_cast<int>(num("topK", 0.0));
  s.top_p = static_cast<float>(num("topP", 0.0));
  s.min_p = static_cast<float>(num("minP", 0.0));
  s.repetition_penalty = static_cast<float>(num("repetitionPenalty", 0.0));
  s.frequency_penalty = static_cast<float>(num("frequencyPenalty", 0.0));
  s.presence_penalty = static_cast<float>(num("presencePenalty", 0.0));
  s.seed = static_cast<unsigned long long>(num("seed", 0.0));
  s.max_tokens = static_cast<int>(num("maxTokens", 0.0));
  // OpenAI logprobs: 0 = off; N > 0 = the chosen token's logprob plus (N - 1)
  // alternatives (so 1 = chosen-only). Retrieved via Request.logprobs().
  s.logprobs = static_cast<int>(num("logprobs", 0.0));
  if (o.Has("jsonSchema") && o.Get("jsonSchema").IsString())
    schema_out = o.Get("jsonSchema").As<Napi::String>().Utf8Value();
  else if (o.Has("responseFormat") && o.Get("responseFormat").IsString())
    schema_out = o.Get("responseFormat").As<Napi::String>().Utf8Value();
  return s;
}

Napi::FunctionReference g_request_ctor;

}  // namespace

// ---- Request --------------------------------------------------------------

class RequestWrap : public Napi::ObjectWrap<RequestWrap> {
 public:
  static void Init(Napi::Env env) {
    Napi::Function f = DefineClass(env, "Request",
                                   {
                                       InstanceMethod("next", &RequestWrap::Next),
                                       InstanceMethod("cancel", &RequestWrap::Cancel),
                                       InstanceMethod("finishReason", &RequestWrap::FinishReason),
                                       InstanceMethod("logprobs", &RequestWrap::Logprobs),
                                       InstanceMethod("dispose", &RequestWrap::Dispose),
                                   });
    g_request_ctor = Napi::Persistent(f);
    g_request_ctor.SuppressDestruct();
  }

  // Construct a JS Request wrapping an owned mlxforge_request*.
  static Napi::Object New(Napi::Env env, mlxforge_request* req) {
    return g_request_ctor.New({Napi::External<mlxforge_request>::New(env, req)});
  }

  explicit RequestWrap(const Napi::CallbackInfo& info) : Napi::ObjectWrap<RequestWrap>(info) {
    if (info.Length() >= 1 && info[0].IsExternal())
      req_ = info[0].As<Napi::External<mlxforge_request>>().Data();
  }

  ~RequestWrap() override { free_req(); }

  mlxforge_request* handle() const { return req_; }

 private:
  mlxforge_request* req_ = nullptr;

  void free_req() {
    if (req_) {
      mlxforge_request_free(req_);
      req_ = nullptr;
    }
  }

  Napi::Value Next(const Napi::CallbackInfo& info);

  void Cancel(const Napi::CallbackInfo&) {
    if (req_) mlxforge_request_cancel(req_);
  }

  Napi::Value FinishReason(const Napi::CallbackInfo& info) {
    return Napi::String::New(info.Env(), req_ ? mlxforge_request_finish_reason(req_) : "");
  }

  // The accumulated per-token log-probs, parsed from the C ABI's OpenAI-shaped
  // JSON (or null when none / not requested). Call after the stream is drained,
  // before dispose().
  Napi::Value Logprobs(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    char* j = req_ ? mlxforge_request_logprobs(req_) : nullptr;
    if (!j) return env.Null();
    std::string s(j);
    mlxforge_string_free(j);
    Napi::Object json = env.Global().Get("JSON").As<Napi::Object>();
    return json.Get("parse").As<Napi::Function>().Call(json, {Napi::String::New(env, s)});
  }

  void Dispose(const Napi::CallbackInfo&) { free_req(); }
};

// Runs one blocking mlxforge_request_next() off the event loop and resolves the
// Promise with the chunk (or null when the stream ends).
class NextWorker : public Napi::AsyncWorker {
 public:
  NextWorker(Napi::Env env, mlxforge_request* req, Napi::Promise::Deferred deferred)
      : Napi::AsyncWorker(env), req_(req), deferred_(deferred) {}

  void Execute() override {
    if (!req_) {
      done_ = true;
      return;
    }
    char* text = nullptr;
    int rc = mlxforge_request_next(req_, &text);
    if (rc == 0 && text) {
      chunk_.assign(text);
      mlxforge_string_free(text);
      have_ = true;
    } else {
      done_ = true;  // rc == 1 (done) or rc < 0 (error): end the stream
    }
  }

  void OnOK() override {
    Napi::HandleScope scope(Env());
    if (have_)
      deferred_.Resolve(Napi::String::New(Env(), chunk_));
    else
      deferred_.Resolve(Env().Null());
  }

  void OnError(const Napi::Error& e) override { deferred_.Reject(e.Value()); }

 private:
  mlxforge_request* req_;
  Napi::Promise::Deferred deferred_;
  std::string chunk_;
  bool have_ = false;
  bool done_ = false;
};

Napi::Value RequestWrap::Next(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  auto deferred = Napi::Promise::Deferred::New(env);
  if (!req_) {
    deferred.Resolve(env.Null());
    return deferred.Promise();
  }
  (new NextWorker(env, req_, deferred))->Queue();
  return deferred.Promise();
}

// Runs the blocking mlxforge_embed_ex off the event loop and resolves a
// Float32Array. Carries the full option set (pooling/add_eos/instruction/
// normalize); -1 pooling/add_eos defer to the model's detected defaults.
class EmbedWorker : public Napi::AsyncWorker {
 public:
  EmbedWorker(Napi::Env env, mlxforge_engine* eng, std::string text, int pooling, int add_eos,
              int skip_normalize, std::string instruction, Napi::Promise::Deferred deferred)
      : Napi::AsyncWorker(env),
        eng_(eng),
        text_(std::move(text)),
        pooling_(pooling),
        add_eos_(add_eos),
        skip_normalize_(skip_normalize),
        instruction_(std::move(instruction)),
        deferred_(deferred) {}

  void Execute() override {
    char* err = nullptr;
    float* v = nullptr;
    size_t n = 0;
    mlxforge_embed_opts opts = {};
    opts.pooling = pooling_;
    opts.add_eos = add_eos_;
    opts.skip_normalize = skip_normalize_;
    opts.instruction = instruction_.empty() ? nullptr : instruction_.c_str();
    int rc = mlxforge_embed_ex(eng_, text_.c_str(), &opts, &v, &n, &err);
    if (rc == 0 && v) {
      data_.assign(v, v + n);
      mlxforge_floats_free(v);
    } else {
      errmsg_ = err ? err : "embed failed";
    }
    mlxforge_string_free(err);
  }

  void OnOK() override {
    Napi::HandleScope scope(Env());
    if (!errmsg_.empty()) {
      deferred_.Reject(Napi::Error::New(Env(), errmsg_).Value());
      return;
    }
    auto arr = Napi::Float32Array::New(Env(), data_.size());
    for (size_t i = 0; i < data_.size(); ++i) arr[i] = data_[i];
    deferred_.Resolve(arr);
  }

  void OnError(const Napi::Error& e) override { deferred_.Reject(e.Value()); }

 private:
  mlxforge_engine* eng_;
  std::string text_;
  int pooling_;
  int add_eos_;
  int skip_normalize_;
  std::string instruction_;
  Napi::Promise::Deferred deferred_;
  std::vector<float> data_;
  std::string errmsg_;
};

// ---- Engine ---------------------------------------------------------------

class EngineWrap : public Napi::ObjectWrap<EngineWrap> {
 public:
  static Napi::Object Init(Napi::Env env, Napi::Object exports) {
    Napi::Function f = DefineClass(env, "Engine",
                                   {
                                       InstanceMethod("ready", &EngineWrap::Ready),
                                       InstanceMethod("modelName", &EngineWrap::ModelName),
                                       InstanceMethod("submitChat", &EngineWrap::SubmitChat),
                                       InstanceMethod("submitText", &EngineWrap::SubmitText),
                                       InstanceMethod("submitImage", &EngineWrap::SubmitImage),
                                       InstanceMethod("submitImages", &EngineWrap::SubmitImages),
                                       InstanceMethod("embed", &EngineWrap::Embed),
                                       InstanceMethod("dispose", &EngineWrap::Dispose),
                                   });
    exports.Set("Engine", f);
    return exports;
  }

  explicit EngineWrap(const Napi::CallbackInfo& info) : Napi::ObjectWrap<EngineWrap>(info) {
    Napi::Env env = info.Env();
    if (info.Length() < 1 || !info[0].IsString())
      throw Napi::TypeError::New(env, "new Engine(spec, opts?) requires a model spec string");
    std::string spec = info[0].As<Napi::String>().Utf8Value();

    mlxforge_engine_opts2 opts = {};
    opts.struct_size = sizeof(opts);
    std::string spill_dir;  // must outlive the create call (opts borrows it)
    if (info.Length() >= 2 && info[1].IsObject()) {
      Napi::Object o = info[1].As<Napi::Object>();
      if (o.Has("maxWaiting") && o.Get("maxWaiting").IsNumber())
        opts.max_waiting = o.Get("maxWaiting").As<Napi::Number>().Int32Value();
      if (o.Has("kvBits") && o.Get("kvBits").IsNumber())
        opts.kv_bits = o.Get("kvBits").As<Napi::Number>().Int32Value();
      if (o.Has("kvGroupSize") && o.Get("kvGroupSize").IsNumber())
        opts.kv_group_size = o.Get("kvGroupSize").As<Napi::Number>().Int32Value();
      if (o.Has("prefixCache") && o.Get("prefixCache").IsBoolean())
        opts.prefix_cache = o.Get("prefixCache").As<Napi::Boolean>().Value() ? 1 : 0;
      if (o.Has("kvBlockSize") && o.Get("kvBlockSize").IsNumber())
        opts.kv_block_size = o.Get("kvBlockSize").As<Napi::Number>().Int32Value();
      if (o.Has("kvPoolBytes") && o.Get("kvPoolBytes").IsNumber())
        opts.kv_pool_bytes = o.Get("kvPoolBytes").As<Napi::Number>().Int64Value();
      if (o.Has("kvSpillDir") && o.Get("kvSpillDir").IsString()) {
        spill_dir = o.Get("kvSpillDir").As<Napi::String>().Utf8Value();
        opts.kv_spill_dir = spill_dir.c_str();
      }
      if (o.Has("kvSpillBytes") && o.Get("kvSpillBytes").IsNumber())
        opts.kv_spill_bytes = o.Get("kvSpillBytes").As<Napi::Number>().Int64Value();
    }

    char* err = nullptr;
    eng_ = mlxforge_engine_create2(spec.c_str(), &opts, &err);
    if (!eng_) {
      std::string msg = err ? err : "failed to create engine";
      mlxforge_string_free(err);
      throw Napi::Error::New(env, msg);
    }
  }

  ~EngineWrap() override { free_engine(); }

 private:
  mlxforge_engine* eng_ = nullptr;

  void free_engine() {
    if (eng_) {
      mlxforge_engine_free(eng_);
      eng_ = nullptr;
    }
  }

  Napi::Value Ready(const Napi::CallbackInfo& info) {
    return Napi::Boolean::New(info.Env(), eng_ && mlxforge_engine_ready(eng_));
  }

  Napi::Value ModelName(const Napi::CallbackInfo& info) {
    return Napi::String::New(info.Env(), eng_ ? mlxforge_engine_model_name(eng_) : "");
  }

  // Shared submit tail: turn an mlxforge_request* (or null + err) into a JS
  // Request, throwing on failure.
  Napi::Value finish_submit(Napi::Env env, mlxforge_request* req, char* err) {
    if (!req) {
      std::string msg = err ? err : "request rejected";
      mlxforge_string_free(err);
      throw Napi::Error::New(env, msg);
    }
    return RequestWrap::New(env, req);
  }

  Napi::Value SubmitChat(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (!eng_) throw Napi::Error::New(env, "engine is disposed");
    if (info.Length() < 1 || !info[0].IsArray())
      throw Napi::TypeError::New(env, "submitChat(messages[], sampling?)");

    Napi::Array arr = info[0].As<Napi::Array>();
    // Own the role/content strings for the duration of the call; reserve so the
    // vector never reallocates (msgs holds c_str() pointers into it).
    std::vector<std::string> store;
    store.reserve(static_cast<size_t>(arr.Length()) * 2);
    std::vector<mlxforge_msg> msgs;
    msgs.reserve(arr.Length());
    for (uint32_t i = 0; i < arr.Length(); ++i) {
      Napi::Object m = arr.Get(i).As<Napi::Object>();
      store.push_back(m.Get("role").As<Napi::String>().Utf8Value());
      store.push_back(m.Get("content").As<Napi::String>().Utf8Value());
      msgs.push_back({store[store.size() - 2].c_str(), store[store.size() - 1].c_str()});
    }

    std::string schema;  // kept alive across the submit call
    mlxforge_sampling s = (info.Length() >= 2 && info[1].IsObject())
                              ? parse_sampling(info[1].As<Napi::Object>(), schema)
                              : mlxforge_sampling{};
    if (!schema.empty()) s.json_schema = schema.c_str();
    char* err = nullptr;
    mlxforge_request* req = mlxforge_submit_chat(eng_, msgs.data(), msgs.size(), &s, &err);
    return finish_submit(env, req, err);
  }

  Napi::Value SubmitText(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (!eng_) throw Napi::Error::New(env, "engine is disposed");
    if (info.Length() < 1 || !info[0].IsString())
      throw Napi::TypeError::New(env, "submitText(prompt, sampling?)");
    std::string prompt = info[0].As<Napi::String>().Utf8Value();
    std::string schema;  // kept alive across the submit call
    mlxforge_sampling s = (info.Length() >= 2 && info[1].IsObject())
                              ? parse_sampling(info[1].As<Napi::Object>(), schema)
                              : mlxforge_sampling{};
    if (!schema.empty()) s.json_schema = schema.c_str();
    char* err = nullptr;
    mlxforge_request* req = mlxforge_submit_text(eng_, prompt.c_str(), &s, &err);
    return finish_submit(env, req, err);
  }

  Napi::Value SubmitImage(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (!eng_) throw Napi::Error::New(env, "engine is disposed");
    if (info.Length() < 2 || !info[0].IsString() || !info[1].IsBuffer())
      throw Napi::TypeError::New(env, "submitImage(prompt, imageBuffer, sampling?)");
    std::string prompt = info[0].As<Napi::String>().Utf8Value();
    Napi::Buffer<uint8_t> buf = info[1].As<Napi::Buffer<uint8_t>>();
    std::string schema;  // kept alive across the submit call
    mlxforge_sampling s = (info.Length() >= 3 && info[2].IsObject())
                              ? parse_sampling(info[2].As<Napi::Object>(), schema)
                              : mlxforge_sampling{};
    if (!schema.empty()) s.json_schema = schema.c_str();
    char* err = nullptr;
    mlxforge_request* req =
        mlxforge_submit_image(eng_, prompt.c_str(), buf.Data(), buf.Length(), &s, &err);
    return finish_submit(env, req, err);
  }

  Napi::Value SubmitImages(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (!eng_) throw Napi::Error::New(env, "engine is disposed");
    if (info.Length() < 2 || !info[0].IsString() || !info[1].IsArray())
      throw Napi::TypeError::New(env, "submitImages(prompt, imageBuffers[], sampling?)");
    std::string prompt = info[0].As<Napi::String>().Utf8Value();
    Napi::Array arr = info[1].As<Napi::Array>();
    // Pointers into the JS Buffers; valid for the synchronous submit call (the
    // engine copies the bytes before returning). The Buffers stay alive via `arr`.
    std::vector<mlxforge_image> images;
    images.reserve(arr.Length());
    for (uint32_t i = 0; i < arr.Length(); ++i) {
      Napi::Value v = arr.Get(i);
      if (!v.IsBuffer()) throw Napi::TypeError::New(env, "each image must be a Buffer");
      Napi::Buffer<uint8_t> buf = v.As<Napi::Buffer<uint8_t>>();
      images.push_back(mlxforge_image{buf.Data(), buf.Length()});
    }
    std::string schema;  // kept alive across the submit call
    mlxforge_sampling s = (info.Length() >= 3 && info[2].IsObject())
                              ? parse_sampling(info[2].As<Napi::Object>(), schema)
                              : mlxforge_sampling{};
    if (!schema.empty()) s.json_schema = schema.c_str();
    char* err = nullptr;
    mlxforge_request* req =
        mlxforge_submit_images(eng_, prompt.c_str(), images.data(), images.size(), &s, &err);
    return finish_submit(env, req, err);
  }

  Napi::Value Embed(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    auto deferred = Napi::Promise::Deferred::New(env);
    if (!eng_) {
      deferred.Reject(Napi::Error::New(env, "engine is disposed").Value());
      return deferred.Promise();
    }
    std::string text =
        (info.Length() >= 1 && info[0].IsString()) ? info[0].As<Napi::String>().Utf8Value() : "";
    // Default to the model's detected behavior (-1): a Qwen3-Embedding checkpoint
    // self-selects last-token pooling + a trailing EOS.
    int pooling = -1, add_eos = -1, skip_normalize = 0;
    std::string instruction;
    if (info.Length() >= 2) {
      if (info[1].IsNumber()) {
        pooling = info[1].As<Napi::Number>().Int32Value();  // legacy: embed(text, pooling)
      } else if (info[1].IsObject()) {
        Napi::Object o = info[1].As<Napi::Object>();
        if (o.Has("pooling") && o.Get("pooling").IsNumber())
          pooling = o.Get("pooling").As<Napi::Number>().Int32Value();
        if (o.Has("addEos") && o.Get("addEos").IsBoolean())
          add_eos = o.Get("addEos").As<Napi::Boolean>().Value() ? 1 : 0;
        if (o.Has("normalize") && o.Get("normalize").IsBoolean())
          skip_normalize = o.Get("normalize").As<Napi::Boolean>().Value() ? 0 : 1;
        if (o.Has("instruction") && o.Get("instruction").IsString())
          instruction = o.Get("instruction").As<Napi::String>().Utf8Value();
      }
    }
    (new EmbedWorker(env, eng_, std::move(text), pooling, add_eos, skip_normalize,
                     std::move(instruction), deferred))
        ->Queue();
    return deferred.Promise();
  }

  void Dispose(const Napi::CallbackInfo&) { free_engine(); }
};

// ---- Module ---------------------------------------------------------------

static Napi::Object InitAll(Napi::Env env, Napi::Object exports) {
  RequestWrap::Init(env);
  EngineWrap::Init(env, exports);
  exports.Set("version", Napi::String::New(env, mlxforge_version()));
  exports.Set("abiVersion", Napi::Number::New(env, mlxforge_abi_version()));
  return exports;
}

NODE_API_MODULE(mlxforge_node, InitAll)
