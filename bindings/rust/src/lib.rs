//! Rust bindings for **libmlxforge** — an embeddable, MLX-native, continuously
//! batched LLM engine for Apple Silicon. Thin and safe over the C ABI in
//! `src/capi/mlxforge.h`; this is a reference binding proving the engine is
//! language-neutral (the same C ABI backs the Node and Swift bindings).
//!
//! ```no_run
//! use mlxforge::{Engine, Sampling};
//! let engine = Engine::load("mlx-community/Llama-3.2-1B-Instruct-4bit").unwrap();
//! let reply = engine.chat(&[("user", "Tell me a joke.")], &Sampling::greedy()).unwrap();
//! println!("{reply}");
//! let v = engine.embed("The cat sat on the mat.", 0).unwrap(); // unit-normalized
//! ```

use std::ffi::{c_char, c_float, c_int, c_void, CStr, CString};
use std::ptr;

#[allow(non_camel_case_types)]
type mlxforge_engine = c_void;
#[allow(non_camel_case_types)]
type mlxforge_request = c_void;

#[repr(C)]
struct EngineOpts {
    max_waiting: c_int,
}

#[repr(C)]
struct Msg {
    role: *const c_char,
    content: *const c_char,
}

#[repr(C)]
struct CImage {
    data: *const u8,
    len: usize,
}

#[repr(C)]
struct CSampling {
    temperature: c_float,
    top_k: c_int,
    top_p: c_float,
    min_p: c_float,
    repetition_penalty: c_float,
    frequency_penalty: c_float,
    presence_penalty: c_float,
    seed: u64,
    max_tokens: c_int,
    json_schema: *const c_char,
}

// Mirrors mlxforge_embed_opts. pooling/add_eos are tri-state: -1 = model default.
#[repr(C)]
struct CEmbedOpts {
    pooling: c_int,
    add_eos: c_int,
    skip_normalize: c_int,
    instruction: *const c_char,
}

extern "C" {
    fn mlxforge_version() -> *const c_char;
    fn mlxforge_abi_version() -> c_int;
    fn mlxforge_string_free(s: *mut c_char);
    fn mlxforge_floats_free(p: *mut c_float);

    fn mlxforge_engine_create(
        spec: *const c_char,
        opts: *const EngineOpts,
        err: *mut *mut c_char,
    ) -> *mut mlxforge_engine;
    fn mlxforge_engine_ready(e: *mut mlxforge_engine) -> c_int;
    fn mlxforge_engine_model_name(e: *mut mlxforge_engine) -> *const c_char;
    fn mlxforge_engine_free(e: *mut mlxforge_engine);

    fn mlxforge_embed(
        e: *mut mlxforge_engine,
        text: *const c_char,
        pooling: c_int,
        out: *mut *mut c_float,
        out_len: *mut usize,
        err: *mut *mut c_char,
    ) -> c_int;
    fn mlxforge_embed_ex(
        e: *mut mlxforge_engine,
        text: *const c_char,
        opts: *const CEmbedOpts,
        out: *mut *mut c_float,
        out_len: *mut usize,
        err: *mut *mut c_char,
    ) -> c_int;

    fn mlxforge_submit_chat(
        e: *mut mlxforge_engine,
        messages: *const Msg,
        n: usize,
        sampling: *const CSampling,
        err: *mut *mut c_char,
    ) -> *mut mlxforge_request;
    fn mlxforge_submit_text(
        e: *mut mlxforge_engine,
        prompt: *const c_char,
        sampling: *const CSampling,
        err: *mut *mut c_char,
    ) -> *mut mlxforge_request;
    fn mlxforge_submit_image(
        e: *mut mlxforge_engine,
        prompt: *const c_char,
        image_data: *const u8,
        image_len: usize,
        sampling: *const CSampling,
        err: *mut *mut c_char,
    ) -> *mut mlxforge_request;
    fn mlxforge_submit_images(
        e: *mut mlxforge_engine,
        prompt: *const c_char,
        images: *const CImage,
        n_images: usize,
        sampling: *const CSampling,
        err: *mut *mut c_char,
    ) -> *mut mlxforge_request;
    fn mlxforge_request_next(r: *mut mlxforge_request, text: *mut *mut c_char) -> c_int;
    fn mlxforge_request_finish_reason(r: *mut mlxforge_request) -> *const c_char;
    fn mlxforge_request_free(r: *mut mlxforge_request);
}

/// libmlxforge version string.
pub fn version() -> String {
    unsafe { CStr::from_ptr(mlxforge_version()).to_string_lossy().into_owned() }
}

/// The C ABI version this build implements.
pub fn abi_version() -> i32 {
    unsafe { mlxforge_abi_version() }
}

/// Take ownership of a C string the library allocated and free it.
unsafe fn take_string(p: *mut c_char) -> Option<String> {
    if p.is_null() {
        return None;
    }
    let s = CStr::from_ptr(p).to_string_lossy().into_owned();
    mlxforge_string_free(p);
    Some(s)
}

/// Sampling parameters. Default is deterministic greedy decoding.
#[derive(Clone, Default)]
pub struct Sampling {
    pub temperature: f32,
    pub top_k: i32,
    pub top_p: f32,
    pub min_p: f32,
    pub repetition_penalty: f32,
    pub frequency_penalty: f32,
    pub presence_penalty: f32,
    pub seed: u64,
    pub max_tokens: i32,
    /// Constrained decoding: "json" or a JSON-Schema string (see the C ABI docs).
    pub json_schema: Option<String>,
}

impl Sampling {
    pub fn greedy() -> Self {
        Sampling::default()
    }
}

/// A batched MLX LLM engine. Many calls may run concurrently from different
/// threads on one engine — they share its continuous-batching scheduler.
pub struct Engine {
    handle: *mut mlxforge_engine,
}

// The engine is internally synchronized (one GPU worker thread); handles may be
// shared across threads.
unsafe impl Send for Engine {}
unsafe impl Sync for Engine {}

impl Engine {
    /// Create an engine for a model spec (local dir, HF repo id, or .gguf).
    pub fn new(spec: &str) -> Result<Engine, String> {
        let cspec = CString::new(spec).map_err(|_| "spec contains NUL".to_string())?;
        let opts = EngineOpts { max_waiting: 0 };
        let mut err: *mut c_char = ptr::null_mut();
        let handle = unsafe { mlxforge_engine_create(cspec.as_ptr(), &opts, &mut err) };
        if handle.is_null() {
            return Err(unsafe { take_string(err) }.unwrap_or_else(|| "engine create failed".into()));
        }
        Ok(Engine { handle })
    }

    /// Create an engine and block until the model has finished loading.
    pub fn load(spec: &str) -> Result<Engine, String> {
        let e = Engine::new(spec)?;
        while !e.ready() {
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        Ok(e)
    }

    pub fn ready(&self) -> bool {
        unsafe { mlxforge_engine_ready(self.handle) != 0 }
    }

    pub fn model_name(&self) -> String {
        unsafe {
            CStr::from_ptr(mlxforge_engine_model_name(self.handle))
                .to_string_lossy()
                .into_owned()
        }
    }

    fn c_sampling<'a>(
        s: &Sampling,
        schema_keep: &'a mut Option<CString>,
    ) -> CSampling {
        let json_schema = match &s.json_schema {
            Some(js) => {
                *schema_keep = CString::new(js.as_str()).ok();
                schema_keep.as_ref().map_or(ptr::null(), |c| c.as_ptr())
            }
            None => ptr::null(),
        };
        CSampling {
            temperature: s.temperature,
            top_k: s.top_k,
            top_p: s.top_p,
            min_p: s.min_p,
            repetition_penalty: s.repetition_penalty,
            frequency_penalty: s.frequency_penalty,
            presence_penalty: s.presence_penalty,
            seed: s.seed,
            max_tokens: s.max_tokens,
            json_schema,
        }
    }

    /// Run a chat completion to completion and return the full text.
    pub fn chat(&self, messages: &[(&str, &str)], sampling: &Sampling) -> Result<String, String> {
        // Own the role/content C strings for the duration of the submit call.
        let owned: Vec<(CString, CString)> = messages
            .iter()
            .map(|(r, c)| {
                (
                    CString::new(*r).unwrap_or_default(),
                    CString::new(*c).unwrap_or_default(),
                )
            })
            .collect();
        let msgs: Vec<Msg> = owned
            .iter()
            .map(|(r, c)| Msg {
                role: r.as_ptr(),
                content: c.as_ptr(),
            })
            .collect();

        let mut schema_keep: Option<CString> = None;
        let cs = Self::c_sampling(sampling, &mut schema_keep);
        let mut err: *mut c_char = ptr::null_mut();
        let req =
            unsafe { mlxforge_submit_chat(self.handle, msgs.as_ptr(), msgs.len(), &cs, &mut err) };
        if req.is_null() {
            return Err(unsafe { take_string(err) }.unwrap_or_else(|| "submit failed".into()));
        }
        Ok(drain(req))
    }

    /// Run a raw-text completion (no chat template) to completion.
    pub fn text(&self, prompt: &str, sampling: &Sampling) -> Result<String, String> {
        let cprompt = CString::new(prompt).map_err(|_| "prompt contains NUL".to_string())?;
        let mut schema_keep: Option<CString> = None;
        let cs = Self::c_sampling(sampling, &mut schema_keep);
        let mut err: *mut c_char = ptr::null_mut();
        let req = unsafe { mlxforge_submit_text(self.handle, cprompt.as_ptr(), &cs, &mut err) };
        if req.is_null() {
            return Err(unsafe { take_string(err) }.unwrap_or_else(|| "submit failed".into()));
        }
        Ok(drain(req))
    }

    /// Run a vision-language completion to completion: a text `prompt` about one
    /// `image` (raw encoded bytes — JPEG/PNG/…). The loaded model must be a
    /// vision-language checkpoint (e.g. Qwen3-VL).
    pub fn image(&self, prompt: &str, image: &[u8], sampling: &Sampling) -> Result<String, String> {
        let cprompt = CString::new(prompt).map_err(|_| "prompt contains NUL".to_string())?;
        let mut schema_keep: Option<CString> = None;
        let cs = Self::c_sampling(sampling, &mut schema_keep);
        let mut err: *mut c_char = ptr::null_mut();
        let req = unsafe {
            mlxforge_submit_image(
                self.handle,
                cprompt.as_ptr(),
                image.as_ptr(),
                image.len(),
                &cs,
                &mut err,
            )
        };
        if req.is_null() {
            return Err(unsafe { take_string(err) }.unwrap_or_else(|| "submit failed".into()));
        }
        Ok(drain(req))
    }

    /// Run a vision-language completion over several images: a text `prompt`
    /// about `images` (each raw encoded bytes), expanded into the prompt in
    /// order. The loaded model must be a vision-language checkpoint (Qwen3-VL).
    pub fn images(&self, prompt: &str, images: &[&[u8]], sampling: &Sampling)
        -> Result<String, String>
    {
        let cprompt = CString::new(prompt).map_err(|_| "prompt contains NUL".to_string())?;
        // Raw pointers into the borrowed slices; valid for the duration of the
        // submit call (the engine copies the bytes synchronously).
        let cimgs: Vec<CImage> = images
            .iter()
            .map(|b| CImage { data: b.as_ptr(), len: b.len() })
            .collect();
        let mut schema_keep: Option<CString> = None;
        let cs = Self::c_sampling(sampling, &mut schema_keep);
        let mut err: *mut c_char = ptr::null_mut();
        let req = unsafe {
            mlxforge_submit_images(
                self.handle,
                cprompt.as_ptr(),
                cimgs.as_ptr(),
                cimgs.len(),
                &cs,
                &mut err,
            )
        };
        if req.is_null() {
            return Err(unsafe { take_string(err) }.unwrap_or_else(|| "submit failed".into()));
        }
        Ok(drain(req))
    }

    /// Embed text into a unit-normalized vector. `pooling`: 0 = mean, 1 = last.
    /// Simple form (no EOS/instruction); for Qwen3-Embedding conventions or to
    /// let the model pick its defaults, use [`Engine::embed_with`].
    pub fn embed(&self, text: &str, pooling: i32) -> Result<Vec<f32>, String> {
        let ctext = CString::new(text).map_err(|_| "text contains NUL".to_string())?;
        let mut out: *mut c_float = ptr::null_mut();
        let mut len: usize = 0;
        let mut err: *mut c_char = ptr::null_mut();
        let rc = unsafe {
            mlxforge_embed(self.handle, ctext.as_ptr(), pooling, &mut out, &mut len, &mut err)
        };
        if rc != 0 || out.is_null() {
            return Err(unsafe { take_string(err) }.unwrap_or_else(|| "embed failed".into()));
        }
        let v = unsafe { std::slice::from_raw_parts(out, len).to_vec() };
        unsafe { mlxforge_floats_free(out) };
        Ok(v)
    }

    /// Embed text with explicit options (Qwen3-Embedding conventions). With a
    /// default [`EmbedOptions`] the model self-selects its convention (a
    /// Qwen3-Embedding checkpoint uses last-token pooling + a trailing EOS).
    pub fn embed_with(&self, text: &str, opts: &EmbedOptions) -> Result<Vec<f32>, String> {
        let ctext = CString::new(text).map_err(|_| "text contains NUL".to_string())?;
        let cinstr = match &opts.instruction {
            Some(s) => Some(CString::new(s.as_str())
                .map_err(|_| "instruction contains NUL".to_string())?),
            None => None,
        };
        let copts = CEmbedOpts {
            pooling: opts.pooling.unwrap_or(-1),
            add_eos: match opts.add_eos {
                None => -1,
                Some(true) => 1,
                Some(false) => 0,
            },
            skip_normalize: if opts.normalize { 0 } else { 1 },
            instruction: cinstr.as_ref().map_or(ptr::null(), |c| c.as_ptr()),
        };
        let mut out: *mut c_float = ptr::null_mut();
        let mut len: usize = 0;
        let mut err: *mut c_char = ptr::null_mut();
        let rc = unsafe {
            mlxforge_embed_ex(self.handle, ctext.as_ptr(), &copts, &mut out, &mut len, &mut err)
        };
        if rc != 0 || out.is_null() {
            return Err(unsafe { take_string(err) }.unwrap_or_else(|| "embed failed".into()));
        }
        let v = unsafe { std::slice::from_raw_parts(out, len).to_vec() };
        unsafe { mlxforge_floats_free(out) };
        Ok(v)
    }
}

/// Options for [`Engine::embed_with`]. A default value defers to the model: a
/// Qwen3-Embedding checkpoint self-selects last-token pooling + a trailing EOS,
/// a plain LLM uses mean pooling.
#[derive(Clone, Debug)]
pub struct EmbedOptions {
    /// `None` = model default; `Some(0)` = mean; `Some(1)` = last token.
    pub pooling: Option<i32>,
    /// `None` = model default; `Some(true)` appends the model's EOS id.
    pub add_eos: Option<bool>,
    /// L2-normalize the pooled vector (default `true`).
    pub normalize: bool,
    /// Optional retrieval instruction; wraps text as "Instruct: {it}\nQuery: {text}".
    pub instruction: Option<String>,
}

impl Default for EmbedOptions {
    fn default() -> Self {
        Self { pooling: None, add_eos: None, normalize: true, instruction: None }
    }
}

impl Drop for Engine {
    fn drop(&mut self) {
        unsafe { mlxforge_engine_free(self.handle) };
    }
}

// Drain a request (UTF-8 chunks) to a String, freeing it.
fn drain(req: *mut mlxforge_request) -> String {
    let mut out = String::new();
    loop {
        let mut text: *mut c_char = ptr::null_mut();
        let rc = unsafe { mlxforge_request_next(req, &mut text) };
        if rc == 0 {
            if let Some(s) = unsafe { take_string(text) } {
                out.push_str(&s);
            }
        } else {
            break;
        }
    }
    let _ = unsafe { mlxforge_request_finish_reason(req) }; // available if needed
    unsafe { mlxforge_request_free(req) };
    out
}
