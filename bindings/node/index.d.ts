// Type definitions for @mlxforge/node

export interface EngineOptions {
  /** Max queued requests before submit is rejected (default 256). */
  maxWaiting?: number;
  /**
   * KV-cache quantization bits (engine-wide): 0 (default) keeps the cache
   * dense fp16; 8 is near-lossless at ~1.9x less cache memory; 4 is ~3.6x.
   * Unsupported models (vision-language, hybrid Qwen3.5) fail engine creation
   * rather than silently falling back to fp16.
   */
  kvBits?: 0 | 4 | 8;
  /** Quantization group size (default 64; must divide the model's head_dim). */
  kvGroupSize?: number;
}

export interface SamplingOptions {
  /** <= 0 (default) => greedy/argmax. */
  temperature?: number;
  /** <= 0 (default) => disabled. */
  topK?: number;
  /** <= 0 or >= 1 (default) => disabled. */
  topP?: number;
  /** <= 0 (default) => disabled. */
  minP?: number;
  /** 0/1 (default) => disabled. */
  repetitionPenalty?: number;
  /** 0 (default) => disabled. */
  frequencyPenalty?: number;
  /** 0 (default) => disabled. */
  presencePenalty?: number;
  /** RNG seed (used when temperature > 0). */
  seed?: number;
  /** Max new tokens (default 64). */
  maxTokens?: number;
  /**
   * OpenAI logprobs. 0 (default) => off. N > 0 => report each emitted token's
   * own log-prob plus its (N - 1) most-likely alternatives (so 1 = chosen-only).
   * Retrieve them from the stream's {@link Stream.logprobs} after consumption.
   */
  logprobs?: number;
  /**
   * Constrained decoding. "json" forces any valid JSON value; otherwise a
   * JSON-Schema string (supported subset: a top-level object with ordered,
   * required, scalar-typed properties). Output is masked to be well-formed JSON.
   */
  jsonSchema?: string;
  /** Alias for `jsonSchema` (e.g. "json"). */
  responseFormat?: string;
}

export interface ChatMessage {
  role: 'system' | 'user' | 'assistant' | 'tool';
  content: string;
}

/** One token's log-probability (OpenAI logprobs `content` entry). */
export interface TokenLogprob {
  /** The token's decoded text. */
  token: string;
  /** Natural-log probability of the token (<= 0). */
  logprob: number;
  /** The token's raw UTF-8 bytes. */
  bytes: number[];
  /** The most-likely alternatives at this position (may be empty). */
  top_logprobs: { token: string; logprob: number; bytes: number[] }[];
}

/** An async-iterable stream of decoded text chunks for one request. */
export interface Stream extends AsyncIterable<string> {
  /** Ask the engine to stop generating this request. */
  cancel(): void;
  /** "stop" | "length" | "cancel" | "" (while running). */
  readonly finishReason: string;
  /**
   * Per-token log-probs, available after the stream has been fully consumed.
   * Returns null when `logprobs` was not requested (or none were produced).
   */
  logprobs(): TokenLogprob[] | null;
}

export class Engine {
  constructor(spec: string, opts?: EngineOptions);
  /** Construct and resolve once the model has finished loading. */
  static load(spec: string, opts?: EngineOptions): Promise<Engine>;
  readonly ready: boolean;
  readonly modelName: string;
  waitReady(pollMs?: number): Promise<void>;
  /** Stream a chat completion (runs batched with other concurrent calls). */
  chat(messages: ChatMessage[], sampling?: SamplingOptions): Stream;
  /** Stream a raw-text completion (no chat template). */
  text(prompt: string, sampling?: SamplingOptions): Stream;
  /**
   * Stream a vision-language completion: a text prompt about one image (a
   * Buffer/Uint8Array of raw encoded bytes — JPEG/PNG/…). The loaded model must
   * be a vision-language checkpoint (e.g. Qwen3-VL).
   */
  image(prompt: string, image: Uint8Array, sampling?: SamplingOptions): Stream;
  /**
   * Stream a vision-language completion over several images (raw encoded bytes
   * each), expanded into the prompt in order. The model must be a vision-language
   * checkpoint (e.g. Qwen3-VL).
   */
  images(prompt: string, images: Uint8Array[], sampling?: SamplingOptions): Stream;
  /** Run a chat to completion and return the full string. */
  complete(messages: ChatMessage[], sampling?: SamplingOptions): Promise<string>;
  /**
   * Embed text into a (by default unit-normalized) vector. With no second
   * argument the model self-selects its convention (a Qwen3-Embedding checkpoint
   * uses last-token pooling + a trailing EOS; a plain LLM uses mean pooling).
   * Pass a legacy pooling number (0 = mean, 1 = last) or an options object.
   */
  embed(text: string, opts?: EmbedOptions | number): Promise<Float32Array>;
  dispose(): void;
}

/** Options for {@link Engine.embed}. Omitted fields use the model's defaults. */
export interface EmbedOptions {
  /** 0 = mean, 1 = last token. Omit to use the model's detected default. */
  pooling?: 0 | 1;
  /** Append the model's EOS id before pooling (Qwen3-Embedding last-token). */
  addEos?: boolean;
  /** Wrap as "Instruct: {instruction}\nQuery: {text}" for retrieval queries. */
  instruction?: string;
  /** L2-normalize the pooled vector. Defaults to true. */
  normalize?: boolean;
}

export const version: string;
export const abiVersion: number;
