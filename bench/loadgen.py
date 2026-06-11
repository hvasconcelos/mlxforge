"""Async load generation over OpenAI-compatible /v1/chat/completions SSE.

Token accounting is deliberately dual: `client_chunks` (content deltas seen on
the wire) is always recorded, `server_*_tokens` comes from `usage` when the
engine reports it (llama.cpp/vLLM honor stream_options.include_usage; mlxforge
streams no usage chunk). Aggregates prefer the server count and fall back to
chunks; the report labels which one each cell used.
"""

from __future__ import annotations

import asyncio
import json
import statistics
import time
from dataclasses import dataclass, field

import httpx


@dataclass
class RequestResult:
    ok: bool = False
    error: str = ""
    start: float = 0.0
    first_token_at: float | None = None
    end: float = 0.0
    client_chunks: int = 0
    server_prompt_tokens: int | None = None
    server_completion_tokens: int | None = None
    finish_reason: str = ""
    saw_done: bool = False  # [DONE] sentinel reached
    quirk: str = ""  # transport error after logical completion (tolerated)

    @property
    def ttft_s(self) -> float | None:
        return (self.first_token_at - self.start) if self.first_token_at else None

    @property
    def latency_s(self) -> float:
        return self.end - self.start

    @property
    def completion_tokens(self) -> int:
        return self.server_completion_tokens or self.client_chunks

    @property
    def token_source(self) -> str:
        return "server" if self.server_completion_tokens else "client"

    @property
    def decode_tps(self) -> float | None:
        # First token excluded: it belongs to prefill (TTFT), not decode.
        if not self.first_token_at or self.completion_tokens < 2:
            return None
        dt = self.end - self.first_token_at
        return (self.completion_tokens - 1) / dt if dt > 0 else None


def sampling_fields(max_tokens: int) -> dict:
    return {
        "temperature": 0,
        "top_p": 1,
        "seed": 1234,
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
    }


async def stream_chat(
    client: httpx.AsyncClient,
    base_url: str,
    model: str,
    messages: list[dict],
    max_tokens: int,
    extra: dict | None = None,
) -> RequestResult:
    body = {"model": model, "messages": messages, **sampling_fields(max_tokens), **(extra or {})}
    r = RequestResult(start=time.monotonic())
    try:
        async with client.stream("POST", f"{base_url}/v1/chat/completions", json=body) as resp:
            if resp.status_code != 200:
                detail = (await resp.aread())[:200]
                r.error = f"HTTP {resp.status_code}: {detail.decode(errors='replace')}"
                r.end = time.monotonic()
                return r
            buf = b""
            async for chunk in resp.aiter_bytes():
                buf += chunk
                # SSE events are blank-line separated; engines vary on \r\n.
                while (sep := _event_end(buf)) is not None:
                    event, buf = buf[: sep[0]], buf[sep[1]:]
                    _consume_event(event, r)
        r.ok = not r.error
    except (httpx.HTTPError, OSError) as e:
        # A transport error after the stream logically completed (we saw [DONE]
        # or a finish_reason) is an engine quirk worth noting, not a failed
        # request — e.g. closing the connection without the chunked terminator.
        if r.saw_done or r.finish_reason:
            r.ok = True
            r.quirk = f"{type(e).__name__} after stream end: {e}"
        else:
            r.error = f"{type(e).__name__}: {e}"
    r.end = time.monotonic()
    return r


def _event_end(buf: bytes) -> tuple[int, int] | None:
    for sep in (b"\r\n\r\n", b"\n\n"):
        i = buf.find(sep)
        if i != -1:
            return (i, i + len(sep))
    return None


def _consume_event(event: bytes, r: RequestResult) -> None:
    for line in event.splitlines():
        if not line.startswith(b"data:"):
            continue
        data = line[5:].strip()
        if data == b"[DONE]":
            r.saw_done = True
            return
        try:
            obj = json.loads(data)
        except ValueError:
            continue
        usage = obj.get("usage")
        if usage:  # usage-only final chunks have empty `choices`
            r.server_prompt_tokens = usage.get("prompt_tokens", r.server_prompt_tokens)
            r.server_completion_tokens = usage.get("completion_tokens", r.server_completion_tokens)
        for choice in obj.get("choices", []):
            if choice.get("finish_reason"):
                r.finish_reason = choice["finish_reason"]
            delta = choice.get("delta", {})
            # reasoning_content counts too: thinking models (Qwen3) emit it as
            # separate deltas on some engines, and those are generated tokens.
            if delta.get("content") or delta.get("reasoning_content"):
                r.client_chunks += 1
                if r.first_token_at is None:
                    r.first_token_at = time.monotonic()


@dataclass
class BatchResult:
    results: list[RequestResult] = field(default_factory=list)
    wall_s: float = 0.0

    @property
    def succeeded(self) -> list[RequestResult]:
        return [r for r in self.results if r.ok]

    @property
    def total_completion_tokens(self) -> int:
        return sum(r.completion_tokens for r in self.succeeded)

    @property
    def aggregate_tps(self) -> float:
        return self.total_completion_tokens / self.wall_s if self.wall_s > 0 else 0.0

    def percentile(self, attr: str, q: float) -> float | None:
        vals = sorted(v for r in self.succeeded if (v := getattr(r, attr)) is not None)
        if not vals:
            return None
        return vals[min(len(vals) - 1, int(q * len(vals)))]

    def mean(self, attr: str) -> float | None:
        vals = [v for r in self.succeeded if (v := getattr(r, attr)) is not None]
        return statistics.fmean(vals) if vals else None

    @property
    def token_source(self) -> str:
        sources = {r.token_source for r in self.succeeded}
        return sources.pop() if len(sources) == 1 else "mixed"

    @property
    def finish_reasons(self) -> str:
        counts: dict[str, int] = {}
        for r in self.succeeded:
            reason = r.finish_reason or "?"
            counts[reason] = counts.get(reason, 0) + 1
        return ",".join(f"{k}:{v}" for k, v in sorted(counts.items()))


async def run_batch(
    client: httpx.AsyncClient,
    base_url: str,
    model: str,
    message_lists: list[list[dict]],
    concurrency: int,
    max_tokens: int,
    extra: dict | None = None,
) -> BatchResult:
    sem = asyncio.Semaphore(concurrency)

    async def one(messages: list[dict]) -> RequestResult:
        async with sem:
            return await stream_chat(client, base_url, model, messages, max_tokens, extra)

    t0 = time.monotonic()
    results = await asyncio.gather(*(one(m) for m in message_lists))
    return BatchResult(results=list(results), wall_s=time.monotonic() - t0)


def make_client() -> httpx.AsyncClient:
    # Long read timeout: an 8k-token prefill on a small GPU can take a while.
    return httpx.AsyncClient(
        timeout=httpx.Timeout(connect=5.0, read=300.0, write=30.0, pool=300.0),
        limits=httpx.Limits(max_connections=64),
    )


async def probe_prompt_tokens(
    client: httpx.AsyncClient, base_url: str, model: str, text: str, extra: dict | None = None
) -> int:
    """Non-streaming 1-token request; every engine reports usage here."""
    body = {
        "model": model,
        "messages": [{"role": "user", "content": text}],
        "max_tokens": 1,
        "temperature": 0,
        **(extra or {}),
    }
    resp = await client.post(f"{base_url}/v1/chat/completions", json=body)
    resp.raise_for_status()
    return int(resp.json()["usage"]["prompt_tokens"])
