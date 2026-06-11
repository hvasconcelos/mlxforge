"""The four benchmark scenarios.

Each runner takes a ServerCtx (a live, warmed, calibrated server) and returns
a list of row dicts that report.py renders verbatim; raw per-request data
rides along in the JSON for re-analysis.

Prompt hygiene: scenarios 1–2 give every request a unique lead-in so engines
with prompt/prefix caches measure real prefill; scenario 4 does the opposite
on purpose (shared system prompt) to measure those caches.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

import httpx

import loadgen
import prompts
from config import ScenarioParams
from engines import EngineAdapter, EngineProc, ServerOpts


@dataclass
class ServerCtx:
    adapter: EngineAdapter
    ep: EngineProc
    model_name: str  # the "model" field requests must carry
    opts: ServerOpts
    client: httpx.AsyncClient
    extra: dict = field(default_factory=dict)
    # Set right after launch by calibrate(); every scenario depends on it.
    cal: prompts.Calibration = field(default=None)  # type: ignore[assignment]

    def health(self) -> dict | None:
        return self.adapter.health_metrics(self.ep.port)


async def calibrate(ctx: ServerCtx) -> prompts.Calibration:
    w1, w2 = prompts.CALIBRATION_PROBE_WORDS
    t1 = await loadgen.probe_prompt_tokens(
        ctx.client, ctx.ep.base_url, ctx.model_name, prompts.probe_text(w1), ctx.extra)
    t2 = await loadgen.probe_prompt_tokens(
        ctx.client, ctx.ep.base_url, ctx.model_name, prompts.probe_text(w2), ctx.extra)
    return prompts.calibration_from_probes(w1, t1, w2, t2)


async def warmup(ctx: ServerCtx) -> None:
    # Two discarded generations cover Metal JIT / first-graph compilation.
    msgs = [[{"role": "user", "content": prompts.synth_prompt(256, ctx.cal, seed=7 + i)}]
            for i in range(2)]
    await loadgen.run_batch(ctx.client, ctx.ep.base_url, ctx.model_name, msgs,
                            concurrency=1, max_tokens=32, extra=ctx.extra)


def _request_raw(r: loadgen.RequestResult) -> dict:
    return {
        "ok": r.ok, "error": r.error, "ttft_s": r.ttft_s, "latency_s": r.latency_s,
        "client_chunks": r.client_chunks, "server_prompt_tokens": r.server_prompt_tokens,
        "server_completion_tokens": r.server_completion_tokens,
        "finish_reason": r.finish_reason, "quirk": r.quirk,
    }


def _batch_raw(b: loadgen.BatchResult) -> dict:
    return {"wall_s": b.wall_s, "requests": [_request_raw(r) for r in b.results]}


async def run_concurrency(ctx: ServerCtx, p: ScenarioParams, quick: bool) -> list[dict]:
    rows = []
    for i, level in enumerate(p.concurrency_levels):
        total = max(8, 2 * level) if not quick else level
        # Repeat the first level once and keep the second pass: the very first
        # batch after warmup still absorbs residual one-time costs. Each pass
        # gets fresh prompts — reusing them would let engines with prompt
        # caches (llama.cpp cache_prompt) skip the kept pass's prefill.
        passes = 1 if (quick or i > 0) else 2
        for pass_i in range(passes):
            msgs = []
            for j in range(total):
                seed = 1000 * level + j + 500_000 * pass_i
                body = prompts.synth_prompt(p.concurrency_prompt_tokens, ctx.cal, seed=seed)
                msgs.append(
                    [{"role": "user", "content": prompts.unique_prefix(seed, ctx.cal) + body}])
            batch = await loadgen.run_batch(
                ctx.client, ctx.ep.base_url, ctx.model_name, msgs,
                concurrency=level, max_tokens=p.concurrency_max_tokens, extra=ctx.extra)
        errors = len(batch.results) - len(batch.succeeded)
        rows.append({
            "concurrency": level,
            "requests": total,
            "agg_tps": round(batch.aggregate_tps, 1),
            "ttft_p50_ms": _ms(batch.percentile("ttft_s", 0.50)),
            "ttft_p95_ms": _ms(batch.percentile("ttft_s", 0.95)),
            "decode_tps_mean": _r1(batch.mean("decode_tps")),
            "latency_p50_s": _r2(batch.percentile("latency_s", 0.50)),
            "mean_completion_tokens": _r1(batch.mean("completion_tokens")),
            "errors": errors,
            "token_source": batch.token_source,
            "finish_reasons": batch.finish_reasons,
            "_raw": _batch_raw(batch),
        })
    return rows


async def run_prompt_sweep(ctx: ServerCtx, p: ScenarioParams, quick: bool) -> list[dict]:
    rows = []
    for size in p.prompt_sizes:
        ttfts, ptoks, raws = [], [], []
        for rep in range(p.prompt_repeats):
            seed = 10_000 + size + rep
            text = (prompts.unique_prefix(seed, ctx.cal)
                    + prompts.synth_prompt(size, ctx.cal, seed=seed))
            before = ctx.health()
            batch = await loadgen.run_batch(
                ctx.client, ctx.ep.base_url, ctx.model_name,
                [[{"role": "user", "content": text}]],
                concurrency=1, max_tokens=p.prompt_max_tokens, extra=ctx.extra)
            after = ctx.health()
            raws.append(_batch_raw(batch))
            for r in batch.succeeded:
                if r.ttft_s is not None:
                    ttfts.append(r.ttft_s)
                if r.server_prompt_tokens:
                    ptoks.append(r.server_prompt_tokens)
                elif before and after:
                    # mlxforge streams no usage; its /health prompt-token counter
                    # over a single request gives the same number.
                    d = (after["decode"]["prompt_tokens_total"]
                         - before["decode"]["prompt_tokens_total"])
                    if d > 0:
                        ptoks.append(d)
        if not ttfts:
            rows.append({"prompt_tokens_target": size, "error": "all requests failed",
                         "_raw": raws})
            continue
        mean_ttft = sum(ttfts) / len(ttfts)
        achieved = round(sum(ptoks) / len(ptoks)) if ptoks else None
        rows.append({
            "prompt_tokens_target": size,
            "prompt_tokens_achieved": achieved,
            "ttft_mean_ms": _ms(mean_ttft),
            "ttft_min_ms": _ms(min(ttfts)),
            "ttft_max_ms": _ms(max(ttfts)),
            # Prefill rate from achieved tokens when usage exists, else target.
            "prefill_tps": round((achieved or size) / mean_ttft) if mean_ttft > 0 else None,
            "_raw": raws,
        })
    return rows


def make_multiturn_messages(system_tokens: int, cal: prompts.Calibration,
                            pass_seed: int) -> list[dict]:
    system = prompts.synth_prompt(system_tokens, cal, seed=pass_seed)
    return [{"role": "system", "content": "You are a meticulous inventory clerk. " + system}]


async def run_multiturn(ctx: ServerCtx, p: ScenarioParams, quick: bool) -> list[dict]:
    rows = []
    for pass_i in range(p.multiturn_passes):
        base = make_multiturn_messages(p.multiturn_system_tokens, ctx.cal, 77_000 + pass_i)
        history: list[dict] = []
        cold_ttft, warm_ttfts, raws = None, [], []
        for turn in range(p.multiturn_turns):
            messages = base + history + [
                {"role": "user", "content": f"Continue the list (part {turn + 1})."}]
            batch = await loadgen.run_batch(
                ctx.client, ctx.ep.base_url, ctx.model_name, [messages],
                concurrency=1, max_tokens=p.multiturn_max_tokens, extra=ctx.extra)
            raws.append(_batch_raw(batch))
            ok = batch.succeeded
            if not ok or ok[0].ttft_s is None:
                break
            if turn == 0:
                cold_ttft = ok[0].ttft_s
            else:
                warm_ttfts.append(ok[0].ttft_s)
            # Keep history growth bounded; the shared prefix is what matters.
            history += [
                {"role": "user", "content": f"Continue the list (part {turn + 1})."},
                {"role": "assistant", "content": f"(items for part {turn + 1} omitted)"},
            ]
        if cold_ttft is None:
            rows.append({"pass": pass_i + 1, "error": "cold turn failed", "_raw": raws})
            continue
        warm = sum(warm_ttfts) / len(warm_ttfts) if warm_ttfts else None
        rows.append({
            "pass": pass_i + 1,
            "turns": p.multiturn_turns,
            "cold_ttft_ms": _ms(cold_ttft),
            "warm_ttft_mean_ms": _ms(warm),
            "warm_speedup": round(cold_ttft / warm, 2) if warm else None,
            "_raw": raws,
        })
    return rows


async def run_config_probes(ctx: ServerCtx, p: ScenarioParams, quick: bool) -> list[dict]:
    """mlxforge config sweep: one launch of the kv_bits×prefix matrix runs this.

    Cost probe (concurrency batch) + benefit probe (multi-turn), bracketed by
    /health snapshots so server-side counters (decode steps, prefix hits)
    sit beside the client-side numbers.
    """
    before = ctx.health() or {}
    level = p.config_probe_concurrency  # already shrunk in quick mode (see ScenarioParams.quick)
    msgs = []
    for j in range(max(8, level)):
        seed = 50_000 + j
        body = prompts.synth_prompt(p.concurrency_prompt_tokens, ctx.cal, seed=seed)
        msgs.append([{"role": "user", "content": prompts.unique_prefix(seed, ctx.cal) + body}])
    cost = await loadgen.run_batch(
        ctx.client, ctx.ep.base_url, ctx.model_name, msgs,
        concurrency=level, max_tokens=p.concurrency_max_tokens, extra=ctx.extra)
    benefit_rows = await run_multiturn(ctx, p, quick)
    after = ctx.health() or {}

    def delta(*path: str) -> int | None:
        def dig(d: dict):
            for k in path:
                d = d.get(k, {}) if isinstance(d, dict) else {}
            return d if isinstance(d, (int, float)) else None
        b, a = dig(before), dig(after)
        return int(a - b) if a is not None and b is not None else None

    warm = [r.get("warm_ttft_mean_ms") for r in benefit_rows if r.get("warm_ttft_mean_ms")]
    return [{
        "kv_bits": ctx.opts.kv_bits,
        "prefix_cache": "on" if ctx.opts.prefix_cache else "off",
        "agg_tps": round(cost.aggregate_tps, 1),
        "ttft_p50_ms": _ms(cost.percentile("ttft_s", 0.50)),
        "warm_ttft_ms": round(sum(warm) / len(warm), 1) if warm else None,
        "decode_steps": delta("decode", "steps"),
        "peak_batch": (after.get("batch", {}) or {}).get("peak"),
        "prefix_hits": delta("prefix_cache", "hits"),
        "tokens_reused": delta("prefix_cache", "tokens_reused"),
        "srv_avg_tps": round((after.get("decode", {}) or {}).get("avg_tokens_per_second", 0), 1),
        "_raw": {"cost": _batch_raw(cost), "benefit": benefit_rows,
                 "health_before": before, "health_after": after},
    }]


def _ms(v: float | None) -> float | None:
    return round(v * 1000, 1) if v is not None else None


def _r1(v: float | None) -> float | None:
    return round(v, 1) if v is not None else None


def _r2(v: float | None) -> float | None:
    return round(v, 2) if v is not None else None


RUNNERS = {
    "concurrency": run_concurrency,
    "prompt": run_prompt_sweep,
    "multiturn": run_multiturn,
    "config": run_config_probes,
}
