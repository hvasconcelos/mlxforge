# /// script
# requires-python = ">=3.11"
# dependencies = ["httpx"]
# ///
"""Cross-engine inference benchmark harness.

Drives mlxforge, llama.cpp (llama-server), vllm-mlx, and omlx through their
OpenAI-compatible HTTP APIs — strictly one engine at a time (they all contend
for the same Metal GPU) — and renders a text report plus raw JSON results.

    uv run bench/bench.py --quick --engines mlxforge --models qwen3-0.6b
    uv run bench/bench.py                      # full run, all detected engines
    uv run bench/bench.py --rerender results/bench-<ts>.json

See bench/README.md for engine install hints and launch-command assumptions.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import report
import scenarios as sc
from config import ENGINE_NAMES, SCENARIO_NAMES, BenchConfig, ModelSpec, load_config
from engines import EngineAdapter, ServerOpts, make_adapters
from loadgen import make_client

# Scenarios each engine can run; "config" is the mlxforge-internal sweep.
CROSS_ENGINE_SCENARIOS = ["concurrency", "prompt", "multiturn"]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--engines", default=",".join(ENGINE_NAMES),
                    help=f"comma list of {ENGINE_NAMES} (missing ones auto-skip)")
    ap.add_argument("--scenarios", default=",".join(SCENARIO_NAMES),
                    help=f"comma list of {SCENARIO_NAMES}")
    ap.add_argument("--models", default="",
                    help="comma list of model keys (default: all configured)")
    ap.add_argument("--config", type=Path, default=None, help="optional TOML overrides")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--quick", action="store_true", help="tiny counts; smoke run in minutes")
    ap.add_argument("--mlxforge-bin", type=Path, default=None)
    ap.add_argument("--ready-timeout", type=float, default=None,
                    help="server readiness timeout in seconds (first run may download models)")
    ap.add_argument("--rerender", type=Path, default=None,
                    help="re-render the text report from a saved results JSON and exit")
    ap.add_argument("--keep-logs", action="store_true",
                    help="keep per-engine server logs even when every run succeeds")
    return ap.parse_args()


def plan_launches(engine: str, model: ModelSpec, scenario_names: list[str],
                  cfg: BenchConfig, config_model: str) -> list[tuple[ServerOpts, list[str]]]:
    """Map (engine, model, scenarios) to server launches: most engines need one
    launch for everything; mlxforge relaunches per config (prefix cache and the
    kv_bits sweep are engine-creation settings, not request settings)."""
    max_conc = max(cfg.params.concurrency_levels + [cfg.params.config_probe_concurrency])
    cross = [s for s in scenario_names if s in CROSS_ENGINE_SCENARIOS]
    if engine == "llamacpp":
        # llama-server preallocates -c × -np of KV, so each launch is sized to
        # its scenario: many small slots for the concurrency sweep, one
        # full-context slot for the prompt sweep and multi-turn.
        launches = []
        if "concurrency" in cross:
            need = cfg.params.concurrency_prompt_tokens + cfg.params.concurrency_max_tokens + 384
            slot = -(-need // 1024) * 1024  # round up to 1k
            launches.append((ServerOpts(label="batch", max_concurrency=max_conc,
                                        ctx_per_slot=slot), ["concurrency"]))
        rest = [s for s in cross if s != "concurrency"]
        if rest:
            launches.append((ServerOpts(max_concurrency=1), rest))
        return launches
    if engine != "mlxforge":
        return [(ServerOpts(max_concurrency=max_conc), cross)] if cross else []

    launches: list[tuple[ServerOpts, list[str]]] = []
    plain = [s for s in cross if s != "multiturn"]
    if plain:
        launches.append((ServerOpts(max_concurrency=max_conc), plain))
    if "multiturn" in cross:
        # The shared-prefix scenario is what the prefix cache exists for.
        launches.append((ServerOpts(label="prefix", prefix_cache=True,
                                    max_concurrency=max_conc), ["multiturn"]))
    if "config" in scenario_names and model.key == config_model:
        for bits in cfg.params.config_kv_bits:
            for prefix in (False, True):
                launches.append((ServerOpts(label=f"kv{bits}-{'pfx' if prefix else 'nopfx'}",
                                            kv_bits=bits, prefix_cache=prefix,
                                            max_concurrency=max_conc), ["config"]))
    return launches


async def run_launch(adapter: EngineAdapter, model: ModelSpec, opts: ServerOpts,
                     scenario_names: list[str], cfg: BenchConfig, quick: bool,
                     log_dir: Path) -> list[dict]:
    runs: list[dict] = []
    ep = adapter.launch(model, opts, log_dir)
    try:
        adapter.wait_ready(ep, model, cfg.ready_timeout)
        async with make_client() as client:
            ctx = sc.ServerCtx(
                adapter=adapter, ep=ep, model_name=adapter.api_model_name(model),
                opts=opts, client=client, extra=adapter.extra_request_fields(opts))
            ctx.cal = await sc.calibrate(ctx)
            print(f"    calibrated: {ctx.cal.tokens_per_word:.2f} tok/word, "
                  f"overhead {ctx.cal.overhead_tokens:.0f} tok", flush=True)
            await sc.warmup(ctx)
            for name in scenario_names:
                print(f"    scenario: {name}", flush=True)
                rows = await sc.RUNNERS[name](ctx, cfg.params, quick)
                runs.append({"scenario": name, "engine": adapter.name, "model": model.key,
                             "server_config": opts.as_dict(), "params": vars(cfg.params),
                             "rows": rows})
    except Exception as e:
        note = f"{type(e).__name__}: {e}"
        print(f"    FAILED: {note}", file=sys.stderr, flush=True)
        runs.append({"scenario": "/".join(scenario_names), "engine": adapter.name,
                     "model": model.key, "server_config": opts.as_dict(),
                     "rows": [], "note": note})
    finally:
        adapter.stop(ep)
    return runs


async def main() -> int:
    args = parse_args()
    if args.rerender:
        results = json.loads(args.rerender.read_text())
        html_path = args.rerender.with_suffix(".html")
        html_path.write_text(report.render_html(results))
        print(report.render(results), end="")
        print(f"\nreport: {html_path}")
        return 0

    cfg = load_config(args.config)
    if args.mlxforge_bin:
        cfg.mlxforge_bin = args.mlxforge_bin
    if args.ready_timeout is not None:
        cfg.ready_timeout = args.ready_timeout
    if args.out_dir:
        cfg.out_dir = args.out_dir
    if args.models:
        keys = args.models.split(",")
        unknown = [m for m in keys if m not in cfg.models]
        if unknown:
            print(f"unknown model keys: {unknown} (known: {list(cfg.models)})", file=sys.stderr)
            return 2
        cfg.model_keys = keys
    if args.quick:
        cfg.params = cfg.params.quick()

    engine_names = [e.strip() for e in args.engines.split(",") if e.strip()]
    scenario_names = [s.strip() for s in args.scenarios.split(",") if s.strip()]
    for name, known in ((engine_names, ENGINE_NAMES), (scenario_names, SCENARIO_NAMES)):
        bad = [n for n in name if n not in known]
        if bad:
            print(f"unknown name(s): {bad} (known: {known})", file=sys.stderr)
            return 2

    models = cfg.selected_models()
    config_model = (cfg.params.config_sweep_model
                    if any(m.key == cfg.params.config_sweep_model for m in models)
                    else models[0].key)

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = cfg.out_dir / f"logs-{ts}"
    adapters = make_adapters(cfg)
    engines_info: dict[str, dict] = {}
    runs: list[dict] = []
    for name in engine_names:
        adapter = adapters[name]
        version = adapter.detect()
        engines_info[name] = {"detected": version is not None, "version": version,
                              "note": adapter.skip_note or None}
        if version is None:
            print(f"== {name}: SKIPPED ({adapter.skip_note})", flush=True)
            continue
        print(f"== {name} ({version})", flush=True)
        for model in models:
            for opts, names in plan_launches(name, model, scenario_names, cfg, config_model):
                print(f"  {model.key} [{opts.label}] -> {','.join(names)}", flush=True)
                runs += await run_launch(adapter, model, opts, names, cfg, args.quick, log_dir)

    results = {
        "meta": {"timestamp": datetime.now().isoformat(timespec="seconds"),
                 "host": report.host_info(), "quick": args.quick,
                 "argv": sys.argv[1:]},
        "engines": engines_info,
        "runs": runs,
    }
    html_path, json_path = report.save(results, cfg.out_dir)
    print()
    print(report.render(results), end="")
    print(f"\nreport: {html_path}\nraw:    {json_path}")

    had_failures = any(r.get("note") for r in runs)
    if log_dir.exists() and not (args.keep_logs or had_failures):
        shutil.rmtree(log_dir)
    elif log_dir.exists():
        print(f"logs:   {log_dir}")
    return 1 if had_failures else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
