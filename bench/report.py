"""Report rendering (HTML file + text for stdout) and raw JSON persistence.

The JSON is the source of truth (`--rerender` round-trips it); the renders are
views: one table per scenario per model, engines × levels as rows, plus an
environment header and a Notes section for skips and comparability caveats.
The saved report is a self-contained HTML file (inline CSS, no assets); the
text render is what goes to stdout.
"""

from __future__ import annotations

import html
import json
import subprocess
from datetime import datetime
from pathlib import Path

# Columns are rendered in this order when present; anything else (minus _raw)
# is appended alphabetically so new scenario fields show up without edits here.
_PREFERRED_ORDER = [
    "concurrency", "requests", "prompt_tokens_target", "prompt_tokens_achieved",
    "pass", "turns", "kv_bits", "prefix_cache",
    "agg_tps", "prefill_tps", "decode_tps_mean",
    "ttft_p50_ms", "ttft_p95_ms", "ttft_mean_ms", "ttft_min_ms", "ttft_max_ms",
    "cold_ttft_ms", "warm_ttft_mean_ms", "warm_ttft_ms", "warm_speedup",
    "latency_p50_s", "mean_completion_tokens",
    "decode_steps", "peak_batch", "prefix_hits", "tokens_reused", "srv_avg_tps",
    "errors", "token_source", "finish_reasons", "error",
]

# Cross-row comparison direction, for best-value highlighting in the HTML.
_HIGHER_BETTER = {"agg_tps", "prefill_tps", "decode_tps_mean", "srv_avg_tps", "warm_speedup"}
_LOWER_BETTER = {"ttft_p50_ms", "ttft_p95_ms", "ttft_mean_ms", "cold_ttft_ms",
                 "warm_ttft_mean_ms", "warm_ttft_ms", "latency_p50_s"}

_SCENARIO_TITLES = {
    "concurrency": "Concurrency sweep",
    "prompt": "Prompt-length sweep (single stream)",
    "config": "mlxforge config sweep (kv_bits × prefix_cache)",
    "multiturn": "Multi-turn shared-prefix reuse",
}


def _sysctl(key: str) -> str:
    out = subprocess.run(["sysctl", "-n", key], capture_output=True, text=True)
    return out.stdout.strip()


def host_info() -> dict:
    sw = subprocess.run(["sw_vers", "-productVersion"], capture_output=True, text=True)
    mem_bytes = int(_sysctl("hw.memsize") or 0)
    return {
        "model": _sysctl("hw.model"),
        "chip": _sysctl("machdep.cpu.brand_string"),
        "ram_gb": round(mem_bytes / 2**30),
        "macos": sw.stdout.strip(),
    }


def _fmt_cell(v) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:g}"
    return str(v)


# --- shared table assembly ----------------------------------------------------

def _table_columns(rows: list[dict], lead_cols: list[str]) -> list[str]:
    keys: list[str] = list(lead_cols)
    seen = set(keys)
    for col in _PREFERRED_ORDER:
        if col not in seen and any(col in r for r in rows):
            keys.append(col)
            seen.add(col)
    for r in rows:
        for col in sorted(r):
            if col not in seen and not col.startswith("_"):
                keys.append(col)
                seen.add(col)
    return keys


def _grouped_tables(results: dict) -> list[tuple[str, str, str, list[dict]]]:
    """(scenario, model, subtitle, rows) per table, in scenario order; each row
    carries an `engine` lead column derived from the run's server-config label."""
    groups: dict[tuple[str, str], list[dict]] = {}
    for run in results["runs"]:
        if run.get("rows"):
            groups.setdefault((run["scenario"], run["model"]), []).append(run)
    tables = []
    for scenario in _SCENARIO_TITLES:
        for (sc, model), group in groups.items():
            if sc != scenario:
                continue
            rows = []
            for run in group:
                label = run["server_config"].get("label", "default")
                engine = run["engine"] if label == "default" else f"{run['engine']}[{label}]"
                for row in run["rows"]:
                    rows.append({"engine": engine,
                                 **{k: v for k, v in row.items() if not k.startswith("_")}})
            tables.append((scenario, model, _scenario_subtitle(sc, group[0].get("params", {})), rows))
    return tables


def _notes(results: dict) -> list[str]:
    notes = [f"{name} skipped: {info['note']}"
             for name, info in results["engines"].items() if not info.get("detected")]
    notes += [f"{r['engine']}/{r['model']}: {r['note']}" for r in results["runs"] if r.get("note")]
    notes += [
        "token_source: 'server' = engine-reported usage; 'client' = SSE content-chunk count.",
        "GGUF quants (e.g. Q4_K_M) are nearest equivalents to MLX group quants, not identical.",
    ]
    return notes


def _engines_line(results: dict) -> str:
    return " · ".join(
        f"{name} {info['version']}" if info.get("detected") else f"{name} SKIPPED"
        for name, info in results["engines"].items())


def _scenario_subtitle(scenario: str, p: dict) -> str:
    if scenario == "concurrency":
        return (f"prompt≈{p.get('concurrency_prompt_tokens')} tok"
                f" · max_tokens={p.get('concurrency_max_tokens')}")
    if scenario == "prompt":
        return f"max_tokens={p.get('prompt_max_tokens')}"
    if scenario == "multiturn":
        return (f"system≈{p.get('multiturn_system_tokens')} tok"
                f" · {p.get('multiturn_turns')} turns")
    return ""


# --- text render (stdout) ------------------------------------------------------

def format_table(rows: list[dict], lead_cols: list[str]) -> str:
    keys = _table_columns(rows, lead_cols)
    table = [keys] + [[_fmt_cell(r.get(k)) for k in keys] for r in rows]
    widths = [max(len(row[i]) for row in table) for i in range(len(keys))]
    lines = ["  ".join(c.ljust(w) for c, w in zip(row, widths)).rstrip() for row in table]
    lines.insert(1, "  ".join("-" * w for w in widths))
    return "\n".join(lines)


def render(results: dict) -> str:
    meta = results["meta"]
    h = meta.get("host", {})
    out = [
        f"mlxforge benchmark harness — {meta.get('timestamp', '')}",
        f"Host: {h.get('model', '?')} · {h.get('chip', '?')} · {h.get('ram_gb', '?')} GB · "
        f"macOS {h.get('macos', '?')}",
        "Engines: " + _engines_line(results),
        "",
    ]
    for scenario, model, subtitle, rows in _grouped_tables(results):
        sub = f" · {subtitle}" if subtitle else ""
        out.append(f"== {_SCENARIO_TITLES[scenario]} · {model}{sub} ==")
        out.append(format_table(rows, lead_cols=["engine"]))
        out.append("")
    out.append("Notes:")
    out += [f"- {n}" for n in _notes(results)]
    return "\n".join(out) + "\n"


# --- HTML render (the report file) ---------------------------------------------

_CSS = """
:root { color-scheme: light dark; }
body { font: 15px/1.5 -apple-system, "Helvetica Neue", sans-serif; margin: 2rem auto;
       max-width: 72rem; padding: 0 1rem; }
h1 { font-size: 1.4rem; margin-bottom: .2rem; }
h2 { font-size: 1.1rem; margin: 2rem 0 .2rem; }
.sub { color: #777; font-size: .85rem; margin: 0 0 .6rem; }
table { border-collapse: collapse; width: 100%; font-size: .85rem; }
th, td { padding: .35rem .6rem; text-align: right; white-space: nowrap; }
th:first-child, td:first-child { text-align: left; }
th { border-bottom: 2px solid #888; font-weight: 600; }
td { border-bottom: 1px solid rgba(136,136,136,.25);
     font-variant-numeric: tabular-nums; }
tr:hover td { background: rgba(136,136,136,.08); }
td.best { font-weight: 700; color: #0a7d33; }
@media (prefers-color-scheme: dark) { td.best { color: #4cc36e; } }
ul.notes { color: #777; font-size: .85rem; }
"""


# Rows are only comparable at the same sweep point (same concurrency level,
# prompt size, ...); highlighting is computed per group of equal key value.
# The config sweep has no group key: every row is a config, all comparable.
_GROUP_KEY = {"concurrency": "concurrency", "prompt": "prompt_tokens_target",
              "multiturn": "pass"}


def _best_values(rows: list[dict], keys: list[str], scenario: str) -> dict[tuple, float]:
    group_key = _GROUP_KEY.get(scenario)
    groups: dict[object, list[dict]] = {}
    for r in rows:
        groups.setdefault(r.get(group_key) if group_key else None, []).append(r)
    best: dict[tuple, float] = {}
    for gval, grows in groups.items():
        if len(grows) < 2:
            continue
        for k in keys:
            vals = [r[k] for r in grows if isinstance(r.get(k), (int, float))]
            if len(vals) < 2 or min(vals) == max(vals):
                continue
            if k in _HIGHER_BETTER:
                best[(gval, k)] = max(vals)
            elif k in _LOWER_BETTER:
                best[(gval, k)] = min(vals)
    return best


def render_html(results: dict) -> str:
    meta = results["meta"]
    h = meta.get("host", {})
    e = html.escape
    out = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        "<title>mlxforge benchmark report</title>",
        f"<style>{_CSS}</style></head><body>",
        f"<h1>mlxforge benchmark report</h1>",
        f"<p class='sub'>{e(meta.get('timestamp', ''))}"
        f"{' · quick run' if meta.get('quick') else ''}<br>"
        f"Host: {e(h.get('model', '?'))} · {e(h.get('chip', '?'))} · "
        f"{e(str(h.get('ram_gb', '?')))} GB · macOS {e(h.get('macos', '?'))}<br>"
        f"Engines: {e(_engines_line(results))}</p>",
    ]
    for scenario, model, subtitle, rows in _grouped_tables(results):
        keys = _table_columns(rows, ["engine"])
        best = _best_values(rows, keys, scenario)
        group_key = _GROUP_KEY.get(scenario)
        out.append(f"<h2>{e(_SCENARIO_TITLES[scenario])} · {e(model)}</h2>")
        if subtitle:
            out.append(f"<p class='sub'>{e(subtitle)}</p>")
        out.append("<table><tr>" + "".join(f"<th>{e(k)}</th>" for k in keys) + "</tr>")
        for r in rows:
            gval = r.get(group_key) if group_key else None
            cells = []
            for k in keys:
                v = r.get(k)
                cls = " class='best'" if v is not None and best.get((gval, k)) == v else ""
                cells.append(f"<td{cls}>{e(_fmt_cell(v))}</td>")
            out.append("<tr>" + "".join(cells) + "</tr>")
        out.append("</table>")
    out.append("<h2>Notes</h2><ul class='notes'>")
    out += [f"<li>{e(n)}</li>" for n in _notes(results)]
    out.append("</ul></body></html>")
    return "\n".join(out) + "\n"


def save(results: dict, out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    json_path = out_dir / f"bench-{ts}.json"
    html_path = out_dir / f"bench-{ts}.html"
    json_path.write_text(json.dumps(results, indent=2, default=str))
    html_path.write_text(render_html(results))
    return html_path, json_path
