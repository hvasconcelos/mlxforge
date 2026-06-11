"""Benchmark configuration: embedded defaults + optional TOML overrides.

The full default config lives here as dataclasses so the harness runs with no
config file at all; `--config FILE.toml` merges overrides on top (stdlib
tomllib, so TOML can carry comments without adding a dependency).
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field, replace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

ENGINE_NAMES = ["mlxforge", "llamacpp", "vllm-mlx", "omlx"]
SCENARIO_NAMES = ["concurrency", "prompt", "config", "multiturn"]


@dataclass(frozen=True)
class ModelSpec:
    key: str
    mlx: str  # HF repo id for the MLX engines (mlxforge, vllm-mlx, omlx)
    gguf: str  # llama-server -hf spec ("repo:quant"); approximate quant match
    gguf_path: str = ""  # local .gguf overrides -hf when set
    omlx_name: str = ""  # model name omlx advertises; default = mlx repo basename

    @property
    def omlx_model(self) -> str:
        return self.omlx_name or self.mlx.rsplit("/", 1)[-1]


@dataclass
class ScenarioParams:
    concurrency_levels: list[int] = field(default_factory=lambda: [1, 4, 8, 16])
    concurrency_prompt_tokens: int = 512
    concurrency_max_tokens: int = 128
    prompt_sizes: list[int] = field(default_factory=lambda: [128, 512, 2048, 8192])
    prompt_repeats: int = 3
    prompt_max_tokens: int = 32
    config_sweep_model: str = "llama1b-4bit"
    config_kv_bits: list[int] = field(default_factory=lambda: [0, 8, 4])
    config_probe_concurrency: int = 8
    multiturn_system_tokens: int = 2048
    multiturn_turns: int = 5  # 1 cold + (turns-1) warm
    multiturn_max_tokens: int = 64
    multiturn_passes: int = 2  # fresh system prompt per pass to validate cold numbers

    def quick(self) -> "ScenarioParams":
        # Tiny counts for a smoke run: minutes, not an evaluation.
        return replace(
            self,
            concurrency_levels=[1, 4],
            concurrency_max_tokens=32,
            prompt_sizes=[128, 512],
            prompt_repeats=1,
            config_kv_bits=[0],  # 2 relaunches (prefix on/off) instead of 6
            config_probe_concurrency=4,
            multiturn_system_tokens=512,
            multiturn_turns=3,
            multiturn_max_tokens=16,
            multiturn_passes=1,
        )


@dataclass
class BenchConfig:
    models: dict[str, ModelSpec] = field(default_factory=dict)
    engines: list[str] = field(default_factory=lambda: list(ENGINE_NAMES))
    scenarios: list[str] = field(default_factory=lambda: list(SCENARIO_NAMES))
    model_keys: list[str] = field(default_factory=list)  # empty = all in `models`
    params: ScenarioParams = field(default_factory=ScenarioParams)
    out_dir: Path = REPO_ROOT / "bench" / "results"
    mlxforge_bin: Path = REPO_ROOT / "build" / "mlxforge"
    ready_timeout: float = 300.0  # first launch may include an HF download
    max_ctx: int = 16384
    omlx_model_dir: str = ""  # default: omlx's own default (~/.omlx/models)
    # Per-engine argv template overrides (TOML [engines.<name>] cmd = [...]),
    # formatted with {model} {port} {ctx} {np} {kv_bits} {prefix_cache} {model_dir}.
    engine_cmds: dict[str, list[str]] = field(default_factory=dict)

    def selected_models(self) -> list[ModelSpec]:
        keys = self.model_keys or list(self.models)
        return [self.models[k] for k in keys]


DEFAULT_MODELS = {
    "llama1b-bf16": ModelSpec(
        key="llama1b-bf16",
        mlx="mlx-community/Llama-3.2-1B-Instruct-bf16",
        gguf="bartowski/Llama-3.2-1B-Instruct-GGUF:f16",
    ),
    "llama1b-4bit": ModelSpec(
        key="llama1b-4bit",
        mlx="mlx-community/Llama-3.2-1B-Instruct-4bit",
        # Nearest available quant — MLX 4-bit group quant != Q4_K_M (noted in report).
        gguf="bartowski/Llama-3.2-1B-Instruct-GGUF:Q4_K_M",
    ),
    "qwen3-0.6b": ModelSpec(
        key="qwen3-0.6b",
        mlx="mlx-community/Qwen3-0.6B-bf16",
        gguf="ggml-org/Qwen3-0.6B-GGUF:f16",
    ),
}


def load_config(toml_path: Path | None = None) -> BenchConfig:
    cfg = BenchConfig(models=dict(DEFAULT_MODELS))
    if toml_path is None:
        return cfg
    data = tomllib.loads(Path(toml_path).read_text())
    for key, m in data.get("models", {}).items():
        base = cfg.models.get(key, ModelSpec(key=key, mlx="", gguf=""))
        cfg.models[key] = replace(base, **{f: m[f] for f in ("mlx", "gguf", "gguf_path", "omlx_name") if f in m})
    for name, e in data.get("engines", {}).items():
        if "cmd" in e:
            cfg.engine_cmds[name] = list(e["cmd"])
    for f in ("ready_timeout", "max_ctx", "omlx_model_dir"):
        if f in data:
            setattr(cfg, f, data[f])
    if "mlxforge_bin" in data:
        cfg.mlxforge_bin = Path(data["mlxforge_bin"])
    sc = data.get("scenarios_params", {})
    for f in vars(cfg.params):
        if f in sc:
            setattr(cfg.params, f, sc[f])
    return cfg
