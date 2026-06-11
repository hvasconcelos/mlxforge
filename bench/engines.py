"""Engine adapters: detect, launch, await readiness, stop.

One adapter per engine. Launch commands are built here but every argv can be
overridden from TOML (`[engines.<name>] cmd = [...]` with {model}/{port}/...
placeholders), so a drifted upstream CLI is a config edit, not a code change.

Servers run one at a time (Metal contention) — the orchestrator enforces that;
this module guarantees a launched process is dead before returning from stop().
"""

from __future__ import annotations

import importlib.metadata
import json
import os
import shutil
import signal
import socket
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

from config import BenchConfig, ModelSpec, REPO_ROOT


@dataclass
class ServerOpts:
    """Per-launch server configuration (the mlxforge config sweep varies these)."""

    label: str = "default"
    kv_bits: int = 0
    prefix_cache: bool = False
    max_concurrency: int = 16  # sizes llama-server's -np slots
    ctx_per_slot: int = 0  # llama-server per-slot context; 0 => cfg.max_ctx

    def as_dict(self) -> dict:
        return {"label": self.label, "kv_bits": self.kv_bits, "prefix_cache": self.prefix_cache,
                "max_concurrency": self.max_concurrency, "ctx_per_slot": self.ctx_per_slot}


@dataclass
class EngineProc:
    proc: subprocess.Popen
    port: int
    log_path: Path

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _http_get_json(url: str, timeout: float = 2.0) -> dict | None:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return json.loads(resp.read())
    except (urllib.error.URLError, OSError, ValueError):
        return None


@dataclass
class EngineAdapter:
    name: str
    cfg: BenchConfig
    skip_note: str = ""
    _version: str | None = field(default=None, repr=False)

    # --- per-engine surface -------------------------------------------------
    def detect(self) -> str | None:
        """Version string when usable, None to skip (set self.skip_note)."""
        raise NotImplementedError

    def launch_cmd(self, model: ModelSpec, port: int, opts: ServerOpts) -> list[str]:
        raise NotImplementedError

    def ready_url(self, port: int) -> str:
        return f"http://127.0.0.1:{port}/v1/models"

    def is_ready(self, body: dict, model: ModelSpec) -> bool:
        return True  # a 200 with parseable JSON is enough by default

    def api_model_name(self, model: ModelSpec) -> str:
        return model.mlx

    def extra_request_fields(self, opts: ServerOpts) -> dict:
        return {}

    def health_metrics(self, port: int) -> dict | None:
        return None  # mlxforge only

    # --- shared mechanics ----------------------------------------------------
    def _format_cmd(self, default: list[str], **subs) -> list[str]:
        template = self.cfg.engine_cmds.get(self.name, default)
        return [a.format(**subs) for a in template]

    def launch(self, model: ModelSpec, opts: ServerOpts, log_dir: Path) -> EngineProc:
        port = free_port()
        cmd = self.launch_cmd(model, port, opts)
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"{self.name}-{model.key}-{opts.label}.log"
        log = open(log_path, "wb")
        log.write((" ".join(cmd) + "\n\n").encode())
        log.flush()
        proc = subprocess.Popen(
            cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
        )
        return EngineProc(proc=proc, port=port, log_path=log_path)

    def wait_ready(self, ep: EngineProc, model: ModelSpec, timeout: float) -> None:
        deadline = time.monotonic() + timeout
        url = self.ready_url(ep.port)
        while time.monotonic() < deadline:
            if ep.proc.poll() is not None:
                raise RuntimeError(
                    f"{self.name} exited with code {ep.proc.returncode} during startup "
                    f"(see {ep.log_path})"
                )
            body = _http_get_json(url)
            if body is not None and self.is_ready(body, model):
                return
            time.sleep(0.5)
        raise TimeoutError(f"{self.name} not ready after {timeout:.0f}s (see {ep.log_path})")

    def stop(self, ep: EngineProc) -> None:
        # SIGINT first (mlxforge drains in-flight requests on it), then escalate.
        # Signals go to the process group: some servers fork helpers.
        for sig, grace in ((signal.SIGINT, 15.0), (signal.SIGTERM, 5.0), (signal.SIGKILL, 5.0)):
            if ep.proc.poll() is not None:
                break
            try:
                os.killpg(os.getpgid(ep.proc.pid), sig)
            except ProcessLookupError:
                break
            try:
                ep.proc.wait(timeout=grace)
                break
            except subprocess.TimeoutExpired:
                continue
        if ep.proc.poll() is None:
            raise RuntimeError(f"{self.name} (pid {ep.proc.pid}) survived SIGKILL")


class MlxforgeAdapter(EngineAdapter):
    def __init__(self, cfg: BenchConfig):
        super().__init__(name="mlxforge", cfg=cfg)

    def detect(self) -> str | None:
        if not self.cfg.mlxforge_bin.exists():
            self.skip_note = f"server binary not found at {self.cfg.mlxforge_bin} (build it first)"
            return None
        out = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "describe", "--tags", "--always", "--dirty"],
            capture_output=True, text=True,
        )
        return out.stdout.strip() or "unknown"

    def launch_cmd(self, model: ModelSpec, port: int, opts: ServerOpts) -> list[str]:
        return self._format_cmd(
            [
                str(self.cfg.mlxforge_bin), "-m", "{model}", "--host", "127.0.0.1",
                "--port", "{port}", "--max-ctx", "{ctx}",
                "--kv-bits", "{kv_bits}", "--prefix-cache", "{prefix_cache}",
            ],
            model=model.mlx, port=port, ctx=self.cfg.max_ctx,
            kv_bits=opts.kv_bits, prefix_cache=int(opts.prefix_cache),
        )

    def ready_url(self, port: int) -> str:
        return f"http://127.0.0.1:{port}/health"

    def is_ready(self, body: dict, model: ModelSpec) -> bool:
        return body.get("status") == "ok"

    def health_metrics(self, port: int) -> dict | None:
        return _http_get_json(f"http://127.0.0.1:{port}/health")


class LlamaCppAdapter(EngineAdapter):
    def __init__(self, cfg: BenchConfig):
        super().__init__(name="llamacpp", cfg=cfg)

    def detect(self) -> str | None:
        if shutil.which("llama-server") is None:
            self.skip_note = "llama-server not on PATH (brew install llama.cpp)"
            return None
        out = subprocess.run(["llama-server", "--version"], capture_output=True, text=True)
        lines = ((out.stderr or "") + (out.stdout or "")).strip().splitlines()
        # Backend-loading chatter precedes the "version: NNNN (sha)" line.
        for line in lines:
            if line.startswith("version"):
                return line.split(":", 1)[1].strip()
        return lines[0] if lines else "unknown"

    def launch_cmd(self, model: ModelSpec, port: int, opts: ServerOpts) -> list[str]:
        np = opts.max_concurrency
        model_args = ["-m", model.gguf_path] if model.gguf_path else ["-hf", model.gguf]
        # -c is the *total* context, split across -np parallel slots — and the
        # whole KV buffer is preallocated, so slots must be right-sized per
        # launch (16 slots × 16k ctx would be tens of GB on small machines).
        slot_ctx = opts.ctx_per_slot or self.cfg.max_ctx
        return self._format_cmd(
            [
                "llama-server", *model_args, "--host", "127.0.0.1", "--port", "{port}",
                "-c", "{total_ctx}", "-np", "{np}", "-ngl", "99", "--no-webui",
            ],
            model=model.gguf, port=port, ctx=self.cfg.max_ctx,
            total_ctx=slot_ctx * np, np=np,
        )

    def ready_url(self, port: int) -> str:
        return f"http://127.0.0.1:{port}/health"

    def api_model_name(self, model: ModelSpec) -> str:
        return model.gguf  # llama-server serves one model; the field is echoed, not matched

    def extra_request_fields(self, opts: ServerOpts) -> dict:
        # cache_prompt is the default in current builds but pinned for safety;
        # ignore_eos keeps fixed-length decodes comparable across engines.
        return {"ignore_eos": True, "cache_prompt": True}


class VllmMlxAdapter(EngineAdapter):
    def __init__(self, cfg: BenchConfig):
        super().__init__(name="vllm-mlx", cfg=cfg)

    def detect(self) -> str | None:
        if shutil.which("vllm-mlx") is None:
            self.skip_note = "vllm-mlx not on PATH (pip install vllm-mlx)"
            return None
        try:
            return importlib.metadata.version("vllm-mlx")
        except importlib.metadata.PackageNotFoundError:
            return "unknown"

    def launch_cmd(self, model: ModelSpec, port: int, opts: ServerOpts) -> list[str]:
        # Assumed invocation per the vllm-mlx README; override via TOML if it drifts.
        return self._format_cmd(
            ["vllm-mlx", "serve", "{model}", "--host", "127.0.0.1", "--port", "{port}",
             "--continuous-batching"],
            model=model.mlx, port=port, ctx=self.cfg.max_ctx,
        )


class OmlxAdapter(EngineAdapter):
    def __init__(self, cfg: BenchConfig):
        super().__init__(name="omlx", cfg=cfg)

    def detect(self) -> str | None:
        if shutil.which("omlx") is None:
            self.skip_note = "omlx not on PATH"
            return None
        return "unknown"

    def launch_cmd(self, model: ModelSpec, port: int, opts: ServerOpts) -> list[str]:
        # omlx discovers models from subdirectories of --model-dir and selects
        # per-request via the "model" field. Assumed invocation; TOML-overridable.
        cmd = ["omlx", "serve", "--host", "127.0.0.1", "--port", "{port}"]
        if self.cfg.omlx_model_dir:
            cmd += ["--model-dir", "{model_dir}"]
        return self._format_cmd(
            cmd, model=model.omlx_model, port=port, model_dir=self.cfg.omlx_model_dir,
        )

    def is_ready(self, body: dict, model: ModelSpec) -> bool:
        # The model list is final once /v1/models answers; a missing model means
        # it isn't in the model-dir, so fail fast instead of timing out.
        ids = [m.get("id", "") for m in body.get("data", [])]
        if not any(model.omlx_model in i for i in ids):
            raise RuntimeError(
                f"omlx is up but '{model.omlx_model}' is not in its model-dir (found: {ids})"
            )
        return True

    def api_model_name(self, model: ModelSpec) -> str:
        return model.omlx_model


def make_adapters(cfg: BenchConfig) -> dict[str, EngineAdapter]:
    adapters = [MlxforgeAdapter(cfg), LlamaCppAdapter(cfg), VllmMlxAdapter(cfg), OmlxAdapter(cfg)]
    return {a.name: a for a in adapters}
