"""The run manifest: what produced a run, and how it ended.

Every run directory holds a ``manifest.json``, written when the run starts
(``status: running``) and completed when it ends (``ok`` or ``failed``), so
that any result can be traced back to exactly what produced it::

    {
      "run_id", "status" (running/ok/failed/interrupted), "error",
      "started", "finished", "wall_time_s",
      "config_path", "command",
      "engine", "seed",
      "code":     {"evacusim": {commit, branch, dirty}, "study": {...}},
      "versions": {"python", "evacusim", "jupedsim", "pydantic"},
      "parameters": {...every resolved parameter (RunConfig)...},
      "results":  {"steps", "sim_time_s", "decisions_made", "events_triggered"},
      "llm":      {"total_requests", "total_tokens", "estimated_cost_gbp", ...}
    }

A ``dirty: true`` repository had uncommitted changes when the run started, so
its commit alone does not reproduce the run.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

import evacusim
from evacusim.config.schema import RunConfig


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def git_state(path: Path) -> dict[str, Any] | None:
    """Commit, branch and dirtiness of the git repository containing ``path``."""

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(path), *args], capture_output=True, text=True, check=True
        ).stdout.strip()

    try:
        return {
            "commit": git("rev-parse", "HEAD"),
            "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
        }
    except (OSError, subprocess.CalledProcessError):
        return None


def _version(package: str) -> str | None:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return None


def start_manifest(
    run_dir: Path, params: RunConfig, *, config_path: str | Path, study_root: str | Path
) -> dict[str, Any]:
    """Write ``manifest.json`` for a starting run and return it."""
    manifest = {
        "run_id": run_dir.name,
        "status": "running",
        "error": None,
        "started": _now(),
        "finished": None,
        "wall_time_s": None,
        "config_path": str(config_path),
        "command": sys.argv,
        "engine": params.decision.engine,
        "seed": params.seed,
        "code": {
            "evacusim": git_state(Path(evacusim.__file__).parent),
            "study": git_state(Path(study_root)),
        },
        "versions": {
            "python": platform.python_version(),
            "evacusim": _version("evacusim"),
            "jupedsim": _version("jupedsim"),
            "pydantic": _version("pydantic"),
        },
        "parameters": params.model_dump(mode="json"),
        "results": None,
        "llm": None,
    }
    write_manifest(run_dir, manifest)
    return manifest


def finish_manifest(
    run_dir: Path,
    manifest: dict[str, Any],
    *,
    error: BaseException | None = None,
    results: dict[str, Any] | None = None,
    llm_provider: Any = None,
) -> None:
    """Record how the run ended and rewrite ``manifest.json``.

    Status: ``failed`` on an error, ``interrupted`` if stopped early by the user
    (Ctrl-C), else ``ok``.
    """
    interrupted = isinstance(error, KeyboardInterrupt) or bool(
        results and results.get("interrupted")
    )
    if interrupted:
        manifest["status"] = "interrupted"
    else:
        manifest["status"] = "failed" if error is not None else "ok"
    manifest["error"] = f"{type(error).__name__}: {error}" if error is not None else None
    manifest["finished"] = _now()
    started = datetime.fromisoformat(manifest["started"])
    manifest["wall_time_s"] = round(
        (datetime.fromisoformat(manifest["finished"]) - started).total_seconds(), 1
    )
    if results is not None:
        manifest["results"] = {
            "steps": results.get("steps"),
            "sim_time_s": results.get("sim_time"),
            "decisions_made": results.get("decisions_made"),
            "events_triggered": results.get("events_triggered"),
        }
    if llm_provider is not None and hasattr(llm_provider, "get_usage_stats"):
        manifest["llm"] = llm_provider.get_usage_stats()
    write_manifest(run_dir, manifest)


def write_manifest(run_dir: Path, manifest: dict[str, Any]) -> None:
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
