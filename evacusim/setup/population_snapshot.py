"""
Population snapshot: save and restore the initial synthetic population between runs.

Serialises ``agents_config`` (the list of per-agent dicts produced by
AgentManager) to a JSON file so subsequent runs can skip the expensive
spawn-position generation, Shapely zone-inference, and random attribute
sampling that slow down simulation startup.

Saved fields per agent
----------------------
Every key that lives in an ``agent_cfg`` dict at the point agents are added
to JuPedSim, including:

- ``id``, ``name``, ``personality_type``, ``age``, ``gender``,
  ``risk_tolerance``, ``knowledge_profile``
- ``initial_zone``, ``agent_role``, ``target``, ``destination``
- ``goal_state``, ``purpose``, ``purpose_memories``, ``initial_goal``
- ``decision_prompt_extra``, ``is_injured``
- ``start_position`` (tuple[float, float])
- ``level_id`` (str)
- ``walking_speed`` (float)  — stored by AgentManager just before add_agent

Configuration
-------------
Add the following keys to the ``agents`` section of your scenario config:

.. code-block:: yaml

    agents:
      # Save snapshot after generating a fresh population:
      snapshot_save_path: snapshots/my_station_500.json

      # On subsequent runs, load instead of regenerating:
      snapshot_load_path: snapshots/my_station_500.json

Both keys are optional and independent — you can save on one run and load
on another, or do both simultaneously (save = verify round-trip).
Paths are resolved relative to the current working directory.

Snapshot compatibility
----------------------
A snapshot is tied to the scenario config that produced it (agent count,
knowledge profiles, role weights, station geometry, etc.).  If you change
any of those you should regenerate the snapshot.  The format version field
will cause an immediate error if a future format change is incompatible.
"""

import json
from pathlib import Path
from typing import Any

from evacusim.utils.logger import get_logger

logger = get_logger(__name__)

_SNAPSHOT_VERSION = 1


def save_snapshot(agents_config: list[dict[str, Any]], path: str | Path) -> None:
    """Persist *agents_config* to a JSON snapshot file.

    Args:
        agents_config: The list of agent-config dicts produced by AgentManager,
            including ``start_position``, ``level_id``, and ``walking_speed``.
        path: Destination file path.  Parent directories are created if needed.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "version": _SNAPSHOT_VERSION,
        "agent_count": len(agents_config),
        "agents": agents_config,
    }
    path.write_text(json.dumps(payload, indent=2, default=_json_default))
    logger.info(f"Population snapshot saved: {path} ({len(agents_config)} agents)")


def load_snapshot(path: str | Path) -> list[dict[str, Any]]:
    """Load *agents_config* from a previously saved JSON snapshot file.

    Tuple fields (``start_position``) are restored from JSON lists.

    Args:
        path: Path to the snapshot file.

    Returns:
        List of agent-config dicts ready to pass to
        ``AgentManager._add_agents_to_jupedsim_from_snapshot``.

    Raises:
        FileNotFoundError: If *path* does not exist.
        ValueError: If the snapshot format version is unsupported.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Population snapshot not found: {path}")

    payload: dict[str, Any] = json.loads(path.read_text())
    version = payload.get("version", 0)
    if version != _SNAPSHOT_VERSION:
        raise ValueError(
            f"Unsupported population snapshot version {version!r} "
            f"(expected {_SNAPSHOT_VERSION}).  Regenerate the snapshot."
        )

    agents: list[dict[str, Any]] = payload["agents"]

    # JSON stores tuples as lists; restore start_position to tuple[float, float].
    for agent in agents:
        if "start_position" in agent and isinstance(agent["start_position"], list):
            agent["start_position"] = tuple(float(v) for v in agent["start_position"])

    logger.info(
        f"Population snapshot loaded: {path} ({len(agents)} agents, "
        f"format version {version})"
    )
    return agents


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _json_default(obj: Any) -> Any:
    """JSON serialisation fallback for non-standard Python types."""
    if isinstance(obj, tuple):
        return list(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serialisable")
