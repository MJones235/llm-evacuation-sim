"""Runtime spawn controller (Feature A).

Consumes a time-sorted Poisson :class:`~evacusim.calibration.poisson_scheduler.SpawnEvent`
schedule and, as simulation time advances, produces agent-config dicts to be
registered into the running simulation.

The controller owns only the *what/when/where* of spawning — building the cfg
and resolving a jittered spawn position.  Actually inserting the agent into
JuPedSim and the decision pipeline is done by
``HybridSimulationRunner.register_runtime_agent`` (the physics/registration
boundary), keeping this class free of simulation dependencies and unit-testable.

Routing: calibration agents are LLM-free and route via their **goal text**
through the rule-based engine — entrance arrivals get a train-boarding goal
(engine picks ``leave_by_train``); train alighters get a leave-the-station goal
(engine routes to the nearest known exit).  ``dest_exit`` is retained on the cfg
for reporting.
"""

from __future__ import annotations

import math
import random
from typing import Any

from evacusim.calibration.poisson_scheduler import SpawnEvent


class RuntimeSpawnController:
    """Pops due spawn events and builds agent configs for runtime insertion."""

    def __init__(
        self,
        schedule: list[SpawnEvent],
        spawn_points: dict[str, dict[str, Any]],
        seed: int = 0,
        jitter_m: float = 0.5,
        train_door_jitter_m: float = 0.3,
        walking_speed_mean: float = 1.34,
        walking_speed_std: float = 0.0,
        walking_speed_min: float = 0.3,
        walking_speed_max: float = 2.2,
        knowledge_profile: str = "novice",
    ) -> None:
        # Schedule must already be time-sorted (build_arrival_schedule guarantees this).
        self._schedule = list(schedule)
        self._spawn_points = dict(spawn_points)
        self._cursor = 0
        self._counter = 0
        # Independent RNG for position jitter and walking speed; derived from
        # seed so runs are reproducible. Shared between the two uses (rather
        # than a second Random instance) so the whole controller's output is
        # a deterministic function of `seed` alone.
        self._rng = random.Random((seed * 2654435761) & 0xFFFFFFFF)
        self._jitter_m = float(jitter_m)
        self._train_door_jitter_m = float(train_door_jitter_m)
        self._walking_speed_mean = float(walking_speed_mean)
        self._walking_speed_std = float(walking_speed_std)
        self._walking_speed_min = float(walking_speed_min)
        self._walking_speed_max = float(walking_speed_max)
        self._knowledge_profile = knowledge_profile

    def _sample_walking_speed(self) -> float:
        """Per-agent walking speed, normally distributed and clamped.

        `walking_speed_std=0` (the default) collapses this to the fixed
        `walking_speed_mean` for every agent — `random.gauss` with sigma=0
        returns mu deterministically, so no separate code path is needed.
        Clamped to [walking_speed_min, walking_speed_max] since an unclamped
        normal can sample non-physical (near-zero or negative) speeds when
        std is large relative to mean, which JuPedSim can't handle.
        """
        speed = self._rng.gauss(self._walking_speed_mean, self._walking_speed_std)
        return max(self._walking_speed_min, min(self._walking_speed_max, speed))

    def __len__(self) -> int:
        return len(self._schedule)

    @property
    def total(self) -> int:
        return len(self._schedule)

    @property
    def spawned(self) -> int:
        return self._cursor

    @property
    def remaining(self) -> int:
        return len(self._schedule) - self._cursor

    def peek_next_time(self) -> float | None:
        """Scheduled time of the next un-popped spawn event, or ``None`` if drained.

        Lets the runner fast-forward over idle spans (no live agents) straight to
        the next arrival instead of stepping physics dt-by-dt through the gap.
        """
        if self._cursor < len(self._schedule):
            return self._schedule[self._cursor].time_s
        return None

    def discard_before(self, start_time_s: float) -> int:
        """Discard scheduled arrivals strictly before an absolute start time."""
        initial_cursor = self._cursor
        while (
            self._cursor < len(self._schedule)
            and self._schedule[self._cursor].time_s < start_time_s
        ):
            self._cursor += 1
        return self._cursor - initial_cursor

    def pop_due(self, current_sim_time: float) -> list[SpawnEvent]:
        """Return (and consume) all events with ``time_s <= current_sim_time``.

        Monotonic: each event is returned exactly once across the run.
        """
        due: list[SpawnEvent] = []
        n = len(self._schedule)
        while self._cursor < n and self._schedule[self._cursor].time_s <= current_sim_time:
            due.append(self._schedule[self._cursor])
            self._cursor += 1
        return due

    def jittered_position(
        self, event: SpawnEvent, attempt: int = 0
    ) -> tuple[tuple[float, float], str]:
        """Return a jittered ``(position, level)`` for a spawn event.

        A deterministic uniform-disc jitter keeps simultaneous arrivals off the
        same point (JuPedSim rejects coincident insertions).  The disc radius
        grows with ``attempt`` so callers can retry — a burst of alighting
        passengers or a spawn near a wall then spreads out until a valid,
        non-colliding position is found.
        """
        sp = self._spawn_points.get(event.location_id)
        if sp is None:
            raise KeyError(
                f"No spawn_point configured for '{event.location_id}' "
                f"(source={event.source}); known: {sorted(self._spawn_points)}"
            )
        base_xy = sp["xy"]
        door_points = sp.get("door_points", [])
        if event.source == "train" and door_points and event.door_index is not None:
            base_xy = door_points[event.door_index % len(door_points)]
        base_x, base_y = float(base_xy[0]), float(base_xy[1])
        level = str(sp.get("level", event.level))
        base_jitter = (
            self._train_door_jitter_m
            if event.source == "train" and door_points
            else self._jitter_m
        )
        radius = base_jitter * (1.0 + float(attempt))
        r = radius * math.sqrt(self._rng.random())
        theta = self._rng.uniform(0.0, 2.0 * math.pi)
        return (base_x + r * math.cos(theta), base_y + r * math.sin(theta)), level

    def _spawn_position(self, event: SpawnEvent) -> tuple[tuple[float, float], str]:
        return self.jittered_position(event, attempt=0)

    def build_agent_cfg(self, event: SpawnEvent) -> tuple[dict[str, Any], tuple[float, float], str]:
        """Build (agent_cfg, position, level_id) for a due spawn event."""
        (x, y), level = self._spawn_position(event)
        idx = self._counter
        self._counter += 1
        agent_id = f"calib_{event.source}_{idx}"

        if event.source == "entrance":
            goal = "Travel to the platform to board a train."
            initial_zone = "concourse"
        else:  # train alighting
            goal = "Leave the station via the nearest exit."
            initial_zone = "platform"

        cfg: dict[str, Any] = {
            "id": agent_id,
            "name": agent_id,
            "level_id": str(level),
            "start_position": (x, y),
            "initial_zone": initial_zone,
            "agent_role": "passenger",
            "target": event.dest_exit,
            "knowledge_profile": self._knowledge_profile,
            "walking_speed": self._sample_walking_speed(),
            "goal_state": goal,
            "initial_goal": goal,
            "is_injured": False,
            # Provenance for the calibration report.
            "spawn_source": event.source,
            "spawn_location": event.location_id,
            "spawn_time_s": event.time_s,
        }
        return cfg, (x, y), str(level)
