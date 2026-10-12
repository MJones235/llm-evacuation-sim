"""Freeing people wedged in place.

JuPedSim's collision-free speed model has local deadlocks: two people
heading opposite ways in a corner can press against each other and the wall
indefinitely, where real people would sidestep. The watchdog notices someone
who has somewhere to go but has not made progress for a while; the runner
then sends them a step aside, and they decide again a moment later.

Not stuck: people who have arrived (within ``arrive_m`` of their target),
who are waiting, and who are queueing for an escalator (a queue may stand
still for a long time).
"""

from __future__ import annotations

import math
from collections.abc import Iterable

Point2 = tuple[float, float]

STUCK_AFTER_S = 45.0
"""No progress for this long is a deadlock, not congestion."""
MIN_PROGRESS_M = 0.25
ARRIVE_M = 1.5
SIDESTEP_M = (1.0, 1.5, 2.0)
REDECIDE_AFTER_S = 3.0
"""Time to take the step aside before deciding again."""


class ProgressWatchdog:
    """Notices people who have not moved towards their target for ``stuck_after_s``."""

    def __init__(
        self,
        stuck_after_s: float = STUCK_AFTER_S,
        min_progress_m: float = MIN_PROGRESS_M,
        arrive_m: float = ARRIVE_M,
    ):
        self.stuck_after_s = stuck_after_s
        self.min_progress_m = min_progress_m
        self.arrive_m = arrive_m
        self._anchor: dict[str, tuple[Point2, float]] = {}
        self.freed: list[dict] = []
        """Every intervention: time, agent and position (for the log and tests)."""

    def stuck(
        self,
        time_s: float,
        positions: dict[str, Point2],
        targets: dict[str, Point2 | None],
        exempt: Iterable[str] = (),
    ) -> list[str]:
        """People stuck now, in id order. ``targets[a]`` is ``None`` when unknown.

        Only people in ``targets`` are watched (the runner passes those who
        are moving); a ``None`` target counts as somewhere to go.
        """
        exempt = set(exempt)
        for gone in set(self._anchor) - set(positions):
            del self._anchor[gone]
        out = []
        for agent_id in sorted(positions):
            pos = positions[agent_id]
            anchor = self._anchor.get(agent_id)
            if (
                agent_id not in targets
                or agent_id in exempt
                or anchor is None
                or math.dist(pos, anchor[0]) >= self.min_progress_m
            ):
                self._anchor[agent_id] = (pos, time_s)
                continue
            target = targets[agent_id]
            arrived = target is not None and math.dist(pos, target) <= self.arrive_m
            if arrived:
                self._anchor[agent_id] = (pos, time_s)
            elif time_s - anchor[1] >= self.stuck_after_s:
                self._anchor[agent_id] = (pos, time_s)  # give the remedy time to work
                self.freed.append({"time_s": time_s, "agent_id": agent_id, "position": pos})
                out.append(agent_id)
        return out


def sidestep_candidates(pos: Point2, target: Point2 | None) -> list[Point2]:
    """Points to step aside to, best first: sideways to the way they face, then any."""
    if target is not None and math.dist(pos, target) > 1e-6:
        heading = math.atan2(target[1] - pos[1], target[0] - pos[0])
    else:
        heading = 0.0
    turns = [math.pi / 2, -math.pi / 2, 3 * math.pi / 4, -3 * math.pi / 4, math.pi, 0.0]
    return [
        (pos[0] + r * math.cos(heading + turn), pos[1] + r * math.sin(heading + turn))
        for r in SIDESTEP_M
        for turn in turns
    ]
