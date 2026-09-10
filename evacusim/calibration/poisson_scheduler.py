"""Deterministic Poisson arrival scheduler for calibration (Feature A).

Turns loaded usage intervals + a train timetable into a single time-sorted list
of :class:`SpawnEvent`s.  Two arrival sources are combined:

* **Entrance stream** — a non-homogeneous Poisson process: within each usage
  interval, inter-arrival times are sampled from an exponential distribution
  with the interval's mean rate (``rate_per_s``).  Sampling is driven by a
  single seeded :class:`random.Random`, so a given ``seed`` always yields the
  same schedule (reproducible calibration runs, no network/LLM).
* **Train alighting** — each :class:`~evacusim.calibration.usage_data.TrainArrival`
  expands into ``alighting`` platform spawn events at the train's arrival time.

The scheduler is pure: it takes data in and returns a list, touching no
simulation state, so it is unit-testable on its own.
"""

from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass(frozen=True)
class SpawnEvent:
    """A single agent to spawn at ``time_s``.

    Attributes:
        time_s: Simulation time (seconds) at which to spawn the agent.
        source: ``"entrance"`` or ``"train"`` — the arrival source.
        location_id: Entrance id (entrance stream) or platform id (train) —
            used to look up the configured spawn point.
        level: Spawn level id (e.g. ``"0"`` concourse, ``"-1"`` platform).
        dest_exit: The agent's journey destination exit id (for reporting; the
            rule-based engine routes via the agent's goal text).
    """

    time_s: float
    source: str
    location_id: str
    level: str
    dest_exit: str


def _sample_entrance_events(intervals, spawn_cfg, rng) -> list[SpawnEvent]:
    entrance_level = str(spawn_cfg["entrance_level"])
    dest_exits = list(spawn_cfg["entrance_dest_exits"])
    events: list[SpawnEvent] = []
    for iv in intervals:
        rate = iv.rate_per_s
        if rate <= 0:
            continue
        t = float(iv.start_s)
        while True:
            t += rng.expovariate(rate)
            if t >= iv.end_s:
                break
            dest = dest_exits[rng.randrange(len(dest_exits))] if dest_exits else ""
            events.append(
                SpawnEvent(
                    time_s=t,
                    source="entrance",
                    location_id=iv.entrance_id,
                    level=entrance_level,
                    dest_exit=dest,
                )
            )
    return events


def _expand_train_events(timetable, spawn_cfg) -> list[SpawnEvent]:
    platform_level = str(spawn_cfg["platform_level"])
    platform_exit = str(spawn_cfg["platform_exit"])
    events: list[SpawnEvent] = []
    for tr in timetable:
        for _ in range(tr.alighting):
            events.append(
                SpawnEvent(
                    time_s=float(tr.arrival_s),
                    source="train",
                    location_id=tr.platform,
                    level=platform_level,
                    dest_exit=platform_exit,
                )
            )
    return events


def build_arrival_schedule(intervals, timetable, spawn_cfg, seed: int = 0) -> list[SpawnEvent]:
    """Build a time-sorted spawn schedule from usage + timetable data.

    Args:
        intervals: List of ``UsageInterval`` (entrance stream rates).
        timetable: List of ``TrainArrival`` (alighting bursts). May be empty.
        spawn_cfg: Dict with keys ``entrance_level``, ``entrance_dest_exits``
            (list), ``platform_level``, ``platform_exit``.
        seed: RNG seed — same seed ⇒ identical schedule.

    Returns:
        Time-sorted list of ``SpawnEvent``. Ties break deterministically on
        ``(time_s, source, location_id)``.
    """
    rng = random.Random(seed)
    events = _sample_entrance_events(intervals, spawn_cfg, rng)
    events += _expand_train_events(timetable, spawn_cfg)
    events.sort(key=lambda e: (e.time_s, e.source, e.location_id))
    return events
