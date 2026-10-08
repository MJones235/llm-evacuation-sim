"""Deterministic Poisson arrival scheduler for calibration (Feature A).

Turns loaded usage intervals + a train timetable into a single time-sorted list
of :class:`SpawnEvent`s.  Two arrival sources are combined:

* **Entrance stream** — a non-homogeneous Poisson process: within each usage
  interval, inter-arrival times are sampled from an exponential distribution
  with the interval's mean rate (``rate_per_s``).  Sampling is driven by a
  single seeded :class:`random.Random`, so a given ``seed`` always yields the
    same schedule (reproducible calibration runs, no network/LLM). Destinations
    are limited to platforms whose timetable service window includes the sampled
    arrival time, with a ten-minute allowance before the first train so early
    passengers can arrive and wait without accumulating for hours.
* **Train alighting** — each :class:`~evacusim.calibration.usage_data.TrainArrival`
    expands into ``alighting`` platform spawn events distributed across configured
    doors and staggered over the first seconds of the dwell.

The scheduler is pure: it takes data in and returns a list, touching no
simulation state, so it is unit-testable on its own.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from evacusim.conventions import train_exit

PLATFORM_OPEN_LEAD_S = 10 * 60
DEFAULT_ALIGHTING_DURATION_S = 12.0
ALIGHTING_START_DELAY_S = 1.0


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
        door_index: Zero-based train door used for an alighting spawn, or None
            for entrance arrivals and legacy schedules without door metadata.
    """

    time_s: float
    source: str
    location_id: str
    level: str
    dest_exit: str
    door_index: int | None = None


def _train_service_windows(timetable) -> dict[str, tuple[float, float]]:
    """Return train-exit service windows, opening ten minutes before service."""
    windows: dict[str, tuple[float, float]] = {}
    for train in timetable:
        exit_id = train_exit(train.platform)
        start, end = windows.get(exit_id, (float("inf"), float("-inf")))
        windows[exit_id] = (
            min(start, max(0.0, float(train.arrival_s) - PLATFORM_OPEN_LEAD_S)),
            max(end, float(train.arrival_s + train.dwell_s)),
        )
    return windows


def _sample_entrance_events(intervals, spawn_cfg, rng, service_windows=None) -> list[SpawnEvent]:
    entrance_level = str(spawn_cfg["entrance_level"])
    dest_exits = list(spawn_cfg["entrance_dest_exits"])
    service_windows = service_windows or {}
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
            available_dest_exits = [
                exit_id
                for exit_id in dest_exits
                if exit_id not in service_windows
                or service_windows[exit_id][0] <= t <= service_windows[exit_id][1]
            ]
            choices = available_dest_exits or dest_exits
            dest = choices[rng.randrange(len(choices))] if choices else ""
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


def _expand_train_events(timetable, spawn_cfg, rng) -> list[SpawnEvent]:
    platform_level = str(spawn_cfg["platform_level"])
    platform_exit = str(spawn_cfg["platform_exit"])
    door_counts = spawn_cfg.get("train_door_counts", {})
    configured_duration = float(
        spawn_cfg.get("train_alighting_duration_s", DEFAULT_ALIGHTING_DURATION_S)
    )
    events: list[SpawnEvent] = []
    for tr in timetable:
        door_count = max(1, int(door_counts.get(str(tr.platform), 1)))
        duration = max(0.0, min(configured_duration, float(tr.dwell_s) - 0.5))
        start_delay = min(ALIGHTING_START_DELAY_S, duration)
        for passenger_index in range(tr.alighting):
            events.append(
                SpawnEvent(
                    time_s=float(tr.arrival_s) + rng.uniform(start_delay, duration),
                    source="train",
                    location_id=tr.platform,
                    level=platform_level,
                    dest_exit=platform_exit,
                    door_index=passenger_index % door_count,
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
    service_windows = _train_service_windows(timetable)
    events = _sample_entrance_events(intervals, spawn_cfg, rng, service_windows)
    train_rng = random.Random(seed ^ 0xA11E17)
    events += _expand_train_events(timetable, spawn_cfg, train_rng)
    events.sort(key=lambda e: (e.time_s, e.source, e.location_id))
    return events
