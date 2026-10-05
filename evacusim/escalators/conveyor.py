"""Two-lane escalator conveyor kinematics — pure Python, no JuPedSim.

Positions are measured along the incline: ``s = 0`` at the boarding comb,
``s = length_m`` at the far comb.

Rules (per lane, per escalator):

* **Stand lane (right).** A stander occupies one step and is carried at belt
  speed. Standers board onto the next free step; with probability
  ``stander_step_gap_prob`` they leave one step free in front of them (most
  people do not stand on consecutive steps). Max boarding rate is therefore
  ``belt_speed / step_depth`` (1.25/s at 0.5 m/s and 0.4 m steps).
* **Walk lane (left).** A walker moves at belt speed plus their own climbing
  pace, keeps at least one free step to the walker ahead (``s_ahead - 2d``),
  never overtakes, and never moves slower than the belt.
* **Belt stopped** (``running = False``): the escalator is a staircase. Every
  rider walks at their own pace, keeping the same spacing rules.
* **Paused** (the far landing is full): nobody moves and nobody boards — the
  model's equivalent of an emergency stop. The caller resumes it once every
  rider waiting at the far comb has been placed.

Nothing on a conveyor can block anything else on it, so a conveyor can never
gridlock; congestion only ever forms on the floor at the boarding comb.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

LANES = ("stand", "walk")


@dataclass(frozen=True)
class ConveyorParams:
    length_m: float
    belt_speed: float = 0.5
    step_depth: float = 0.4
    stander_step_gap_prob: float = 0.5

    def __post_init__(self) -> None:
        if self.length_m <= 0:
            raise ValueError("length_m must be > 0")
        if self.belt_speed < 0:
            raise ValueError("belt_speed must be >= 0")
        if self.step_depth <= 0:
            raise ValueError("step_depth must be > 0")
        if not 0.0 <= self.stander_step_gap_prob <= 1.0:
            raise ValueError("stander_step_gap_prob must be in [0, 1]")


@dataclass
class Rider:
    agent_id: str
    lane: str
    # Own pace along the incline (m/s). Walkers add it to the belt; standers use
    # it only when the belt is stopped and the escalator becomes a staircase.
    walk_speed: float
    board_time: float
    s: float = 0.0
    # Seconds spent at the far comb waiting for room on the landing.
    stall_wait_s: float = 0.0
    discharge_attempts: int = 0


@dataclass
class Conveyor:
    params: ConveyorParams
    rng: random.Random = field(default_factory=random.Random)
    running: bool = True
    paused: bool = False
    closed: bool = False

    def __post_init__(self) -> None:
        # Front (highest s) first, i.e. boarding order.
        self.riders: dict[str, list[Rider]] = {lane: [] for lane in LANES}
        # Clearance the most recent stander needs before the next can board.
        self._stand_gap = self._sample_stand_gap()
        self.paused_s = 0.0

    # ------------------------------------------------------------------
    # Boarding
    # ------------------------------------------------------------------

    def _sample_stand_gap(self) -> float:
        d = self.params.step_depth
        return 2 * d if self.rng.random() < self.params.stander_step_gap_prob else d

    def required_gap(self, lane: str) -> float:
        if lane == "walk":
            return 2 * self.params.step_depth
        return self._stand_gap

    def can_admit(self, lane: str) -> bool:
        """True if a passenger may step onto ``lane`` right now."""
        if lane not in self.riders:
            raise ValueError(f"unknown lane {lane!r}")
        if self.closed or self.paused:
            return False
        lane_riders = self.riders[lane]
        if not lane_riders:
            return True
        return lane_riders[-1].s >= self.required_gap(lane) - 1e-9

    def board(self, agent_id: str, lane: str, walk_speed: float, time_s: float) -> Rider:
        """Put a passenger on the bottom step of ``lane``. Caller checks ``can_admit``."""
        rider = Rider(agent_id=agent_id, lane=lane, walk_speed=max(0.0, walk_speed),
                      board_time=time_s)
        self.riders[lane].append(rider)
        if lane == "stand":
            self._stand_gap = self._sample_stand_gap()
        return rider

    # ------------------------------------------------------------------
    # Motion
    # ------------------------------------------------------------------

    def step(self, dt: float) -> list[Rider]:
        """Advance every rider by ``dt``; return riders now at the far comb."""
        if self.paused:
            self.paused_s += dt
            for lane_riders in self.riders.values():
                for rider in lane_riders:
                    if rider.s >= self.params.length_m:
                        rider.stall_wait_s += dt
            return self.arrived()

        length = self.params.length_m
        belt = self.params.belt_speed if self.running else 0.0
        d = self.params.step_depth
        for lane, lane_riders in self.riders.items():
            ahead_s = None
            for rider in lane_riders:
                if lane == "stand" and self.running:
                    # Standers ride their step; spacing was fixed at boarding.
                    new_s = rider.s + belt * dt
                else:
                    gap = 2 * d if lane == "walk" else d
                    new_s = rider.s + (belt + rider.walk_speed) * dt
                    if ahead_s is not None:
                        new_s = min(new_s, ahead_s - gap)
                    # Never slower than the belt (the step carries you).
                    new_s = max(new_s, rider.s + belt * dt)
                rider.s = min(new_s, length)
                ahead_s = rider.s
        return self.arrived()

    def arrived(self) -> list[Rider]:
        length = self.params.length_m
        return [r for lane_riders in self.riders.values() for r in lane_riders
                if r.s >= length - 1e-9]

    def remove(self, agent_id: str) -> Rider | None:
        for lane_riders in self.riders.values():
            for i, rider in enumerate(lane_riders):
                if rider.agent_id == agent_id:
                    return lane_riders.pop(i)
        return None

    # ------------------------------------------------------------------
    # Control
    # ------------------------------------------------------------------

    def pause(self) -> None:
        self.paused = True

    def resume(self) -> None:
        self.paused = False

    def close(self) -> None:
        """Stop boarding; riders already on finish their ride."""
        self.closed = True

    def open(self) -> None:
        self.closed = False

    def stop_belt(self) -> None:
        """Turn the escalator off — it becomes a staircase."""
        self.running = False

    def start_belt(self) -> None:
        self.running = True

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def rider_count(self) -> int:
        return sum(len(v) for v in self.riders.values())

    def has_rider(self, agent_id: str) -> bool:
        return any(r.agent_id == agent_id for v in self.riders.values() for r in v)

    def snapshot(self) -> list[list]:
        """``[[agent_id, lane, s_m], …]`` for recording/visualisation."""
        return [[r.agent_id, lane, round(r.s, 3)]
                for lane, lane_riders in self.riders.items() for r in lane_riders]

    def max_boarding_rate(self, lane: str, walk_speed: float = 0.0) -> float:
        """Theoretical ceiling (people/s) for ``lane`` at this belt speed."""
        d = self.params.step_depth
        if lane == "stand":
            return self.params.belt_speed / d
        return (self.params.belt_speed + walk_speed) / (2 * d)
