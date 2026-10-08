"""EscalatorSystem — couples floor simulations (JuPedSim, per level) to conveyors.

Lifecycle of a passenger using an escalator:

1. **Assign** — the decision layer picks an escalator exit
   (``MultiLevelJuPedSimulation.set_agent_destination_exit``). The passenger is
   given a lane (walk with probability ``walk_share``) and a queue slot on the
   boarding landing.
2. **Queue** — every waiting passenger targets their *own* waypoint slot, so no
   two people ever converge on one point (the cause of the old model's
   gridlock). The earliest arrivals in each lane stand in that lane's short
   line directly in front of the comb; later arrivals take overflow slots
   spread over the landing, nearest-first. An apron in front of the comb and
   every discharge landing are kept clear of waiting people.
3. **Admit** — when the conveyor has room in a lane, the head of that lane's
   line is switched to the full-width boarding strip at the comb (one admitted
   person walking in per lane at a time).
4. **Board** — JuPedSim removes them at the strip; they are put on the
   conveyor and leave the floor simulation entirely.
5. **Ride** — ``Conveyor`` kinematics (see ``conveyor.py``).
6. **Discharge** — at the far comb they are placed on the other level's landing
   (their lane's spot first, then anywhere on a short strip across the comb),
   with their own walking speed, heading for the escalator's egress target. If
   the landing is full the belt pauses until there is room; nobody is ever
   sent back.

A spike on the real geometry (2026-10-05) showed JuPedSim's own
``NotifiableQueueStage`` gridlocks here (waiting agents mill in front of the
released one), while this slot scheme cleared 250 agents with no stall.
"""

from __future__ import annotations

import contextlib
import math
import random
import zlib
from dataclasses import dataclass
from itertools import pairwise
from typing import TYPE_CHECKING, Any

import jupedsim as jps
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union

from evacusim.escalators.conveyor import LANES, Conveyor, ConveyorParams
from evacusim.escalators.spec_loader import EscalatorSpec
from evacusim.utils.logger import get_logger

if TYPE_CHECKING:
    from evacusim.jps.multi_level_simulation import MultiLevelJuPedSimulation

logger = get_logger(__name__)

Point2 = tuple[float, float]

DEFAULT_TIME_GAP = 1.0  # JuPedSim CollisionFreeSpeedModel default
MIN_AGENT_SEP = 0.45  # JuPedSim rejects centres closer than ~0.4 m
SLOT_CLEARANCE = 0.25
MAX_OVERFLOW_SLOTS = 300
STRIP_NEAR = 0.05  # boarding strip extends 0.05–0.45 m out from the comb
STRIP_FAR = 0.45
ADMIT_RADIUS = 1.0  # lane head must be this close to the strip to be admitted
ADMITTED_CLEAR = 0.35  # previous admission counts as "boarding" until this close
AT_COMB = 0.3  # a waiting agent this close to the strip simply steps on
ADMIT_TIMEOUT_S = 20.0  # admitted but still not on after this: back into the queue


@dataclass
class QueueEntry:
    lane: str
    assigned_s: float
    join_s: float | None = None
    slot: Point2 | None = None
    admitted: bool = False
    admitted_s: float | None = None
    tight: bool = False


@dataclass
class _Entry:
    """Boarding-side geometry and state for one escalator."""

    spec: EscalatorSpec
    conveyor: Conveyor
    strip: Polygon
    strip_target: Point2
    stage_id: int
    journey_id: int
    landing_point: Point2
    line_slots: dict[str, list[Point2]]
    apron: Polygon
    join_radius: float
    overflow_slots: list[Point2]
    discharge_candidates: dict[str, list[Point2]]
    discharge_area: Polygon
    egress: Point2
    egress_points: list[Point2]
    egress_next: int = 0


class EscalatorSystem:
    def __init__(
        self,
        ml: MultiLevelJuPedSimulation,
        specs: list[EscalatorSpec],
        seed: int = 0,
        closed_exits: set[str] | None = None,
    ):
        self.ml = ml
        self.seed = int(seed)
        self.escalators: dict[str, _Entry] = {}
        self.queue: dict[str, dict[str, QueueEntry]] = {}
        self.queued_on: dict[str, str] = {}  # agent -> exit_name (waiting/admitted)
        self.riding: dict[str, str] = {}  # agent -> exit_name (on a conveyor)
        self.boarding_hold: dict[str, list[tuple[str, str, float]]] = {}
        self.admitted_walking: dict[str, dict[str, str | None]] = {}
        # Last time each escalator admitted or boarded someone (stall diagnostics).
        self._last_activity: dict[str, float] = {}
        self._last_stall_warning: dict[str, float] = {}
        self.ride_log: list[dict[str, Any]] = []
        # Agents walking to a closed escalator they do not yet know is closed;
        # they find out (and re-decide) on reaching its landing.
        self.discovering: dict[str, str] = {}
        self._open_rides: dict[str, dict[str, Any]] = {}
        self._waypoint_cache: dict[tuple[str, Point2], tuple[int, int]] = {}

        for i, spec in enumerate(specs):
            if spec.from_level not in ml.simulations or spec.to_level not in ml.simulations:
                logger.warning(f"Escalator {spec.exit_name}: level not loaded, skipped")
                continue
            offered = spec.exit_name not in (closed_exits or ())
            self.escalators[spec.exit_name] = self._build_entry(spec, i, offered)
            self.queue[spec.exit_name] = {}
            self.boarding_hold[spec.exit_name] = []
            self.admitted_walking[spec.exit_name] = {lane: None for lane in LANES}
        self._build_overflow_slots()

        for exit_name in closed_exits or ():
            if exit_name in self.escalators:
                self.close(exit_name)

        for level_sim in ml.simulations.values():
            level_sim.route_listener = self._on_external_route

        logger.info(
            "Escalator conveyors: "
            + ", ".join(
                f"{e.spec.exit_name} ({e.spec.direction}, {e.spec.length_m:.1f} m, "
                f"{e.spec.belt_speed} m/s, {len(e.overflow_slots)} queue slots)"
                for e in self.escalators.values()
            )
        )

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def _floor(self, level: str):
        return self.ml.simulations[level].geometry_manager._combined_geometry

    @staticmethod
    def _offset_band(spec_comb, near: float, far: float) -> Polygon:
        (ax, ay), (bx, by) = spec_comb.a, spec_comb.b
        nx, ny = spec_comb.floor_normal
        return Polygon(
            [
                (ax + nx * near, ay + ny * near),
                (bx + nx * near, by + ny * near),
                (bx + nx * far, by + ny * far),
                (ax + nx * far, ay + ny * far),
            ]
        )

    def _build_entry(self, spec: EscalatorSpec, index: int, offered: bool = True) -> _Entry:
        level_sim = self.ml.simulations[spec.from_level]
        floor = self._floor(spec.from_level)
        clear = floor.buffer(-SLOT_CLEARANCE)

        strip = self._offset_band(spec.entry, STRIP_NEAR, STRIP_FAR).intersection(floor).convex_hull
        if strip.is_empty or strip.area < 0.1:
            raise ValueError(f"Escalator {spec.exit_name}: boarding strip is not on the floor")
        stage_id = level_sim.simulation.add_exit_stage(list(strip.exterior.coords)[:-1])
        journey_id = level_sim.simulation.add_journey(jps.JourneyDescription([stage_id]))
        mx, my = spec.entry.mid
        nx, ny = spec.entry.floor_normal
        landing_point = (mx + nx * 1.0, my + ny * 1.0)
        if offered:
            # Pre-blocked escalators are never offered as exits (as before);
            # their landing position is recorded by GeometryManager instead.
            level_sim.exit_manager.register_exit(
                spec.exit_name, stage_id, journey_id, landing_point
            )

        heading = spec.boarding_heading
        line_slots: dict[str, list[Point2]] = {}
        for lane in LANES:
            pts: list[Point2] = []
            for i in range(spec.queue_line_slots):
                p = spec.entry.lane_point(lane, 0.7 + spec.queue_line_spacing * i, heading)
                if not clear.contains(Point(p)):
                    break
                pts.append(p)
            line_slots[lane] = pts
        line_len = max((len(v) for v in line_slots.values()), default=0)
        apron_depth = 1.0 + spec.queue_line_spacing * line_len
        apron = self._offset_band(spec.entry, 0.0, apron_depth).buffer(0.3)

        # Discharge landing on the other level.
        to_floor = self._floor(spec.to_level).buffer(-0.2)
        depth = spec.landing_search_depth
        discharge_area = self._offset_band(spec.exit, 0.35, 0.35 + depth).intersection(to_floor)
        heading_out = spec.alighting_heading
        candidates: dict[str, list[Point2]] = {}
        grid: list[Point2] = []
        if not discharge_area.is_empty:
            x0, y0, x1, y1 = discharge_area.bounds
            gx = x0
            while gx <= x1:
                gy = y0
                while gy <= y1:
                    if discharge_area.contains(Point(gx, gy)):
                        grid.append((gx, gy))
                    gy += MIN_AGENT_SEP
                gx += MIN_AGENT_SEP
        for lane in LANES:
            first = spec.exit.lane_point(lane, 0.6, heading_out)
            first_ok = [first] if to_floor.contains(Point(first)) else []
            candidates[lane] = first_ok + sorted(grid, key=lambda p: math.dist(p, first))
            if not candidates[lane]:
                raise ValueError(f"Escalator {spec.exit_name}: no free floor at the exit landing")

        # Walk-off targets: spread over the floor 3-8 m out from the exit comb
        # and handed out in turn, so people stepping off fan out instead of
        # converging on one point and backing up onto the landing. (The
        # geometry's egress points are at the comb itself.)
        ex, ey = spec.exit.mid
        enx, eny = spec.exit.floor_normal
        to_router = self.ml.simulations[spec.to_level]._routing_engine
        landing = (ex + enx * 0.6, ey + eny * 0.6)
        egress_points: list[Point2] = []
        for out in (3.0, 4.0, 5.0, 6.0, 7.0, 8.0):
            for lateral in (0.0, -0.8, 0.8, -1.6, 1.6, -2.4, 2.4):
                q = (ex + enx * out + eny * lateral, ey + eny * out - enx * lateral)
                if to_floor.contains(Point(q)):
                    d = _route_length(to_router, landing, q)
                    if d is not None and d <= out + 3.0:
                        egress_points.append(q)
        egress = (
            egress_points[0]
            if egress_points
            else (spec.egress_target or spec.exit.lane_point("stand", 0.6, heading_out))
        )
        if not egress_points:
            egress_points = [egress]

        conveyor = Conveyor(
            ConveyorParams(
                spec.length_m, spec.belt_speed, spec.step_depth, spec.stander_step_gap_prob
            ),
            rng=random.Random(self.seed * 1000 + index),
        )
        return _Entry(
            spec=spec,
            conveyor=conveyor,
            strip=strip,
            strip_target=(strip.centroid.x, strip.centroid.y),
            stage_id=stage_id,
            journey_id=journey_id,
            landing_point=landing_point,
            line_slots=line_slots,
            apron=apron,
            join_radius=apron_depth + 1.5,
            overflow_slots=[],
            discharge_candidates=candidates,
            discharge_area=self._offset_band(spec.exit, 0.0, 0.35 + depth + 0.6).buffer(0.3),
            egress=egress,
            egress_points=egress_points,
        )

    def _build_overflow_slots(self) -> None:
        """Overflow queue slots per escalator, avoiding every apron and discharge landing."""
        for level in self.ml.simulations:
            keep_clear = unary_union(
                [e.apron for e in self.escalators.values() if e.spec.from_level == level]
                + [e.discharge_area for e in self.escalators.values() if e.spec.to_level == level]
            )
            clear = self._floor(level).buffer(-SLOT_CLEARANCE)
            router = self.ml.simulations[level]._routing_engine
            for entry in self.escalators.values():
                if entry.spec.from_level != level:
                    continue
                spec = entry.spec
                origin = Point(entry.landing_point)
                grid = spec.queue_slot_grid
                x0, y0, x1, y1 = origin.buffer(spec.queue_max_route_m).bounds
                scored: list[tuple[float, Point2]] = []
                gx = x0
                while gx <= x1:
                    gy = y0
                    while gy <= y1:
                        p = Point(gx, gy)
                        if (
                            clear.contains(p)
                            and not keep_clear.contains(p)
                            and origin.distance(p) <= spec.queue_max_route_m
                        ):
                            d = _route_length(router, (gx, gy), entry.landing_point)
                            if d is not None and d <= spec.queue_max_route_m:
                                scored.append((d, (round(gx, 3), round(gy, 3))))
                        gy += grid
                    gx += grid
                scored.sort()
                entry.overflow_slots = [p for _, p in scored[:MAX_OVERFLOW_SLOTS]]
                if not entry.overflow_slots:
                    logger.warning(f"Escalator {spec.exit_name}: no overflow queue slots found")

    # ------------------------------------------------------------------
    # Public API used by MultiLevelJuPedSimulation / decision layer
    # ------------------------------------------------------------------

    def handles(self, exit_name: str) -> bool:
        return exit_name in self.escalators

    def exits_from_level(self, level: str) -> list[str]:
        return [n for n, e in self.escalators.items() if e.spec.from_level == level]

    def is_in_transit(self, agent_id: str) -> bool:
        return agent_id in self.riding or any(
            agent_id == held[0] for hold in self.boarding_hold.values() for held in hold
        )

    def is_riding(self, agent_id: str) -> bool:
        return self.is_in_transit(agent_id)

    def is_committed(self, agent_id: str) -> bool:
        """Queueing at the landing (joined) or admitted to step on."""
        exit_name = self.queued_on.get(agent_id)
        if exit_name is None:
            return False
        qe = self.queue[exit_name].get(agent_id)
        return qe is not None and (qe.admitted or qe.join_s is not None)

    def queued_exit(self, agent_id: str) -> str | None:
        return self.queued_on.get(agent_id)

    def load(self, exit_name: str) -> int:
        """People committed to this escalator: queueing, boarding and riding."""
        entry = self.escalators.get(exit_name)
        if entry is None:
            return 0
        return (
            len(self.queue[exit_name])
            + entry.conveyor.rider_count()
            + len(self.boarding_hold[exit_name])
        )

    def lane_for(self, agent_id: str, exit_name: str) -> str:
        spec = self.escalators[exit_name].spec
        u = random.Random(zlib.crc32(f"{self.seed}:{agent_id}:{exit_name}".encode())).random()
        return "walk" if u < spec.walk_share else "stand"

    def assign(self, agent_id: str, exit_name: str, time_s: float) -> bool:
        """Commit a floor agent to an escalator's queue. Returns False if refused."""
        entry = self.escalators.get(exit_name)
        if entry is None:
            return False
        if self.ml.agent_levels.get(agent_id) != entry.spec.from_level:
            logger.error(
                f"[ESCALATOR] {agent_id} cannot use {exit_name}: not on level "
                f"{entry.spec.from_level}"
            )
            return False
        current = self.queued_on.get(agent_id)
        if entry.conveyor.closed:
            # Nobody is told remotely: they walk over and see it is closed.
            if current is not None:
                self.unassign(agent_id)
            if self.discovering.get(agent_id) != exit_name:
                self.discovering[agent_id] = exit_name
                self._route_to_point(entry.spec.from_level, agent_id, entry.landing_point)
                logger.info(f"[ESCALATOR] {agent_id} heading for closed {exit_name}")
            return False
        if current == exit_name:
            return True
        if current is not None:
            self.unassign(agent_id)
        self.queue[exit_name][agent_id] = QueueEntry(
            lane=self.lane_for(agent_id, exit_name), assigned_s=time_s
        )
        self.queued_on[agent_id] = exit_name
        self._assign_slots(entry.spec.from_level)
        return True

    def unassign(self, agent_id: str) -> None:
        exit_name = self.queued_on.pop(agent_id, None)
        if exit_name is None:
            return
        q = self.queue[exit_name].pop(agent_id, None)
        level_sim = self.ml.simulations[self.escalators[exit_name].spec.from_level]
        if q is not None and q.tight:
            self._set_time_gap(level_sim, agent_id, DEFAULT_TIME_GAP)
        if level_sim.agent_assigned_exits.get(agent_id) == exit_name:
            level_sim.agent_assigned_exits.pop(agent_id, None)
        walking = self.admitted_walking[exit_name]
        for lane, aid in walking.items():
            if aid == agent_id:
                walking[lane] = None

    def close(self, exit_name: str) -> list[str]:
        """Stop boarding; release everyone queueing. Returns released agent ids."""
        entry = self.escalators.get(exit_name)
        if entry is None:
            return []
        entry.conveyor.close()
        released = list(self.queue[exit_name])
        level_sim = self.ml.simulations[entry.spec.from_level]
        for agent_id in released:
            self.unassign(agent_id)
            pos = level_sim.get_agent_position(agent_id)
            if pos is not None:
                self._hold_in_place(level_sim, agent_id, pos)
            self.ml.agents_needing_redecision.add(agent_id)
        logger.info(f"[ESCALATOR] {exit_name} closed; released {len(released)} queueing agents")
        return released

    def open(self, exit_name: str) -> None:
        entry = self.escalators.get(exit_name)
        if entry is not None:
            entry.conveyor.open()

    def board(self, agent_id: str, exit_name: str, time_s: float) -> None:
        """Called when JuPedSim removed ``agent_id`` at ``exit_name``'s boarding strip."""
        entry = self.escalators[exit_name]
        q = self.queue[exit_name].pop(agent_id, None)
        self.queued_on.pop(agent_id, None)
        lane = q.lane if q is not None else self.lane_for(agent_id, exit_name)
        walking = self.admitted_walking[exit_name]
        if walking.get(lane) == agent_id:
            walking[lane] = None
        self.ml.agent_levels.pop(agent_id, None)
        self._open_rides[agent_id] = {
            "agent_id": agent_id,
            "escalator": exit_name,
            "direction": entry.spec.direction,
            "lane": lane,
            "chose_s": q.assigned_s if q is not None else time_s,
            "queue_join_s": (q.join_s if q is not None and q.join_s is not None else time_s),
            "board_s": time_s,
        }
        if entry.conveyor.can_admit(lane):
            self._put_on_conveyor(entry, agent_id, lane, time_s)
        else:
            # Admitted while there was room, but the lane filled or the belt
            # paused before they stepped on: they wait at the comb.
            self.boarding_hold[exit_name].append((agent_id, lane, time_s))

    def step(self, dt: float, time_s: float) -> None:
        positions = {
            lvl: sim.agent_tracker.get_all_positions() for lvl, sim in self.ml.simulations.items()
        }
        pending: dict[str, list[Point2]] = {lvl: [] for lvl in self.ml.simulations}
        for exit_name, entry in self.escalators.items():
            conveyor = entry.conveyor
            for rider in conveyor.step(dt):
                if self._discharge(entry, rider, positions, pending, time_s):
                    continue
                if not conveyor.paused:
                    logger.warning(f"[ESCALATOR] {exit_name}: landing full — belt paused")
                conveyor.pause()
                break
            else:
                if conveyor.paused:
                    logger.info(f"[ESCALATOR] {exit_name}: landing clear — belt resumed")
                conveyor.resume()
            hold = self.boarding_hold[exit_name]
            while hold and entry.conveyor.can_admit(hold[0][1]):
                agent_id, lane, _ = hold.pop(0)
                self._put_on_conveyor(entry, agent_id, lane, time_s)
            self._manage_queue(exit_name, entry, positions[entry.spec.from_level], time_s)
        for level in self.ml.simulations:
            if any(self.queue[n] for n in self.exits_from_level(level)):
                self._assign_slots(level, positions[level])
        if self.discovering:
            self._check_discoveries(positions)

    # ------------------------------------------------------------------
    # Snapshots / logs
    # ------------------------------------------------------------------

    def frame_snapshot(self) -> dict[str, Any]:
        out = {}
        for exit_name, entry in self.escalators.items():
            q = self.queue[exit_name]
            out[exit_name] = {
                "direction": entry.spec.direction,
                "length_m": entry.spec.length_m,
                "running": entry.conveyor.running,
                "stalled": entry.conveyor.paused,
                "closed": entry.conveyor.closed,
                "queue": {lane: sum(1 for v in q.values() if v.lane == lane) for lane in LANES},
                "riders": entry.conveyor.snapshot(),
            }
        return out

    def geometry(self) -> dict[str, Any]:
        """Static geometry for visualisation (see ``escalators.drawing``)."""
        out = {}
        for name, e in self.escalators.items():
            spec = e.spec
            out[name] = {
                "letter": spec.letter,
                "direction": spec.direction,
                "length_m": spec.length_m,
                "belt_speed": spec.belt_speed,
                "step_depth": spec.step_depth,
                "from_level": spec.from_level,
                "to_level": spec.to_level,
                "entry_comb": [list(spec.entry.a), list(spec.entry.b)],
                "exit_comb": [list(spec.exit.a), list(spec.exit.b)],
                "entry_normal": list(spec.entry.floor_normal),
                "exit_normal": list(spec.exit.floor_normal),
                "entry_width": spec.entry.width,
                "exit_width": spec.exit.width,
                "entry_strip_m": self._strip_length(spec.from_level, spec.letter),
                "exit_strip_m": self._strip_length(spec.to_level, spec.letter),
            }
        return out

    def _strip_length(self, level: str, letter: str) -> float:
        corridors = self.ml.simulations[level].geometry_manager.escalator_corridors
        poly = next(
            (p for n, p in corridors.items() if n.startswith(f"esc.{letter}.corridor.")), None
        )
        if poly is None:
            return 0.0
        r = list(poly.minimum_rotated_rectangle.exterior.coords)
        return max(math.dist(r[0], r[1]), math.dist(r[1], r[2]))

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _on_external_route(self, agent_id: str) -> None:
        """A level simulation re-routed this agent: they have left any queue."""
        self.discovering.pop(agent_id, None)
        if agent_id in self.queued_on:
            self.unassign(agent_id)

    def _check_discoveries(self, positions: dict[str, dict[str, Point2]]) -> None:
        for agent_id, exit_name in list(self.discovering.items()):
            entry = self.escalators[exit_name]
            pos = positions[entry.spec.from_level].get(agent_id)
            if pos is None:
                self.discovering.pop(agent_id)
                continue
            if math.dist(pos, entry.landing_point) <= entry.join_radius:
                self.discovering.pop(agent_id)
                self._hold_in_place(self.ml.simulations[entry.spec.from_level], agent_id, pos)
                self.ml.agents_needing_redecision.add(agent_id)
                logger.info(f"[ESCALATOR] {agent_id} reached closed {exit_name} — re-deciding")

    def _put_on_conveyor(self, entry: _Entry, agent_id: str, lane: str, time_s: float) -> None:
        base = self.ml.agent_base_speed.get(agent_id, 1.34)
        entry.conveyor.board(agent_id, lane, base * entry.spec.walk_speed_factor, time_s)
        self.riding[agent_id] = entry.spec.exit_name

    def _discharge(self, entry: _Entry, rider, positions, pending, time_s: float) -> bool:
        spec = entry.spec
        level = spec.to_level
        rider.discharge_attempts += 1
        nearby = [
            p for p in positions[level].values() if entry.discharge_area.contains(Point(p))
        ] + pending[level]
        spot = next(
            (
                c
                for c in entry.discharge_candidates[rider.lane]
                if all(math.dist(c, p) >= MIN_AGENT_SEP for p in nearby)
            ),
            None,
        )
        if spot is None:
            return False
        to_sim = self.ml.simulations[level]
        base = self.ml.agent_base_speed.get(rider.agent_id, 1.34)
        try:
            to_sim.add_agent(
                rider.agent_id, spot, walking_speed=base, assign_default_destination=False
            )
        except Exception as exc:  # JuPedSim clearance edge cases
            logger.warning(f"[ESCALATOR] could not place {rider.agent_id} at {spot}: {exc}")
            return False
        pending[level].append(spot)
        entry.conveyor.remove(rider.agent_id)
        self.riding.pop(rider.agent_id, None)
        self.ml.agent_levels[rider.agent_id] = level
        self.ml.recently_transferred_agents.add(rider.agent_id)
        egress = entry.egress_points[entry.egress_next % len(entry.egress_points)]
        entry.egress_next += 1
        to_sim.set_agent_target(rider.agent_id, egress)
        self.ml.transfer_escape_waypoints[rider.agent_id] = egress
        record = self._open_rides.pop(rider.agent_id, {})
        record.update(
            {
                "alight_s": time_s,
                "ride_s": round(time_s - rider.board_time, 2),
                "stall_wait_s": round(rider.stall_wait_s, 2),
                "discharge_attempts": rider.discharge_attempts,
            }
        )
        self.ride_log.append(record)
        logger.info(
            f"[ESCALATOR] {rider.agent_id} rode {spec.exit_name} ({rider.lane}) "
            f"in {record['ride_s']:.1f}s → level {level} at {spot}"
        )
        return True

    def _manage_queue(
        self, exit_name: str, entry: _Entry, positions: dict[str, Point2], time_s: float
    ) -> None:
        q = self.queue[exit_name]
        level_sim = self.ml.simulations[entry.spec.from_level]
        # Drop agents no longer on this floor (boarded a train, removed, …).
        for agent_id in [a for a in q if a not in positions]:
            if not self.is_in_transit(agent_id):
                self.unassign(agent_id)
        for agent_id, qe in q.items():
            d = math.dist(positions[agent_id], entry.spec.entry.mid)
            if qe.join_s is None and d <= entry.join_radius:
                qe.join_s = time_s
            elif qe.join_s is not None and not qe.admitted and d > entry.join_radius + 2.0:
                # Pushed or wandered well away from the landing: lose the place
                # rather than hold a line slot from afar.
                qe.join_s = None

        walking = self.admitted_walking[exit_name]
        strip_zone = entry.strip.buffer(AT_COMB)

        # Watchdog: an admission that has not boarded in time (boxed in) goes
        # back to the queue so it can never hold a lane indefinitely.
        for agent_id, qe in q.items():
            if (
                qe.admitted
                and qe.admitted_s is not None
                and time_s - qe.admitted_s > ADMIT_TIMEOUT_S
            ):
                qe.admitted, qe.admitted_s, qe.slot = False, None, None
                if walking.get(qe.lane) == agent_id:
                    walking[qe.lane] = None
                level_sim.agent_assigned_exits.pop(agent_id, None)
                logger.info(f"[ESCALATOR] {agent_id} could not reach {exit_name} comb — requeued")

        # Anyone already standing at the comb steps on when either lane has a
        # free step (crowd pressure can push a waiting agent onto the strip,
        # where — not being routed there — they would block it).
        for agent_id, qe in q.items():
            if (
                qe.admitted
                or qe.join_s is None
                or not strip_zone.contains(Point(positions[agent_id]))
            ):
                continue
            lane = (
                qe.lane
                if entry.conveyor.can_admit(qe.lane)
                else next((other for other in LANES if entry.conveyor.can_admit(other)), None)
            )
            if lane is not None:
                qe.lane = lane
                self._admit(exit_name, entry, agent_id, qe, time_s)

        for lane in LANES:
            prev = walking[lane]
            if (
                prev is not None
                and prev in positions
                and math.dist(positions[prev], entry.strip_target) > ADMITTED_CLEAR
            ):
                continue
            walking[lane] = None
            if not entry.conveyor.can_admit(lane):
                continue
            # Whoever in this lane's line is nearest the comb steps on (normally
            # the head-slot holder; not insisting on it means an absent or
            # boxed-in head can never stop the line).
            line = entry.line_slots[lane]
            radius = ADMIT_RADIUS if line else ADMIT_RADIUS + 0.5
            candidates = [
                (math.dist(positions[agent_id], entry.strip_target), qe.join_s, agent_id)
                for agent_id, qe in q.items()
                if qe.lane == lane
                and not qe.admitted
                and qe.join_s is not None
                and (not line or qe.slot in line)
            ]
            candidates = [c for c in candidates if c[0] <= radius]
            if not candidates:
                continue
            agent_id = min(candidates)[2]
            self._admit(exit_name, entry, agent_id, q[agent_id], time_s)
            walking[lane] = agent_id

        self._warn_if_stalled(exit_name, entry, positions, time_s)

    def _warn_if_stalled(
        self, exit_name: str, entry: _Entry, positions: dict[str, Point2], time_s: float
    ) -> None:
        """Log the queue state if people wait but nobody has boarded for a minute."""
        q = self.queue[exit_name]
        joined = [a for a, qe in q.items() if qe.join_s is not None]
        if not joined or entry.conveyor.paused or entry.conveyor.closed:
            self._last_activity[exit_name] = time_s
            return
        idle = time_s - self._last_activity.get(exit_name, time_s)
        if idle < 60 or time_s - self._last_stall_warning.get(exit_name, -1e9) < 60:
            return
        self._last_stall_warning[exit_name] = time_s
        detail = "; ".join(
            f"{a} {q[a].lane} adm={q[a].admitted} slot={q[a].slot and tuple(round(v, 2) for v in q[a].slot)} "
            f"at {tuple(round(v, 2) for v in positions[a])} "
            f"d_strip={math.dist(positions[a], entry.strip_target):.2f}"
            for a in sorted(joined, key=lambda a: math.dist(positions[a], entry.strip_target))[:6]
        )
        logger.warning(
            f"[ESCALATOR] {exit_name}: {len(joined)} queueing but no boarding for "
            f"{idle:.0f}s — {detail}; walking={self.admitted_walking[exit_name]}"
        )

    def _admit(
        self, exit_name: str, entry: _Entry, agent_id: str, qe: QueueEntry, time_s: float
    ) -> None:
        """Switch a queued agent onto the boarding strip journey."""
        level_sim = self.ml.simulations[entry.spec.from_level]
        qe.admitted, qe.admitted_s = True, time_s
        self._last_activity[exit_name] = time_s
        jps_id = level_sim.agent_tracker.get_jps_id(agent_id)
        level_sim.simulation.switch_agent_journey(
            agent_id=jps_id, journey_id=entry.journey_id, stage_id=entry.stage_id
        )
        level_sim.agent_assigned_exits[agent_id] = exit_name

    def _assign_slots(self, level: str, positions: dict[str, Point2] | None = None) -> None:
        """FIFO slot assignment for every queue on ``level`` (slots never shared)."""
        level_sim = self.ml.simulations[level]
        if positions is None:
            positions = level_sim.agent_tracker.get_all_positions()
        taken: set[Point2] = set()
        for exit_name in self.exits_from_level(level):
            entry = self.escalators[exit_name]
            q = self.queue[exit_name]

            def rank(item):
                agent_id, qe = item
                if qe.join_s is None:
                    return (1, qe.assigned_s, 0.0)
                # On the landing, order is physical (as in a real queue): a
                # line-slot holder still near their slot keeps its place (stable
                # ordering, no flicker), anyone displaced is ranked by where they
                # actually are — so nobody pushed behind the line can own its head.
                pos = positions.get(agent_id)
                here = math.dist(pos, entry.strip_target) if pos is not None else math.inf
                line = entry.line_slots[qe.lane]
                if qe.slot in line and pos is not None and math.dist(pos, qe.slot) <= 1.5:
                    here = math.dist(qe.slot, entry.strip_target)
                return (0, here, qe.join_s)

            waiting = [
                (None, None, agent_id)
                for agent_id, _ in sorted(
                    ((a, qe) for a, qe in q.items() if not qe.admitted), key=rank
                )
            ]
            used = {lane: 0 for lane in LANES}
            overflow = iter(entry.overflow_slots)
            for _, _, agent_id in waiting:
                qe = q[agent_id]
                line = entry.line_slots[qe.lane]
                slot = None
                if qe.join_s is not None and used[qe.lane] < len(line):
                    slot = line[used[qe.lane]]
                    used[qe.lane] += 1
                else:
                    for candidate in overflow:
                        if candidate not in taken:
                            slot = candidate
                            break
                    if slot is None:
                        slot = (
                            entry.overflow_slots[-1]
                            if entry.overflow_slots
                            else entry.landing_point
                        )
                taken.add(slot)
                in_line = slot in line
                if in_line and not qe.tight:
                    self._set_time_gap(level_sim, agent_id, entry.spec.queue_time_gap)
                    qe.tight = True
                elif not in_line and qe.tight:
                    self._set_time_gap(level_sim, agent_id, DEFAULT_TIME_GAP)
                    qe.tight = False
                if qe.slot != slot:
                    qe.slot = slot
                    self._route_to_point(level, agent_id, slot)

    def _route_to_point(self, level: str, agent_id: str, point: Point2) -> None:
        level_sim = self.ml.simulations[level]
        jps_id = level_sim.agent_tracker.get_jps_id(agent_id)
        if jps_id is None:
            return
        key = (level, point)
        if key not in self._waypoint_cache:
            stage = level_sim.simulation.add_waypoint_stage(point, 0.25)
            self._waypoint_cache[key] = (
                stage,
                level_sim.simulation.add_journey(jps.JourneyDescription([stage])),
            )
        stage, journey = self._waypoint_cache[key]
        level_sim.simulation.switch_agent_journey(
            agent_id=jps_id, journey_id=journey, stage_id=stage
        )
        level_sim.agent_tracker.set_target(agent_id, point)

    def _hold_in_place(self, level_sim, agent_id: str, pos: Point2) -> None:
        jps_id = level_sim.agent_tracker.get_jps_id(agent_id)
        if jps_id is None:
            return
        stage = level_sim.simulation.add_waypoint_stage(pos, 0.5)
        journey = level_sim.simulation.add_journey(jps.JourneyDescription([stage]))
        level_sim.simulation.switch_agent_journey(
            agent_id=jps_id, journey_id=journey, stage_id=stage
        )

    @staticmethod
    def _set_time_gap(level_sim, agent_id: str, value: float) -> None:
        jps_id = level_sim.agent_tracker.get_jps_id(agent_id)
        if jps_id is None:
            return
        with contextlib.suppress(Exception):
            level_sim.simulation.agent(jps_id).model.time_gap = value


def _route_length(router, start: Point2, target: Point2) -> float | None:
    try:
        wps = router.compute_waypoints(start, target)
    except Exception:
        return None
    return sum(math.dist(a, b) for a, b in pairwise(wps))
