"""Build ``EscalatorSpec`` records from geometry comb metadata plus config.

Geometry (``jupedsim.escalator_comb`` polys in each ``level_*.xml``) says *where*
each escalator's combs are; config ``simulation.escalators`` says how it
behaves (belt speed, walking share, …), with per-escalator overrides keyed by
the decision-facing exit name (e.g. ``escalator_f_up``).
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any

Point2 = tuple[float, float]

DEFAULTS: dict[str, Any] = {
    "belt_speed": 0.5,
    "step_depth": 0.4,
    "walk_share": {"up": 0.25, "down": 0.40},
    "stander_step_gap_prob": 0.5,
    # Fraction of an agent's floor walking speed they manage on the incline.
    "walk_speed_factor": {"up": 0.5, "down": 0.7},
    "queue_line_slots": 4,
    "queue_line_spacing": 0.6,
    "queue_slot_grid": 0.7,
    "queue_max_route_m": 25.0,
    "queue_time_gap": 0.5,
    "landing_search_depth": 1.5,
}


@dataclass(frozen=True)
class Comb:
    level: str
    a: Point2
    b: Point2
    # Unit normal pointing from the comb onto the landing floor.
    floor_normal: Point2

    @property
    def mid(self) -> Point2:
        return ((self.a[0] + self.b[0]) / 2, (self.a[1] + self.b[1]) / 2)

    @property
    def width(self) -> float:
        return math.dist(self.a, self.b)

    def lane_point(self, lane: str, out_m: float, direction_of_travel: Point2) -> Point2:
        """Point ``out_m`` onto the landing, on ``lane``'s side of the comb.

        ``direction_of_travel`` is the rider's heading at this comb; stand is
        on the right of travel, walk on the left.
        """
        tx, ty = direction_of_travel
        rx, ry = ty, -tx                       # right of travel
        side = 1.0 if lane == "stand" else -1.0
        off = side * self.width / 4
        mx, my = self.mid
        nx, ny = self.floor_normal
        return (mx + rx * off + nx * out_m, my + ry * off + ny * out_m)


@dataclass(frozen=True)
class EscalatorSpec:
    exit_name: str
    letter: str
    direction: str            # "up" | "down"
    entry: Comb
    exit: Comb
    egress_target: Point2 | None
    length_m: float
    belt_speed: float
    step_depth: float
    walk_share: float
    stander_step_gap_prob: float
    walk_speed_factor: float
    queue_line_slots: int
    queue_line_spacing: float
    queue_slot_grid: float
    queue_max_route_m: float
    queue_time_gap: float
    landing_search_depth: float

    @property
    def from_level(self) -> str:
        return self.entry.level

    @property
    def to_level(self) -> str:
        return self.exit.level

    @property
    def boarding_heading(self) -> Point2:
        """Direction of travel when stepping on (away from the entry landing)."""
        return (-self.entry.floor_normal[0], -self.entry.floor_normal[1])

    @property
    def alighting_heading(self) -> Point2:
        """Direction of travel when stepping off (onto the exit landing)."""
        return self.exit.floor_normal


def load_combs(xml_path: str | Path, level: str) -> list[dict[str, Any]]:
    root = ET.parse(str(xml_path)).getroot()
    combs = []
    for poly in root.findall('.//poly[@type="jupedsim.escalator_comb"]'):
        pts = [tuple(map(float, p.split(","))) for p in poly.get("shape", "").split()]
        if len(pts) != 2:
            raise ValueError(f"{xml_path}: comb {poly.get('id')} must be a 2-point segment")
        record = {
            "level": str(level),
            "letter": poly.get("escalator", ""),
            "role": poly.get("role", ""),
            "exit_name": poly.get("exit_name", ""),
            "direction": poly.get("direction", ""),
            "length_m": float(poly.get("length_m", "nan")),
            "a": pts[0],
            "b": pts[1],
            "floor_normal": (float(poly.get("floor_nx")), float(poly.get("floor_ny"))),
        }
        if poly.get("egress_x") is not None:
            record["egress"] = (float(poly.get("egress_x")), float(poly.get("egress_y")))
        combs.append(record)
    return combs


def _pick(value: Any, direction: str) -> Any:
    return value.get(direction) if isinstance(value, dict) else value


def build_specs(
    network_path: str | Path,
    levels: list[str],
    escalator_config: dict[str, Any] | None = None,
) -> list[EscalatorSpec]:
    """Return one spec per escalator that has both an entry and an exit comb."""
    cfg = escalator_config or {}
    defaults = {**DEFAULTS, **(cfg.get("defaults") or {})}
    overrides = cfg.get("overrides") or {}

    records: dict[str, dict[str, dict]] = {}
    for level in levels:
        path = Path(network_path) / f"level_{level}.xml"
        if not path.exists():
            continue
        for comb in load_combs(path, level):
            records.setdefault(comb["exit_name"], {})[comb["role"]] = comb

    specs = []
    for exit_name, sides in sorted(records.items()):
        if "entry" not in sides or "exit" not in sides:
            raise ValueError(f"escalator {exit_name}: needs one entry and one exit comb, "
                             f"got {sorted(sides)}")
        entry, exit_ = sides["entry"], sides["exit"]
        direction = entry["direction"]
        params = {**defaults, **(overrides.get(exit_name) or {})}
        length = float(params.get("length_m", entry["length_m"]))
        specs.append(EscalatorSpec(
            exit_name=exit_name,
            letter=entry["letter"],
            direction=direction,
            entry=Comb(entry["level"], entry["a"], entry["b"], entry["floor_normal"]),
            exit=Comb(exit_["level"], exit_["a"], exit_["b"], exit_["floor_normal"]),
            egress_target=exit_.get("egress"),
            length_m=length,
            belt_speed=float(params["belt_speed"]),
            step_depth=float(params["step_depth"]),
            walk_share=float(_pick(params["walk_share"], direction)),
            stander_step_gap_prob=float(params["stander_step_gap_prob"]),
            walk_speed_factor=float(_pick(params["walk_speed_factor"], direction)),
            queue_line_slots=int(params["queue_line_slots"]),
            queue_line_spacing=float(params["queue_line_spacing"]),
            queue_slot_grid=float(params["queue_slot_grid"]),
            queue_max_route_m=float(params["queue_max_route_m"]),
            queue_time_gap=float(params["queue_time_gap"]),
            landing_search_depth=float(params["landing_search_depth"]),
        ))
    unknown = set(overrides) - {s.exit_name for s in specs}
    if unknown:
        raise ValueError(f"simulation.escalators.overrides for unknown escalators: {sorted(unknown)}")
    return specs
