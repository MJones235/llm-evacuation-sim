"""Map a rider's position along an escalator onto the floor plans, for drawing.

An escalator's plan footprint is one strip per level (the ``esc.X.corridor.*``
polygons). The boarding comb is the outer end of the entry level's strip and
the alighting comb the outer end of the exit level's strip; the strips meet
halfway up, where the escalator passes between the two plans.

``escalators.json`` (written per run by ``ResultsWriter``) holds the geometry
these helpers need; ``EscalatorSystem.geometry()`` produces it.
"""

from __future__ import annotations

from typing import Any


def _mid(seg: list[list[float]]) -> tuple[float, float]:
    (ax, ay), (bx, by) = seg
    return ((ax + bx) / 2, (ay + by) / 2)


def rider_floor_position(geom: dict[str, Any], s: float, lane: str) -> tuple[str, float, float]:
    """Return (level, x, y) on the plan for a rider ``s`` metres up the incline.

    ``s`` is scaled so the two plan strips together represent the full
    incline. Standers are drawn on the right of travel, walkers on the left.
    """
    length = float(geom["length_m"])
    entry_m = float(geom["entry_strip_m"])
    exit_m = float(geom["exit_strip_m"])
    plan_s = max(0.0, min(1.0, s / length)) * (entry_m + exit_m)
    side = 1.0 if lane == "stand" else -1.0

    if plan_s <= entry_m:
        mx, my = _mid(geom["entry_comb"])
        nx, ny = geom["entry_normal"]
        tx, ty = -nx, -ny  # into the escalator
        d = plan_s
        level = geom["from_level"]
        width = geom["entry_width"]
    else:
        mx, my = _mid(geom["exit_comb"])
        nx, ny = geom["exit_normal"]
        tx, ty = nx, ny  # towards the far comb
        d = -(entry_m + exit_m - plan_s)  # behind the exit comb
        level = geom["to_level"]
        width = geom["exit_width"]
    rx, ry = ty, -tx  # right of travel
    off = side * width / 4
    return level, mx + tx * d + rx * off, my + ty * d + ry * off
