"""Routing agents who have just stepped off an escalator onto a new level.

An agent leaving an escalator first needs to clear the landing. Until it
has, it should not stop to make a decision (it would block the comb). This
module decides, for such an agent, whether its decision should be deferred
this cycle, and routes it meanwhile:

- a boarder with a known platform walks straight to a point on that platform
  and decides once there;
- an alighter arriving on the concourse decides immediately (to pick a
  street exit);
- anyone else walks to the escalator's egress waypoint first.

The waypoints come from the pedestrian simulation
(``transfer_escape_waypoints`` and ``transfer_platform_waypoints``).
"""

from __future__ import annotations

import hashlib
import random
import re

from shapely.geometry import Point
from shapely.ops import nearest_points

from evacusim.decision.situation import GoalTracker, goal_is_train_oriented
from evacusim.utils.logger import get_logger

logger = get_logger(__name__)

ARRIVAL_RADIUS_M = 2.0
"""An agent within this distance of its waypoint has arrived."""


class PostTransferRouting:
    """Defers decisions for agents still clearing an escalator landing.

    Args:
        jps_sim: The multi-level pedestrian simulation.
        zones_polygons: Named zone polygons (platform zones are used).
        agent_cfg: Per-agent records (the ``target`` platform is used).
        goals: Agents' current goals.
    """

    def __init__(
        self, jps_sim, zones_polygons: dict, agent_cfg: dict[str, dict], goals: GoalTracker
    ):
        self._jps_sim = jps_sim
        self._zones_polygons = zones_polygons
        self._agent_cfg = agent_cfg
        self._goals = goals

    def should_defer(self, agent_id: str, position: tuple[float, float]) -> bool:
        """True if the agent should keep walking rather than decide this cycle."""
        platform_waypoints = getattr(self._jps_sim, "transfer_platform_waypoints", {})
        if agent_id in platform_waypoints:
            if _distance(position, platform_waypoints[agent_id]) > ARRIVAL_RADIUS_M:
                return True
            del platform_waypoints[agent_id]
            logger.debug(f"{agent_id}: reached assigned platform — decision now permitted")
            return False

        escape_waypoints = getattr(self._jps_sim, "transfer_escape_waypoints", {})
        if agent_id not in escape_waypoints:
            return False

        # A boarder with a known platform goes straight there.
        platform_waypoint = self._platform_approach_waypoint(agent_id, position)
        if platform_waypoint is not None:
            del escape_waypoints[agent_id]
            self._jps_sim.set_agent_target(agent_id, platform_waypoint)
            platform_waypoints[agent_id] = platform_waypoint
            logger.debug(
                f"{agent_id}: transferred — continuing directly to assigned platform "
                f"at {platform_waypoint}"
            )
            return True

        # Someone arriving on the concourse to leave chooses a street exit now.
        level = self._jps_sim.get_agent_level(agent_id)
        if level == "0" and not goal_is_train_oriented(self._goals.goals.get(agent_id, "")):
            del escape_waypoints[agent_id]
            logger.debug(f"{agent_id}: transferred to concourse — choosing a street exit")
            return False

        # Anyone else clears the escalator via its egress waypoint first.
        distance = _distance(position, escape_waypoints[agent_id])
        if distance > ARRIVAL_RADIUS_M:
            logger.debug(
                f"{agent_id}: en route to post-transfer waypoint "
                f"({distance:.1f}m away) — deferring decision"
            )
            return True
        del escape_waypoints[agent_id]
        logger.debug(f"{agent_id}: reached post-transfer waypoint — decision now permitted")
        return False

    def _platform_approach_waypoint(
        self, agent_id: str, position: tuple[float, float]
    ) -> tuple[float, float] | None:
        """A stable per-agent random point on the agent's target platform, if it has one."""
        target = str(self._agent_cfg.get(agent_id, {}).get("target", "")).lower()
        platform_zone = target[len("train_") :] if target.startswith("train_platform_") else target
        if not re.fullmatch(r"platform_[1-4]", platform_zone):
            return None

        polygon = self._zones_polygons.get(platform_zone)
        if polygon is None or polygon.is_empty:
            return None
        safe_polygon = polygon.buffer(-0.3)
        if safe_polygon.is_empty:
            safe_polygon = polygon

        level_id = self._jps_sim.get_agent_level(agent_id)
        level_sim = getattr(self._jps_sim, "simulations", {}).get(level_id)
        combined = getattr(getattr(level_sim, "geometry_manager", None), "_combined_geometry", None)
        if combined is not None and not combined.is_empty:
            accessible = safe_polygon.intersection(combined.buffer(-0.05))
            if not accessible.is_empty:
                safe_polygon = accessible

        seed_bytes = hashlib.sha256(f"{agent_id}:{platform_zone}".encode()).digest()[:8]
        rng = random.Random(int.from_bytes(seed_bytes, "big"))
        min_x, min_y, max_x, max_y = safe_polygon.bounds
        for _ in range(500):
            candidate = Point(rng.uniform(min_x, max_x), rng.uniform(min_y, max_y))
            if safe_polygon.contains(candidate) and candidate.distance(Point(position)) <= 30.0:
                return (float(candidate.x), float(candidate.y))

        waypoint = nearest_points(Point(position), safe_polygon)[1]
        return (float(waypoint.x), float(waypoint.y))


def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5
