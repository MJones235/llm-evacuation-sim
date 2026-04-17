"""
Spatial analysis for observation generation.

Handles zone identification, exit queries, and geometry-based calculations.
"""

import re
from typing import Any

from evacusim.utils.logger import get_logger

logger = get_logger(__name__)


class SpatialAnalyzer:
    """
    Analyzes spatial relationships in the station environment.

    Handles:
    - Zone identification from agent position
    - Nearest exit calculations
    - Distance categorization
    - Visual range checks for blocked exits
    """

    def __init__(self, station_layout: dict[str, Any], exit_registry=None):
        """
        Initialize spatial analyzer.

        Args:
            station_layout: Station geometry and zone information
            exit_registry: ExitNameRegistry for display name translation
        """
        self.zones = station_layout.get("zones", {})
        self.zones_polygons = station_layout.get("zones_polygons", {})
        self.exits = station_layout.get("exits", {})
        self.exits_polygons = station_layout.get("exits_polygons", {})
        self.obstacles = station_layout.get("obstacles", {})
        self.walkable_areas = station_layout.get("walkable_areas", {})
        self.exit_registry = exit_registry
        # Concourse-level down-escalator zones (zone_name -> centroid coords)
        self.down_access_exits: dict[str, tuple[float, float]] = station_layout.get(
            "down_access_exits", {}
        )
        # Cache of per-level walkable-area unions for boundary-aware LOS checks.
        # Keyed by level_id string; populated lazily on first use.
        self._walkable_union_cache: dict[str, Any] = {}

    def identify_zone(self, position: tuple[float, float]) -> str:
        """
        Identify which zone a position is in.

        Priority order (highest to lowest):
        1. Main footbridge (foot_bridge)
        2. Platform zones (jps.platform_N)
        3. Connector zones (platform_N_to_M)
        4. General zones (everything else)

        Args:
            position: (x, y) coordinates

        Returns:
            Zone name or "unknown area"
        """

        def _covers_or_contains(polygon, point):
            try:
                if polygon.covers(point):
                    return True
            except Exception:
                pass
            try:
                return polygon.contains(point)
            except Exception:
                return False

        def _is_main_footbridge(zone_name: str) -> bool:
            """Check if this is the main footbridge zone."""
            name_lower = zone_name.lower()
            return "foot" in name_lower and "bridge" in name_lower and "_to_" not in name_lower

        def _is_platform_zone(zone_name: str) -> bool:
            """Check if this is a platform zone."""
            name_lower = zone_name.lower()
            return "platform" in name_lower and "_to_" not in name_lower

        def _is_connector_zone(zone_name: str) -> bool:
            """Check if this is a connector zone between platforms."""
            name_lower = zone_name.lower()
            return "platform_" in name_lower and "_to_" in name_lower

        if self.zones_polygons:
            try:
                from shapely.geometry import Point

                point = Point(position)

                # Priority 1: Check main footbridge first
                for zone_name, polygon in self.zones_polygons.items():
                    if _is_main_footbridge(zone_name):
                        if _covers_or_contains(polygon, point):
                            return zone_name

                # Priority 2: Check platform zones
                for zone_name, polygon in self.zones_polygons.items():
                    if _is_platform_zone(zone_name):
                        if _covers_or_contains(polygon, point):
                            return zone_name

                # Priority 3: Check connector zones
                for zone_name, polygon in self.zones_polygons.items():
                    if _is_connector_zone(zone_name):
                        if _covers_or_contains(polygon, point):
                            return zone_name

                # Priority 4: Check all other zones
                for zone_name, polygon in self.zones_polygons.items():
                    if (
                        not _is_main_footbridge(zone_name)
                        and not _is_platform_zone(zone_name)
                        and not _is_connector_zone(zone_name)
                    ):
                        if _covers_or_contains(polygon, point):
                            return zone_name
            except Exception:
                pass

        if self.walkable_areas:
            try:
                from shapely.geometry import Point

                point = Point(position)
                for area_name, polygon in self.walkable_areas.items():
                    if _covers_or_contains(polygon, point):
                        return area_name
            except Exception:
                pass

        # Fallback to rectangular bounds
        for zone_name, zone_bounds in self.zones.items():
            if self._point_in_bounds(position, zone_bounds):
                return zone_name
        return "unknown area"

    def get_nearest_exit_info(
        self, position: tuple[float, float], agent_level: str | None = None, jps_sim=None
    ) -> str:
        """
        Get information about the nearest exit on the agent's current level.

        Args:
            position: (x, y) coordinates
            agent_level: Current level ID (e.g., "0", "-1")
            jps_sim: JuPedSim simulation (for multi-level exit access)

        Returns:
            String like "exit_name (distance_category)"
        """
        # Get level-specific exits for multi-level simulations
        exits_to_use = self.exits

        if agent_level and jps_sim and hasattr(jps_sim, "simulations"):
            level_sim = jps_sim.simulations.get(agent_level)
            if level_sim:
                # Get exits from the agent's current level
                level_exits = {}
                for exit_name in level_sim.exit_manager.evacuation_exits.keys():
                    # Find position for this exit
                    if exit_name.startswith("escalator_"):
                        # Get from walkable areas matching this escalator
                        for (
                            zone_name,
                            zone_poly,
                        ) in level_sim.geometry_manager.walkable_areas.items():
                            # Match escalator ID (e.g., "escalator_a_up" matches "L-1_esc_a_up")
                            esc_id = exit_name.replace("escalator_", "")
                            if f"_esc_{esc_id}" in zone_name:
                                level_exits[exit_name] = (
                                    zone_poly.centroid.x,
                                    zone_poly.centroid.y,
                                )
                                break
                    elif exit_name in level_sim.geometry_manager.entrance_areas:
                        poly = level_sim.geometry_manager.entrance_areas[exit_name]
                        level_exits[exit_name] = (poly.centroid.x, poly.centroid.y)

                if level_exits:
                    exits_to_use = level_exits

        if not exits_to_use:
            return "unknown"

        min_dist = float("inf")
        nearest_name = "unknown"

        for name, coords in exits_to_use.items():
            dist = ((position[0] - coords[0]) ** 2 + (position[1] - coords[1]) ** 2) ** 0.5
            if dist < min_dist:
                min_dist = dist
                nearest_name = name

        # Make escalator names more readable
        display_name = nearest_name
        if nearest_name.startswith("escalator_"):
            # Convert "escalator_a_up" to "escalator A (up to concourse)"
            parts = nearest_name.replace("escalator_", "").split("_")
            if len(parts) == 2:
                letter, direction = parts
                if direction == "up":
                    display_name = f"escalator {letter.upper()} (up to concourse)"
                elif direction == "down":
                    display_name = f"escalator {letter.upper()} (down to platforms)"
                else:
                    display_name = f"escalator {letter.upper()}"

        # Categorize distance to prevent small changes from triggering LLM calls
        if min_dist >= 100:
            dist_category = "100m+"
        elif min_dist >= 50:
            dist_category = "50-100m"
        else:
            dist_category = "<50m"
        return f"{display_name} ({dist_category})"

    def get_visible_exits(
        self,
        position: tuple[float, float],
        agent_level: str | None = None,
        jps_sim=None,
        inactive_exits: set[str] | None = None,
        blocked_exits: set[str] | None = None,
    ) -> list[dict[str, str]]:
        """
        Get all exits visible from the agent's position.

        An exit is visible if it has an unobstructed line of sight from the
        agent's position.  Agents are assumed to have unlimited visual range,
        so only the geometry (obstacles) gates visibility.

        Args:
            position: Agent's (x, y) position
            agent_level: Current level ID (e.g., "0", "-1")
            jps_sim: JuPedSim simulation (for multi-level exit access)
            inactive_exits: Set of canonical exit keys to exclude from visibility
                (used to hide train exits before the train arrives).
            blocked_exits: Set of currently-blocked exit canonical IDs to exclude
                from the visible exits list (they will appear instead in the
                visual blocked-exit observations).

        Returns:
            List of dicts with keys ``id`` (canonical exit key), ``name``
            (display name) and ``distance`` (category string)
        """
        visible_by_key: dict[str, dict[str, Any]] = {}

        # Get level-specific exits for multi-level simulations
        exits_to_check = {}

        if agent_level and jps_sim and hasattr(jps_sim, "simulations"):
            level_sim = jps_sim.simulations.get(agent_level)
            if level_sim:
                # Get all exits on this level (street exits + up escalators)
                for exit_name in level_sim.exit_manager.evacuation_exits.keys():
                    if exit_name in level_sim.exit_manager.exit_coordinates:
                        exits_to_check[exit_name] = level_sim.exit_manager.exit_coordinates[
                            exit_name
                        ]
                # Also include pre-blocked exits (removed from navmesh at startup).
                # Their physical location (the escalator shaft entrance) is still
                # visible to agents — they should appear in the exit list and only
                # become "blocked" once an agent gets close enough to see the barrier.
                if level_sim and hasattr(level_sim, "geometry_manager"):
                    for exit_name, pos in level_sim.geometry_manager.blocked_exit_positions.items():
                        if exit_name not in exits_to_check:
                            exits_to_check[exit_name] = pos
            # Level 0: also add down-access escalator zones (not in evacuation_exits;
            # they are walkable-area level-transfer triggers)
            if agent_level == "0" and self.down_access_exits:
                exits_to_check.update(self.down_access_exits)
        else:
            # Single-level or no level info: use all exits
            exits_to_check = self.exits

        # Filter out exits that are not yet active (e.g. train exits before train arrives).
        if inactive_exits:
            exits_to_check = {
                name: pos for name, pos in exits_to_check.items()
                if self._canonical_visible_exit_key(name) not in inactive_exits
            }

        # Filter out exits that are currently blocked.
        # - For runtime-blocked exits (created by an event after spawn): hide
        #   immediately — agents can reasonably assume the exit is inaccessible
        #   once an announcement or visual cue reaches them.
        # - For pre-blocked exits (shaft removed from navmesh at startup, stored
        #   in blocked_exit_positions): keep them visible UNTIL the agent is
        #   within the discovery radius.  The physical escalator shaft is still
        #   there and labelled; only the barrier at the entrance reveals the
        #   blockage.  This produces the "walk-up-and-discover" behaviour.
        if blocked_exits:
            # Resolve the level geometry manager once for the proximity check.
            _pre_blocked_pos: dict[str, tuple[float, float]] = {}
            if agent_level and jps_sim and hasattr(jps_sim, "simulations"):
                _lvl = jps_sim.simulations.get(agent_level)
                if _lvl and hasattr(_lvl, "geometry_manager"):
                    _pre_blocked_pos = _lvl.geometry_manager.blocked_exit_positions
            _DISCOVERY_RADIUS = 8.0

            def _keep_exit(name: str, pos: tuple[float, float]) -> bool:
                if name not in blocked_exits:
                    return True  # not blocked — always visible
                if name in _pre_blocked_pos:
                    # Pre-blocked: only hide once within discovery range
                    dist_sq = (position[0] - pos[0]) ** 2 + (position[1] - pos[1]) ** 2
                    return dist_sq > _DISCOVERY_RADIUS ** 2
                return False  # runtime-blocked: hide immediately

            exits_to_check = {
                name: pos for name, pos in exits_to_check.items()
                if _keep_exit(name, pos)
            }

        # Prefer level-specific obstacles for line-of-sight checks.
        # NOTE: On underground platform levels (level_id starting with "-"),
        # JuPedSim obstacle polygons represent escalator shaft walls and internal
        # structural elements.  These obstacles block every ray to any escalator
        # TZ centroid, making ALL escalator exits invisible if we use obstacle LOS.
        # Escalator exits should always be visible to agents on the same level
        # (the station layout is open and agents can see escalator signage).
        # We therefore skip obstacle-based LOS for exit visibility and rely only
        # on the walkable-area boundary check (which is sufficient on level 0
        # for filtering out street exits around concourse corners).
        level_obstacles = None
        level_walkable_geom = None
        if agent_level and jps_sim and hasattr(jps_sim, "simulations"):
            level_sim = jps_sim.simulations.get(agent_level)
            if level_sim and hasattr(level_sim, "geometry_manager"):
                # Build (and cache) the union of all walkable areas for this level.
                # Used to reject sight lines that pass through a wall formed by
                # the concourse boundary itself (i.e. the concave gap between the
                # ABC escalator bank and the Eldon Square corridor).
                if agent_level not in self._walkable_union_cache:
                    try:
                        from shapely.ops import unary_union
                        polys = list(
                            level_sim.geometry_manager.walkable_areas.values()
                        )
                        self._walkable_union_cache[agent_level] = (
                            unary_union(polys) if polys else None
                        )
                    except Exception:
                        self._walkable_union_cache[agent_level] = None
                level_walkable_geom = self._walkable_union_cache.get(agent_level)

        # Check each exit for visibility
        for exit_name, exit_pos in exits_to_check.items():
            distance = ((position[0] - exit_pos[0]) ** 2 + (position[1] - exit_pos[1]) ** 2) ** 0.5

            if self._has_line_of_sight(position, exit_pos, level_obstacles, level_walkable_geom):
                # Use display name if registry available
                display_name = (
                    self.exit_registry.get_display_name(exit_name)
                    if self.exit_registry
                    else exit_name
                )

                # Some levels expose the same physical escalator under two IDs
                # (e.g. "escalator_a_down" and "L0_esc_a_down"). Deduplicate here.
                canonical_key = self._canonical_visible_exit_key(exit_name)
                existing = visible_by_key.get(canonical_key)

                if existing is None:
                    visible_by_key[canonical_key] = {
                        "name": display_name,
                        "distance_m": distance,
                    }
                else:
                    # Keep closest distance category and prefer richer human labels.
                    existing["distance_m"] = min(existing["distance_m"], distance)
                    existing["name"] = self._prefer_exit_label(existing["name"], display_name)

        visible_exits = []
        for canonical_key, exit_info in sorted(
            visible_by_key.items(), key=lambda kv: (kv[1]["distance_m"], kv[1]["name"].lower())
        ):
            distance = exit_info["distance_m"]
            if distance < 10:
                dist_cat = "very close"
            elif distance < 20:
                dist_cat = "nearby"
            else:
                dist_cat = "visible in distance"

            visible_exits.append({"id": canonical_key, "name": exit_info["name"], "distance": dist_cat})

        return visible_exits

    def _has_line_of_sight(
        self,
        start: tuple[float, float],
        end: tuple[float, float],
        obstacles: dict[str, Any] | None = None,
        walkable_geom: Any | None = None,
    ) -> bool:
        """Return True if the segment from start to end is not blocked.

        Two checks are applied:
        1. No explicit obstacle polygon crosses the sight line.
        2. The sight line lies entirely within the walkable area (prevents
           false-positive visibility across concave concourse boundaries, e.g.
           the gap between the ABC escalator bank and the Eldon Square corridor).
        """
        obstacles_to_check = obstacles if obstacles is not None else self.obstacles

        try:
            from shapely.geometry import LineString

            sight_line = LineString([start, end])

            # Check 1: walkable boundary. If the sight line exits the passable
            # footprint at any point the two positions cannot see each other.
            if walkable_geom is not None:
                try:
                    if not walkable_geom.covers(sight_line):
                        return False
                except Exception:
                    pass  # geometry error; fall through to obstacle check

            # Check 2: explicit obstacle polygons.
            # obstacles_to_check may be a list (from geometry_manager) or a dict.
            if obstacles_to_check:
                polys_iter = (
                    obstacles_to_check.values()
                    if isinstance(obstacles_to_check, dict)
                    else obstacles_to_check
                )
                for poly in polys_iter:
                    if poly is None:
                        continue
                    # Treat any interior intersection as blocked. Endpoint touching is fine.
                    if sight_line.crosses(poly) or sight_line.within(poly):
                        return False

                    if sight_line.intersects(poly):
                        inter = sight_line.intersection(poly)
                        if not inter.is_empty and not inter.touches(sight_line.boundary):
                            return False
        except Exception:
            # Fallback: allow visibility if geometry checks fail outright.
            return True

        return True

    def _canonical_visible_exit_key(self, exit_name: str) -> str:
        """Map equivalent exit IDs to one canonical key for display deduplication."""
        escalator_match = re.match(r"^escalator_([a-z])_(up|down)$", exit_name)
        if escalator_match:
            letter, direction = escalator_match.groups()
            return f"escalator_{letter}_{direction}"

        zone_escalator_match = re.match(r"^L[^_]+_esc_([a-z])_(up|down)$", exit_name)
        if zone_escalator_match:
            letter, direction = zone_escalator_match.groups()
            return f"escalator_{letter}_{direction}"

        return exit_name

    def _prefer_exit_label(self, current_label: str, candidate_label: str) -> str:
        """Prefer the more descriptive label when equivalent exits collide."""
        current_specific = " to " in current_label.lower()
        candidate_specific = " to " in candidate_label.lower()

        if candidate_specific and not current_specific:
            return candidate_label
        if current_specific and not candidate_specific:
            return current_label

        if len(candidate_label) > len(current_label):
            return candidate_label
        return current_label

    def get_visible_blocked_exits(
        self,
        position: tuple[float, float],
        blocked_exits: set[str],
        agent_level: str | None = None,
        jps_sim=None,
    ) -> list[dict[str, Any]]:
        """
        Get blocked exits that an agent can perceive from their current position.

        Uses the same line-of-sight check as :meth:`get_visible_exits` so that
        walls (e.g. the platform tunnel walls separating platforms from the
        escalator shafts) correctly block awareness of a blockage until the
        agent has a clear sightline to it.

        Args:
            position: Agent's (x, y) position
            blocked_exits: Set of blocked exit canonical IDs
            agent_level: Current level ID (e.g. "0", "-1")
            jps_sim: JuPedSim simulation (for level-specific geometry)

        Returns:
            List of visible blocked exits with name and distance category
        """
        # Resolve level-specific exit coordinates and LOS geometry.
        level_obstacles = None
        level_walkable_geom = None
        exits_to_check: dict[str, tuple[float, float]] = {}

        if agent_level and jps_sim and hasattr(jps_sim, "simulations"):
            level_sim = jps_sim.simulations.get(str(agent_level))
            if level_sim:
                if hasattr(level_sim, "exit_manager"):
                    exits_to_check = dict(level_sim.exit_manager.exit_coordinates)
                if hasattr(level_sim, "geometry_manager"):
                    level_obstacles = getattr(level_sim.geometry_manager, "obstacles", None)
                    if agent_level not in self._walkable_union_cache:
                        try:
                            from shapely.ops import unary_union
                            polys = list(level_sim.geometry_manager.walkable_areas.values())
                            self._walkable_union_cache[agent_level] = (
                                unary_union(polys) if polys else None
                            )
                        except Exception:
                            self._walkable_union_cache[agent_level] = None
                    level_walkable_geom = self._walkable_union_cache.get(agent_level)
        else:
            exits_to_check = self.exits

        visible_blocked = []

        for exit_name in blocked_exits:
            exit_pos = exits_to_check.get(exit_name)
            # Determine whether this is a pre-blocked exit (navmesh removed).
            # Pre-blocked exits are surrounded by obstacle polygons that define
            # the escalator shaft — any LOS ray to their centroid will be
            # blocked by those same obstacles even at close range.  For these
            # exits, use a simple proximity check inside a discovery radius so
            # that agents who navigate toward the blocked shaft can still learn
            # about the blockage.
            is_pre_blocked = False
            if exit_pos is None:
                # Not in regular exit_coordinates — look in blocked_exit_positions.
                if agent_level and jps_sim and hasattr(jps_sim, "simulations"):
                    lvl = jps_sim.simulations.get(str(agent_level))
                    if lvl and hasattr(lvl, "geometry_manager"):
                        bp = lvl.geometry_manager.blocked_exit_positions.get(exit_name)
                        if bp is not None:
                            exit_pos = bp
                            is_pre_blocked = True
            if exit_pos is None:
                # Final fallback: single-level exits dict
                exit_pos = self.exits.get(exit_name)
            if exit_pos is None:
                continue

            distance = (
                (position[0] - exit_pos[0]) ** 2 + (position[1] - exit_pos[1]) ** 2
            ) ** 0.5

            if is_pre_blocked:
                # Obstacle-based LOS cannot work for pre-blocked exits (the shaft
                # walls block every ray).  Use proximity: the blockage is visible
                # once the agent is close enough to see the barrier/marshal
                # standing at the escalator entrance (~8 m).  This is small enough
                # that agents on platform 1/2 need to walk toward E/F before
                # discovering the blockage, giving the "walk-up-and-discover"
                # behaviour, yet large enough to be detected comfortably.
                _PREBLOCKED_DISCOVERY_RADIUS = 8.0
                if distance > _PREBLOCKED_DISCOVERY_RADIUS:
                    continue
            else:
                if not self._has_line_of_sight(position, exit_pos, level_obstacles, level_walkable_geom):
                    continue

            if distance < 5:
                dist_cat = "very close"
            elif distance < 15:
                dist_cat = "nearby"
            else:
                dist_cat = "visible in the distance"

            display_name = (
                self.exit_registry.get_display_name(exit_name)
                if self.exit_registry
                else exit_name
            )
            visible_blocked.append({"name": display_name, "distance": dist_cat})

        return visible_blocked

    def _point_in_bounds(self, point: tuple[float, float], bounds: dict) -> bool:
        """Check if a point is within rectangular bounds."""
        x, y = point
        return bounds["x_min"] <= x <= bounds["x_max"] and bounds["y_min"] <= y <= bounds["y_max"]
