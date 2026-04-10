"""
Evacuation exit management for JuPedSim simulation.

Handles creation and management of evacuation exits and journey routing
to exits in the station environment.
"""

import re
from typing import Any

from evacusim.utils.logger import get_logger
from evacusim.jps.stage_manager import StageManager
from evacusim.jps.geometry_processor import GeometryProcessor

logger = get_logger(__name__)


class ExitManager:
    """
    Manages evacuation exits and journey routing.

    Handles:
    - Creation of exit stages from entrance polygons
    - Journey routing to exits
    - Exit ID and journey ID tracking
    - Exit placement validation
    """

    def __init__(
        self,
        stage_manager: StageManager,
        entrance_areas: dict[str, Any],
        walkable_areas_with_obstacles: dict[str, Any],
        walkable_areas: dict[str, Any] | None = None,
        level_id: str | int = "0",
    ):
        """
        Initialize exit manager and create evacuation exits.

        Args:
            stage_manager: StageManager instance for creating exits and journeys
            entrance_areas: Dictionary of entrance area polygons (street exits)
            walkable_areas_with_obstacles: Dictionary of walkable area polygons
            walkable_areas: Dictionary of all walkable areas (for escalator detection)
            level_id: Level identifier for filtering escalator zones (default: "0")

        Raises:
            RuntimeError: If no valid exits can be created from entrance areas
        """
        self.stage_manager = stage_manager
        self.entrance_areas = entrance_areas
        self.walkable_areas_with_obstacles = walkable_areas_with_obstacles
        self.walkable_areas = walkable_areas or {}
        self.level_id = str(level_id)

        # Setup evacuation exits and routes
        escalable_zones = [z for z in (walkable_areas or {}).keys() if f"L{self.level_id}_esc" in z]
        logger.info(
            f"Setting up evacuation exits for level {self.level_id}: "
            f"entrance_areas={list(entrance_areas.keys()) if entrance_areas else 'None'}, "
            f"escalable_zones={escalable_zones}"
        )

        walkable_geometry = GeometryProcessor.combine_geometry(
            list(walkable_areas_with_obstacles.values())
        )

        self.evacuation_exits, self.evacuation_journeys = self._setup_evacuation_exits(
            walkable_geometry
        )

        # Also store exit coordinates for translation layer
        self.exit_coordinates = {}  # exit_name -> (x, y)
        self._populate_exit_coordinates()

        logger.info(
            f"Created {len(self.evacuation_exits)} exits: {list(self.evacuation_exits.keys())}"
        )

    def _setup_evacuation_exits(
        self, walkable_geometry: Any
    ) -> tuple[dict[str, int], dict[str, int]]:
        """
        Create evacuation exit stages at entrance locations or escalators.

        For levels with street exits (concourse): Creates exits at entrance areas.
        For levels without exits (platforms): Creates exits at escalator locations,
        allowing agents to naturally choose which escalator to use.

        Args:
            walkable_geometry: Combined walkable geometry for validation

        Returns:
            Tuple of (evacuation_exits dict, evacuation_journeys dict)
        """
        evacuation_exits = {}
        evacuation_journeys = {}

        if not self.entrance_areas:
            # No street exits - look for escalators to use as "exits" from this level
            logger.info(
                f"No entrance areas found. Checking for escalators in walkable_areas: {list(self.walkable_areas.keys())}"
            )
            escalator_exits, escalator_journeys = self._create_escalator_exits(walkable_geometry)
            if escalator_exits:
                logger.info(
                    f"Created {len(escalator_exits)} escalator exits on this level: {list(escalator_exits.keys())}"
                )
                return escalator_exits, escalator_journeys
            else:
                logger.warning("No escalators found in walkable_areas")
                return {}, {}

        failed_exits = []
        for entrance_name, entrance_polygon in self.entrance_areas.items():
            exit_id = self._create_convex_exit_from_polygon(
                entrance_name, entrance_polygon, walkable_geometry
            )

            if exit_id is None:
                failed_exits.append(entrance_name)
                logger.warning(f"Failed to create exit at '{entrance_name}'")
                continue

            evacuation_exits[entrance_name] = exit_id

            # Create journey to this exit
            journey_id = self.stage_manager.create_simple_exit_journey(
                journey_name=f"journey_to_{entrance_name}", exit_id=exit_id
            )
            evacuation_journeys[entrance_name] = journey_id

            logger.info(
                f"Created evacuation exit '{entrance_name}' "
                f"(exit={exit_id}, journey={journey_id})"
            )

        if not evacuation_exits and self.entrance_areas:
            # Only warn if we expected to create exits but failed
            logger.warning(
                f"Failed to create any evacuation exits. "
                f"Attempted exits: {list(self.entrance_areas.keys())}. "
                f"All exits failed. Check geometry configuration and ensure "
                f"entrance areas overlap with walkable areas."
            )

        # Also register down-escalator exits on levels that have street exits (e.g. level 0).
        # These escalator zones are not JPS entrance areas, but agents must be able to
        # exit through them to trigger the level transfer down to the platform level.
        # Only register "down" direction escalators to avoid polluting the exit set
        # with the UP escalators that are merely arrival zones on this level.
        escalator_pattern = re.compile(r"^L([^_]+)_esc_([a-f])_(down)$")
        for zone_name, zone_polygon in self.walkable_areas.items():
            match = escalator_pattern.match(zone_name)
            if not match:
                continue
            level, esc_id, _direction = match.groups()
            if level != self.level_id:
                continue
            exit_name = f"escalator_{esc_id}_down"
            if exit_name in evacuation_exits:
                continue  # Already registered (shouldn't happen, but be safe)
            try:
                coords = list(zone_polygon.exterior.coords)[:-1]
                if len(coords) < 3:
                    continue
                exit_id = self.stage_manager.create_exit_at_coordinates(
                    exit_name=exit_name, coords=coords
                )
                journey_id = self.stage_manager.create_simple_exit_journey(
                    journey_name=f"journey_to_{exit_name}", exit_id=exit_id
                )
                evacuation_exits[exit_name] = exit_id
                evacuation_journeys[exit_name] = journey_id
                logger.info(
                    f"Created down-escalator exit '{exit_name}' on level {self.level_id} "
                    f"(from {zone_name}, exit={exit_id})"
                )
            except Exception as e:
                logger.warning(f"Failed to create down-escalator exit '{exit_name}': {e}")

        return evacuation_exits, evacuation_journeys

    def _create_escalator_exits(
        self, walkable_geometry: Any
    ) -> tuple[dict[str, int], dict[str, int]]:
        """
        Create real terminal exits at escalator locations for levels without street exits.

        Escalators are real exits from one level. When agents reach an escalator exit,
        they exit the level and can be transferred to the other level.

        In multi-level simulations, each escalator is accessible from both levels:
        - L-1 agents exit through escalator zones to reach level 0
        - L0 agents exit through corresponding escalator zones to reach level -1

        Escalators are identified by naming pattern: L{level}_esc_{id}_{direction}
        Example: L-1_esc_a_up, L0_esc_b_down

        Args:
            walkable_geometry: Combined walkable geometry for validation

        Returns:
            Tuple of (evacuation_exits dict, evacuation_journeys dict)
        """
        escalator_pattern = re.compile(r"^L([^_]+)_esc_([a-f])_(up|down)$")
        evacuation_exits = {}
        evacuation_journeys = {}

        for zone_name, zone_polygon in self.walkable_areas.items():
            match = escalator_pattern.match(zone_name)
            if not match:
                continue

            level, esc_id, direction = match.groups()

            # Only create exits for escalators on the current level
            if level != self.level_id:
                logger.debug(f"Skipping escalator zone {zone_name} (not for level {self.level_id})")
                continue

            # On platform levels (no street exits), only UP escalators are departure exits.
            # DOWN escalator zones on this level are *arrival* zones — agents spawn here
            # after transferring from above.  Registering them as JuPedSim exit stages
            # would immediately remove any newly-spawned agent and bounce them back up.
            if direction == "down":
                logger.debug(
                    f"Skipping escalator zone {zone_name}: 'down' zones are arrival zones "
                    f"on level {self.level_id}, not departure exits"
                )
                continue

            exit_name = f"escalator_{esc_id}_{direction}"

            try:
                # Create a real terminal exit at the escalator zone
                # Use the polygon coordinates directly as the exit boundary
                coords = list(zone_polygon.exterior.coords)[:-1]  # Remove closing coord
                if len(coords) < 3:
                    logger.warning(f"Escalator {exit_name} has invalid polygon (<3 points)")
                    continue

                exit_id = self.stage_manager.create_exit_at_coordinates(
                    exit_name=exit_name, coords=coords
                )

                # Create a simple journey to this exit
                journey_id = self.stage_manager.create_simple_exit_journey(
                    journey_name=f"journey_to_{exit_name}", exit_id=exit_id
                )

                evacuation_exits[exit_name] = exit_id
                evacuation_journeys[exit_name] = journey_id

                logger.info(
                    f"Created escalator exit '{exit_name}' "
                    f"(exit={exit_id}, journey={journey_id})"
                )

            except Exception as e:
                logger.warning(f"Failed to create escalator exit '{exit_name}': {e}")
                continue

        return evacuation_exits, evacuation_journeys

    def _populate_exit_coordinates(self):
        """Populate exit_coordinates dict with exit center positions."""
        # Add entrance area coordinates (street exits)
        for entrance_name, entrance_polygon in self.entrance_areas.items():
            if entrance_name in self.evacuation_exits:
                centroid = entrance_polygon.centroid
                self.exit_coordinates[entrance_name] = (centroid.x, centroid.y)
                logger.debug(
                    f"Added street exit {entrance_name} at {centroid.x:.2f}, {centroid.y:.2f}"
                )

        # Add escalator coordinates (escalator zone centers) - ONLY for current level
        escalator_pattern = re.compile(r"^L([^_]+)_esc_([a-f])_(up|down)$")
        for zone_name, zone_polygon in self.walkable_areas.items():
            match = escalator_pattern.match(zone_name)
            if match:
                level, esc_id, direction = match.groups()
                # ONLY match escalators for the current level
                if level != self.level_id:
                    logger.debug(
                        f"Skipping escalator zone {zone_name} (not for level {self.level_id})"
                    )
                    continue
                exit_name = f"escalator_{esc_id}_{direction}"
                if exit_name in self.evacuation_exits:
                    centroid = zone_polygon.centroid
                    self.exit_coordinates[exit_name] = (centroid.x, centroid.y)
                    logger.info(
                        f"Added escalator exit {exit_name} (from {zone_name}) at {centroid.x:.2f}, {centroid.y:.2f}"
                    )
                else:
                    # Only warn if this is actually supposed to be an escalator exit on this level
                    # For Level 0, finding L0_esc_* zones but not having them in evacuation_exits is expected
                    # because Level 0 uses street exits, not escalator exits
                    if level == self.level_id:
                        logger.debug(
                            f"Zone {zone_name} matches escalator pattern but exit_name '{exit_name}' not in evacuation_exits. Available: {list(self.evacuation_exits.keys())}"
                        )
                    else:
                        logger.debug(
                            f"Skipping escalator zone {zone_name} (not for level {self.level_id})"
                        )

        logger.info(
            f"Populated {len(self.exit_coordinates)} exit coordinates for level {self.level_id}: {list(self.exit_coordinates.keys())}"
        )

    def _create_convex_exit_from_polygon(
        self,
        exit_name: str,
        polygon: Any,
        walkable_geometry: Any,
    ) -> int | None:
        """
        Create a convex rectangular exit centered on polygon centroid.

        Args:
            exit_name: Name for the exit
            polygon: Entrance polygon to create exit from
            walkable_geometry: Combined walkable geometry for validation

        Returns:
            Exit ID if successful, None if exit couldn't be placed
        """
        # Find a valid centroid within walkable geometry
        centroid = polygon.centroid
        if not walkable_geometry.contains(centroid):
            intersection = polygon.intersection(walkable_geometry)
            if not intersection.is_empty:
                centroid = intersection.representative_point()
            else:
                logger.error(
                    f"Exit '{exit_name}': polygon doesn't intersect walkable geometry. "
                    f"Polygon bounds: {polygon.bounds}"
                )
                return None

        # Create exit with standard size
        exit_size = 4.0  # 4x4 meter exit (standard)
        exit_coords = [
            (centroid.x - exit_size / 2, centroid.y - exit_size / 2),
            (centroid.x + exit_size / 2, centroid.y - exit_size / 2),
            (centroid.x + exit_size / 2, centroid.y + exit_size / 2),
            (centroid.x - exit_size / 2, centroid.y + exit_size / 2),
        ]

        exit_id = self.stage_manager.create_exit_at_coordinates(exit_name, exit_coords)
        logger.info(
            f"Created {exit_size}m x {exit_size}m exit '{exit_name}' " f"at {centroid.coords[0]}"
        )
        return exit_id

    def get_default_exit(self) -> tuple[str, int, int]:
        """
        Get default exit for agents.

        On levels with street exits, returns the first available street exit.
        On platform levels, returns the first available escalator exit.

        Returns:
            Tuple of (exit_name, exit_id, journey_id)
        """
        if not self.evacuation_exits:
            # This should not happen since we create escalator exits on platform levels
            logger.error(
                "No evacuation exits available on this level! "
                "This indicates a configuration problem."
            )
            raise RuntimeError("No exits available for agent initialization")

        exit_name = list(self.evacuation_exits.keys())[0]
        exit_id = self.evacuation_exits[exit_name]
        journey_id = self.evacuation_journeys[exit_name]

        return exit_name, exit_id, journey_id

    def get_exit_ids(self, exit_name: str) -> tuple[int, int]:
        """
        Get stage and journey IDs for a specific exit.

        Args:
            exit_name: Name of the evacuation exit

        Returns:
            Tuple of (stage_id, journey_id)

        Raises:
            KeyError: If exit name not found
        """
        if exit_name not in self.evacuation_journeys:
            raise KeyError(
                f"Unknown exit: {exit_name}. "
                f"Available exits: {list(self.evacuation_journeys.keys())}"
            )

        stage_id = self.evacuation_exits[exit_name]
        journey_id = self.evacuation_journeys[exit_name]

        return stage_id, journey_id
