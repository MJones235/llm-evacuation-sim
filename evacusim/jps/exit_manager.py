"""
Evacuation exit management for JuPedSim simulation.

Handles creation and management of evacuation exits and journey routing
to exits in the station environment.
"""

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
        exit_thresholds: dict[str, Any] | None = None,
        train_entrance_areas: dict[str, Any] | None = None,
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
        self.exit_thresholds = exit_thresholds or {}
        self.train_entrance_areas = train_entrance_areas or {}
        # Names of exits that represent train boarding points.  They are
        # registered in JuPedSim at init time but hidden from agent observations
        # until the corresponding train_arrival event fires in EventManager.
        self.train_exits: set[str] = set()

        # Setup evacuation exits and routes. Escalator boarding strips are
        # registered later by EscalatorSystem via register_exit().
        logger.info(
            f"Setting up evacuation exits for level {self.level_id}: "
            f"entrance_areas={list(entrance_areas.keys()) if entrance_areas else 'None'}"
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
        self._populate_train_exit_coordinates()

        logger.info(
            f"Created {len(self.evacuation_exits)} exits: {list(self.evacuation_exits.keys())}"
        )

    def _setup_evacuation_exits(
        self, walkable_geometry: Any
    ) -> tuple[dict[str, int], dict[str, int]]:
        """
        Create evacuation exit stages at street entrances and train doors.

        Escalator boarding strips are not created here: EscalatorSystem builds
        them and registers them with register_exit().

        Args:
            walkable_geometry: Combined walkable geometry for validation

        Returns:
            Tuple of (evacuation_exits dict, evacuation_journeys dict)
        """
        evacuation_exits = {}
        evacuation_journeys = {}

        if not self.entrance_areas:
            # No street exits (platform level): only train boarding exits here;
            # the escalators are registered by EscalatorSystem.
            return self._create_train_exits()

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

        return evacuation_exits, evacuation_journeys

    def _create_train_exits(self) -> tuple[dict[str, int], dict[str, int]]:
        """
        Create JuPedSim exit stages for train boarding areas.

        Train exits are registered in JuPedSim at simulation start so that
        pathfinding can route agents to them, but they are hidden from agent
        observations until the corresponding ``train_arrival`` event fires.
        Their names are stored in ``self.train_exits`` for the observation
        layer to use when filtering.

        Returns:
            Tuple of (evacuation_exits dict, evacuation_journeys dict)
        """
        evacuation_exits: dict[str, int] = {}
        evacuation_journeys: dict[str, int] = {}

        for exit_name, polygon in self.train_entrance_areas.items():
            coords = list(polygon.exterior.coords)[:-1]
            if len(coords) < 3:
                logger.warning(f"Train entrance '{exit_name}' has invalid polygon (<3 points), skipping.")
                continue
            try:
                exit_id = self.stage_manager.create_exit_at_coordinates(
                    exit_name=exit_name, coords=coords
                )
                journey_id = self.stage_manager.create_simple_exit_journey(
                    journey_name=f"journey_to_{exit_name}", exit_id=exit_id
                )
                evacuation_exits[exit_name] = exit_id
                evacuation_journeys[exit_name] = journey_id
                self.train_exits.add(exit_name)
                centroid = polygon.centroid
                logger.info(
                    f"Created train exit '{exit_name}' at ({centroid.x:.2f}, {centroid.y:.2f}) "
                    f"(exit={exit_id}, journey={journey_id})"
                )
            except Exception as e:
                logger.warning(f"Failed to create train exit '{exit_name}': {e}")

        return evacuation_exits, evacuation_journeys

    def _populate_exit_coordinates(self):
        """Populate exit_coordinates dict with exit center positions.

        For street exits that have a threshold polygon (corridor-mouth marker),
        the threshold centroid is used instead of the entrance polygon centroid.
        This makes distance and line-of-sight calculations reflect the point
        where the corridor meets the open concourse rather than the far end of
        the corridor, which is often occluded from most vantage points.
        """
        # Add entrance area coordinates (street exits)
        for entrance_name, entrance_polygon in self.entrance_areas.items():
            if entrance_name in self.evacuation_exits:
                if entrance_name in self.exit_thresholds:
                    ref_poly = self.exit_thresholds[entrance_name]
                    centroid = ref_poly.centroid
                    logger.info(
                        f"Using threshold for {entrance_name} at ({centroid.x:.2f}, {centroid.y:.2f})"
                        f" (entrance centroid was ({entrance_polygon.centroid.x:.2f}, {entrance_polygon.centroid.y:.2f}))"
                    )
                else:
                    centroid = entrance_polygon.centroid
                    logger.debug(
                        f"Added street exit {entrance_name} at {centroid.x:.2f}, {centroid.y:.2f}"
                    )
                self.exit_coordinates[entrance_name] = (centroid.x, centroid.y)

        logger.info(
            f"Populated {len(self.exit_coordinates)} exit coordinates for level {self.level_id}: {list(self.exit_coordinates.keys())}"
        )

    def _populate_train_exit_coordinates(self) -> None:
        """Add centroid coordinates for train boarding exits to exit_coordinates."""
        for exit_name, polygon in self.train_entrance_areas.items():
            if exit_name in self.evacuation_exits:
                centroid = polygon.centroid
                self.exit_coordinates[exit_name] = (centroid.x, centroid.y)
                logger.debug(
                    f"Added train exit coordinate for '{exit_name}': "
                    f"({centroid.x:.2f}, {centroid.y:.2f})"
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
        Train exits are skipped: they only exist while a train is in.

        Returns:
            Tuple of (exit_name, exit_id, journey_id)
        """
        if not self.evacuation_exits:
            logger.error(
                "No evacuation exits available on this level! "
                "This indicates a configuration problem."
            )
            raise RuntimeError("No exits available for agent initialization")

        # Train boarding exits only exist while a train is in; never a default.
        names = [n for n in self.evacuation_exits if n not in self.train_exits]
        exit_name = (names or list(self.evacuation_exits.keys()))[0]
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

    def register_exit(
        self, exit_name: str, stage_id: int, journey_id: int, coordinate: tuple[float, float]
    ) -> None:
        """Register an exit stage created elsewhere (escalator boarding strips)."""
        self.evacuation_exits[exit_name] = stage_id
        self.evacuation_journeys[exit_name] = journey_id
        self.exit_coordinates[exit_name] = coordinate
        logger.info(
            f"Registered exit '{exit_name}' on level {self.level_id} at "
            f"({coordinate[0]:.2f}, {coordinate[1]:.2f})"
        )
