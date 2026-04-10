"""
Stage management for JuPedSim simulations.

Handles creation of exits, waypoints, and journeys for routing agents through space.
Provides a higher-level API for managing JuPedSim stages and journeys.

Stages in JuPedSim:
    - Exits: Terminal stages where agents leave the simulation
    - Waypoints: Intermediate stages where agents pass through or wait

Journeys:
    - Sequences of stages that define an agent's path through the simulation
    - Agents follow journeys and can be switched to different journeys dynamically
"""

import jupedsim as jps
from shapely.geometry import Polygon

from evacusim.utils.logger import get_logger

logger = get_logger(__name__)


class StageManager:
    """Manages stages (exits, waypoints) and journeys in JuPedSim simulations."""

    def __init__(self, simulation: jps.Simulation) -> None:
        """
        Initialize stage manager.

        Args:
            simulation: JuPedSim simulation object
        """
        self.simulation: jps.Simulation = simulation
        self.exits: dict[str, int] = {}  # Map exit name -> stage ID
        self.waypoints: dict[str, int] = {}  # Map waypoint name -> stage ID
        self.journeys: dict[str, int] = {}  # Map journey name -> journey ID

    def create_exit_at_zone_centroid(
        self, zone_name: str, zone_polygon: Polygon, width: float = 30.0, height: float = 30.0
    ) -> int:
        """
        Create an exit stage at the centroid of a zone.

        JuPedSim requires convex polygons for exits. This creates a
        rectangular exit centered on the zone's centroid.

        Args:
            zone_name: Name of the zone (for tracking)
            zone_polygon: Polygon defining the zone
            width: Width of exit rectangle in meters
            height: Height of exit rectangle in meters

        Returns:
            Stage ID of created exit
        """
        centroid = zone_polygon.centroid

        # Create rectangular exit
        exit_coords = [
            (centroid.x - width / 2, centroid.y - height / 2),
            (centroid.x + width / 2, centroid.y - height / 2),
            (centroid.x + width / 2, centroid.y + height / 2),
            (centroid.x - width / 2, centroid.y + height / 2),
        ]

        stage_id = self.simulation.add_exit_stage(exit_coords)
        self.exits[zone_name] = stage_id

        logger.debug(f"Created exit stage: {zone_name} (id={stage_id})")
        logger.debug(f"  Exit location: ({centroid.x:.1f}, {centroid.y:.1f})")
        logger.debug(f"  Exit size: {width}m x {height}m")

        return stage_id  # type: ignore[no-any-return]

    def create_exit_at_coordinates(self, exit_name: str, coords: list[tuple[float, float]]) -> int:
        """
        Create an exit stage at specific coordinates.

        Args:
            exit_name: Name for the exit (for tracking)
            coords: List of (x, y) coordinate tuples defining exit polygon

        Returns:
            Stage ID of created exit
        """
        stage_id = self.simulation.add_exit_stage(coords)
        self.exits[exit_name] = stage_id

        logger.debug(f"Created exit stage: {exit_name} (id={stage_id})")

        return stage_id  # type: ignore[no-any-return]

    def create_waypoint(
        self, waypoint_name: str, coords: list[tuple[float, float]], distance: float = 1.0
    ) -> int:
        """
        Create a waypoint stage.

        Args:
            waypoint_name: Name for the waypoint (for tracking)
            coords: List of (x, y) coordinate tuples defining waypoint polygon
            distance: Distance threshold for considering waypoint reached

        Returns:
            Stage ID of created waypoint
        """
        stage_id = self.simulation.add_waypoint_stage(coords, distance)
        self.waypoints[waypoint_name] = stage_id

        logger.debug(f"Created waypoint stage: {waypoint_name} (id={stage_id})")

        return stage_id  # type: ignore[no-any-return]

    def create_waiting_stage(
        self, name: str, position: tuple[float, float], distance: float = 5.0
    ) -> int:
        """
        Create a waiting area (waypoint) at a specific position.
        Agents will walk to this position and wait (remain there).

        Args:
            name: Name for the waiting stage (for tracking)
            position: (x, y) center position for the waiting area
            distance: Distance threshold for considering stage reached (agents stop when within this distance)

        Returns:
            Stage ID of created waiting stage
        """
        # Waypoint takes a position tuple and distance threshold
        stage_id = self.simulation.add_waypoint_stage(position, distance)
        self.waypoints[name] = stage_id

        return stage_id  # type: ignore[no-any-return]

    def create_journey(self, journey_name: str, stage_ids: list[int]) -> int:
        """
        Create a journey through a sequence of stages.

        Args:
            journey_name: Name for the journey (for tracking)
            stage_ids: List of stage IDs in order

        Returns:
            Journey ID
        """
        journey = jps.JourneyDescription(stage_ids)
        journey_id = self.simulation.add_journey(journey)
        self.journeys[journey_name] = journey_id

        return journey_id  # type: ignore[no-any-return]

    def create_simple_exit_journey(self, journey_name: str, exit_id: int) -> int:
        """
        Create a simple journey that goes directly to an exit.

        Args:
            journey_name: Name for the journey (for tracking)
            exit_id: Stage ID of the exit

        Returns:
            Journey ID
        """
        return self.create_journey(journey_name, [exit_id])

    def get_exit_id(self, exit_name: str) -> int | None:
        """
        Get stage ID for a named exit.

        Args:
            exit_name: Name of the exit to retrieve

        Returns:
            Stage ID, or None if exit not found
        """
        return self.exits.get(exit_name)

    def get_waypoint_id(self, waypoint_name: str) -> int | None:
        """
        Get stage ID for a named waypoint.

        Args:
            waypoint_name: Name of the waypoint to retrieve

        Returns:
            Stage ID, or None if waypoint not found
        """
        return self.waypoints.get(waypoint_name)

    def get_journey_id(self, journey_name: str) -> int | None:
        """
        Get journey ID for a named journey.

        Args:
            journey_name: Name of the journey to retrieve

        Returns:
            Journey ID, or None if journey not found
        """
        return self.journeys.get(journey_name)
