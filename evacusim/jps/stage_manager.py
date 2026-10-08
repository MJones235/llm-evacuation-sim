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
