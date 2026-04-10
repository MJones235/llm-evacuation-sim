"""
Pedestrian simulation interface for station_concordia.

This module defines the abstract interface that any pedestrian simulation
backend must implement to work with the station_concordia hybrid simulation.

This allows swapping out JuPedSim for alternative simulation engines without
modifying the rest of the codebase.
"""

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class PedestrianSimulation(Protocol):
    """
    Interface for pedestrian simulation backends.

    This protocol defines the contract that any pedestrian simulation
    implementation must satisfy to integrate with the Concordia evacuation
    simulation system.

    Implementations must provide:
    - Core simulation stepping and time tracking
    - Agent lifecycle management (add, position queries)
    - Agent behavior control (targets, speeds, exits)
    - Environment geometry access (walkable areas, exits, obstacles)

    Example implementations:
    - ConcordiaJuPedSimulation (JuPedSim backend)
    - Future: Social force model, ORCA, etc.
    """

    # Core properties
    dt: float  # Timestep in seconds

    # Core simulation control methods
    def step(self) -> bool:
        """
        Advance simulation by one timestep.

        Returns:
            True if simulation should continue, False if complete
            (e.g., all agents have exited)
        """
        ...

    def get_simulation_time(self) -> float:
        """
        Get current simulation time in seconds.

        Returns:
            Current simulation time
        """
        ...

    # Agent management methods
    def add_agent(
        self, agent_id: str, position: tuple[float, float], walking_speed: float = 1.34
    ) -> None:
        """
        Add an agent to the simulation.

        Args:
            agent_id: Unique identifier for the agent (Concordia agent ID)
            position: Initial (x, y) position in meters
            walking_speed: Desired walking speed in m/s (default: 1.34 m/s)

        Raises:
            Exception: If agent cannot be added (invalid position, etc.)
        """
        ...

    def get_agent_position(self, agent_id: str) -> tuple[float, float] | None:
        """
        Get agent's current position.

        Args:
            agent_id: Concordia agent ID

        Returns:
            Agent's (x, y) position in meters, or (0.0, 0.0) if agent has exited
        """
        ...

    def get_all_agent_positions(self) -> dict[str, tuple[float, float]]:
        """
        Get positions of all active agents.

        Returns:
            Dictionary mapping agent IDs to (x, y) positions in meters
        """
        ...

    def get_nearby_agents(self, agent_id: str, radius: float) -> list[dict[str, Any]]:
        """
        Get information about agents within radius of given agent.

        Args:
            agent_id: ID of the querying agent
            radius: Search radius in meters

        Returns:
            List of nearby agent info dictionaries, each containing:
                - 'agent_id': str - Agent identifier
                - 'distance': float - Distance in meters
                - 'position': tuple[float, float] - (x, y) position
        """
        ...

    # Agent control methods
    def set_agent_target(self, agent_id: str, target: tuple[float, float]) -> None:
        """
        Set an agent's movement target by creating a waypoint.

        Args:
            agent_id: Concordia agent ID
            target: Target (x, y) position in meters

        Raises:
            KeyError: If agent is not in the simulation (may have exited)
            Exception: If target cannot be set (unreachable position, etc.)
        """
        ...

    def set_agent_evacuation_exit(self, agent_id: str, exit_name: str) -> None:
        """
        Direct an agent to a specific evacuation exit.

        Args:
            agent_id: Concordia agent ID
            exit_name: Name of the evacuation exit

        Raises:
            KeyError: If agent or exit is not found
            Exception: If exit routing cannot be established
        """
        ...

    def set_agent_speed(self, agent_id: str, speed: float) -> None:
        """
        Set an agent's walking speed mid-simulation.

        Args:
            agent_id: Concordia agent ID
            speed: Walking speed in m/s

        Raises:
            KeyError: If agent is not in the simulation
            ValueError: If speed is invalid (negative, etc.)
        """
        ...

    # Environment geometry properties
    @property
    def walkable_areas(self) -> dict[str, Any]:
        """
        Get walkable area polygons.

        Returns:
            Dictionary mapping area names to polygon geometries
        """
        ...

    @property
    def walkable_areas_with_obstacles(self) -> dict[str, Any]:
        """
        Get walkable area polygons with obstacles integrated.

        Returns:
            Dictionary mapping area names to polygon geometries with holes
        """
        ...

    @property
    def entrance_areas(self) -> dict[str, Any]:
        """
        Get entrance/exit area polygons.

        Returns:
            Dictionary mapping entrance names to polygon geometries
        """
        ...

    @property
    def platform_areas(self) -> dict[str, Any]:
        """
        Get platform area polygons.

        Returns:
            Dictionary mapping platform names to polygon geometries
        """
        ...

    @property
    def obstacles(self) -> list[Any]:
        """
        Get obstacle geometries.

        Returns:
            List of obstacle polygon geometries
        """
        ...

    @property
    def evacuation_exits(self) -> dict[str, int]:
        """
        Get evacuation exit identifiers.

        Returns:
            Dictionary mapping exit names to internal exit IDs
        """
        ...

    @property
    def evacuation_journeys(self) -> dict[str, int]:
        """
        Get evacuation journey identifiers.

        Returns:
            Dictionary mapping exit names to internal journey IDs
        """
        ...

    # Utility methods for setup
    def generate_spawn_positions(
        self, num_agents: int, seed: int = 42
    ) -> list[tuple[float, float]]:
        """
        Generate spawn positions for agents within the walkable geometry.

        Args:
            num_agents: Number of spawn positions to generate
            seed: Random seed for reproducibility

        Returns:
            List of (x, y) coordinate tuples for spawn positions

        Raises:
            RuntimeError: If unable to generate spawn positions
        """
        ...
