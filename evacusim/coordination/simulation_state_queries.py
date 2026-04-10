"""
Simulation state query utilities.

This module provides helper methods for querying the pedestrian simulation state,
including agent positions, nearby agents, and event history.
"""

from typing import Any

from evacusim.utils.logger import get_logger
from evacusim.jps.simulation_interface import PedestrianSimulation

logger = get_logger(__name__)


class SimulationStateQueries:
    """Helper class for querying simulation state."""

    def __init__(self, jps_simulation: PedestrianSimulation):
        """
        Initialize state queries.

        Args:
            jps_simulation: Pedestrian simulation instance (implements PedestrianSimulation)
        """
        self.jps_sim = jps_simulation

    def get_agent_position(self, agent_id: str) -> tuple[float, float] | None:
        """
        Get agent's current position from JuPedSim.

        Args:
            agent_id: Concordia agent ID

        Returns:
            Agent's (x, y) position, or None if agent has exited
        """
        return self.jps_sim.get_agent_position(agent_id)

    def get_nearby_agents(self, agent_id: str, radius: float) -> list[dict[str, Any]]:
        """
        Get information about nearby agents.

        Args:
            agent_id: Concordia agent ID
            radius: Search radius in meters

        Returns:
            List of nearby agent info dictionaries
        """
        return self.jps_sim.get_nearby_agents(agent_id, radius)

    def get_recent_events(
        self, event_history: list[dict[str, Any]], current_sim_time: float, count: int = 3
    ) -> list[str]:
        """
        Get recent events relevant to agents.

        Args:
            event_history: List of all events
            current_sim_time: Current simulation time
            count: Number of recent events to return

        Returns:
            List of event messages (only events that have already occurred)
        """
        # Return only events that have occurred (time <= current_sim_time)
        occurred_events = [e["message"] for e in event_history if e["time"] <= current_sim_time]
        # Return last N events
        return occurred_events[-count:]
