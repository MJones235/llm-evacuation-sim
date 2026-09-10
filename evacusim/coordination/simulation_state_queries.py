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

        If an event entry contains a ``message_template`` key (i.e. the original
        message had a ``{elapsed_time}`` placeholder), the placeholder is resolved
        dynamically using the elapsed time since the event fired so that agents
        always receive an up-to-date duration phrase.

        Args:
            event_history: List of all events
            current_sim_time: Current simulation time
            count: Number of recent events to return

        Returns:
            List of event messages (only events that have already occurred)
        """
        occurred_messages: list[str] = []
        for e in event_history:
            if e["time"] > current_sim_time:
                continue
            template = e.get("message_template")
            if template:
                elapsed = current_sim_time - e["time"]
                minutes = round(elapsed / 60)
                if minutes < 1:
                    phrase = "less than a minute"
                elif minutes == 1:
                    phrase = "1 minute"
                else:
                    phrase = f"{minutes} minutes"
                msg = template.replace("{elapsed_time}", phrase)
            else:
                msg = e["message"]
            occurred_messages.append(msg)
        # Return last N events
        return occurred_messages[-count:]
