"""
Agent tracking and management for JuPedSim simulation.

Tracks agent state, positions, and provides spatial queries for nearby agents.
"""

from typing import Any

import jupedsim as jps

from evacusim.utils.logger import get_logger

logger = get_logger(__name__)


class AgentTracker:
    """
    Tracks agent state and provides spatial queries.

    Handles:
    - Agent ID mapping (Concordia ID <-> JuPedSim ID)
    - Position queries for individual and all agents
    - Nearby agent queries with radius-based filtering
    - Agent target tracking
    """

    def __init__(self, simulation: jps.Simulation):
        """
        Initialize agent tracker.

        Args:
            simulation: JuPedSim simulation instance
        """
        self.simulation = simulation

        # Agent tracking
        self.agent_ids: dict[str, int] = {}  # Concordia ID -> JuPedSim ID
        self.jps_to_concordia: dict[int, str] = {}  # JuPedSim ID -> Concordia ID
        self.agent_targets: dict[str, tuple[float, float]] = {}

    def add_agent(
        self,
        agent_id: str,
        jps_id: int,
    ) -> None:
        """
        Register an agent in the tracker.

        Args:
            agent_id: Concordia agent ID
            jps_id: JuPedSim agent ID
        """
        self.agent_ids[agent_id] = jps_id
        self.jps_to_concordia[jps_id] = agent_id

        logger.debug(f"Registered agent {agent_id} (JPS ID: {jps_id})")

    def set_target(self, agent_id: str, target: tuple[float, float]) -> None:
        """
        Record agent's target position.

        Args:
            agent_id: Concordia agent ID
            target: Target (x, y) position
        """
        self.agent_targets[agent_id] = target

    def get_position(self, agent_id: str) -> tuple[float, float] | None:
        """
        Get agent's current position.

        Args:
            agent_id: Concordia agent ID

        Returns:
            Agent's (x, y) position, or None if agent has exited
        """
        if agent_id not in self.agent_ids:
            # Agent may have exited - this is expected
            return None

        jps_id = self.agent_ids[agent_id]

        # Find agent in current simulation state
        for agent in self.simulation.agents():
            if agent.id == jps_id:
                return (float(agent.position[0]), float(agent.position[1]))

        # Agent not found - likely exited (expected condition)
        return None

    def get_all_positions(self) -> dict[str, tuple[float, float]]:
        """
        Get positions of all agents for visualization.

        Returns:
            Dictionary mapping Concordia agent IDs to (x, y) positions
        """
        positions = {}

        for agent in self.simulation.agents():
            concordia_id = self.jps_to_concordia.get(agent.id)
            if concordia_id:
                positions[concordia_id] = (
                    float(agent.position[0]),
                    float(agent.position[1]),
                )

        return positions

    def get_nearby_agents(self, agent_id: str, radius: float) -> list[dict[str, Any]]:
        """
        Get information about agents within radius of given agent.

        Args:
            agent_id: ID of the querying agent
            radius: Search radius in meters

        Returns:
            List of nearby agent info dictionaries with id, distance, position, is_moving
        """
        if agent_id not in self.agent_ids:
            return []

        center_pos = self.get_position(agent_id)
        if center_pos is None:
            return []

        nearby = []

        # Get all agents currently in simulation
        all_agents = list(self.simulation.agents())

        for agent in all_agents:
            # Skip self
            other_id = self.jps_to_concordia.get(agent.id)
            if not other_id or other_id == agent_id:
                continue

            # Calculate distance
            dx = agent.position[0] - center_pos[0]
            dy = agent.position[1] - center_pos[1]
            dist = (dx**2 + dy**2) ** 0.5

            if dist <= radius:
                # Determine if moving based on orientation (has target)
                is_moving = hasattr(agent, "orientation") and agent.orientation is not None

                nearby.append(
                    {
                        "id": other_id,
                        "distance": dist,
                        "position": (float(agent.position[0]), float(agent.position[1])),
                        "is_moving": is_moving,
                        "target_exit": None,  # Will be enriched by HybridSimulationRunner
                    }
                )

        return nearby

    def is_agent_active(self, agent_id: str) -> bool:
        """
        Check if agent is still in simulation.

        Args:
            agent_id: Concordia agent ID

        Returns:
            True if agent is active, False if exited
        """
        if agent_id not in self.agent_ids:
            return False

        jps_id = self.agent_ids[agent_id]

        # Check if agent exists in simulation
        for agent in self.simulation.agents():
            if agent.id == jps_id:
                return True

        return False

    def get_jps_id(self, agent_id: str) -> int | None:
        """
        Get JuPedSim ID for a Concordia agent.

        Args:
            agent_id: Concordia agent ID

        Returns:
            JuPedSim ID if agent exists, None otherwise
        """
        return self.agent_ids.get(agent_id)

    def get_concordia_id(self, jps_id: int) -> str | None:
        """
        Get Concordia ID for a JuPedSim agent.

        Args:
            jps_id: JuPedSim agent ID

        Returns:
            Concordia ID if agent exists, None otherwise
        """
        return self.jps_to_concordia.get(jps_id)

    def remove_agent(self, agent_id: str) -> None:
        """Remove an agent from tracking after they exit simulation."""
        jps_id = self.agent_ids.pop(agent_id, None)
        if jps_id is not None:
            self.jps_to_concordia.pop(jps_id, None)
        self.agent_targets.pop(agent_id, None)
