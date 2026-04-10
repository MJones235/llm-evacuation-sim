"""
Spawn position management for Station Concordia simulations.

This module is responsible for:
- Delegating spawn position generation to the simulation backend
- Providing a simple interface for generating spawn positions
"""

from evacusim.utils.logger import get_logger
from evacusim.jps.simulation_interface import PedestrianSimulation

logger = get_logger(__name__)


class SpawnManager:
    """Handles generation of spawn positions for agents."""

    @staticmethod
    def generate_spawn_positions(
        jps_sim: PedestrianSimulation,
        num_agents: int,
        seed: int = 42,
    ) -> list[tuple[float, float]]:
        """
        Generate spawn positions for agents within the geometry.

        Delegates to the simulation backend which handles geometry-specific
        spawn position generation.

        Args:
            jps_sim: Pedestrian simulation instance (implements PedestrianSimulation)
            num_agents: Number of spawn positions to generate
            seed: Random seed for reproducibility

        Returns:
            List of (x, y) coordinate tuples for spawn positions

        Raises:
            RuntimeError: If unable to generate spawn positions
        """
        logger.info(f"Generating {num_agents} spawn positions...")

        # Delegate to simulation backend
        spawn_positions = jps_sim.generate_spawn_positions(num_agents, seed)

        logger.info(f"Generated {len(spawn_positions)} spawn positions")

        return spawn_positions
