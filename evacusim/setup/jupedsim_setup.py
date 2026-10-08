"""
JuPedSim simulation setup for Station Concordia simulations.

This module is responsible for:
- Initializing JuPedSim simulation instances
- Loading station geometry from network files
- Configuring simulation parameters
- Supporting single-level and multi-level simulations
"""

from pathlib import Path

from evacusim.config.schema import BlockExitEvent, RunConfig, as_dict
from evacusim.jps.jupedsim_integration import (
    ConcordiaJuPedSimulation,
)
from evacusim.jps.multi_level_simulation import (
    MultiLevelJuPedSimulation,
)
from evacusim.jps.simulation_interface import PedestrianSimulation
from evacusim.utils.logger import get_logger
from evacusim.utils.seeding import derive_seed

logger = get_logger(__name__)


class JuPedSimSetup:
    """Handles JuPedSim simulation initialization."""

    @staticmethod
    def create_simulation(params: RunConfig) -> PedestrianSimulation:
        """
        Create the pedestrian simulation for the station geometry.

        Creates either a single-level or multi-level simulation.

        Args:
            params: The run's parameters

        Returns:
            Initialized simulation instance (ConcordiaJuPedSimulation or MultiLevelJuPedSimulation)
        """
        sim = params.simulation
        network_path = Path(sim.network_path)

        # Exits closed from the start: explicit startup blocks plus block_exit
        # events at t <= 0. Later block_exit events are applied by the
        # EventManager when their time comes.
        initially_blocked_exits = set(sim.initially_blocked_exits)
        for event in params.events:
            if isinstance(event, BlockExitEvent) and event.time <= 0.0:
                initially_blocked_exits.update(event.exits)
        if initially_blocked_exits:
            logger.info(f"Pre-blocking exits at simulation start: {initially_blocked_exits}")

        if sim.multi_level:
            logger.info(
                f"Loading multi-level station geometry from {network_path} "
                f"(levels: {', '.join(sim.levels)})..."
            )
            jps_sim = MultiLevelJuPedSimulation(
                network_path=network_path,
                dt=sim.dt,
                exit_radius=10.0,
                levels=sim.levels,
                initially_blocked_exits=initially_blocked_exits,
                escalator_config=as_dict(sim.escalators) if sim.escalators else None,
                escalator_seed=derive_seed(params.seed, "escalators"),
            )
            jps_sim.clock_offset_s = sim.start_time_s
            logger.info("Multi-level JuPedSim simulation created successfully")
        else:
            logger.info(f"Loading station geometry from {network_path} (level {sim.level_id})...")
            jps_sim = ConcordiaJuPedSimulation(
                network_path=network_path,
                dt=sim.dt,
                exit_radius=10.0,
                level_id=sim.level_id,
            )
            logger.info("JuPedSim simulation created successfully")

        return jps_sim
