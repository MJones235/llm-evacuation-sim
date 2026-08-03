"""
JuPedSim simulation setup for Station Concordia simulations.

This module is responsible for:
- Initializing JuPedSim simulation instances
- Loading station geometry from network files
- Configuring simulation parameters
- Supporting single-level and multi-level simulations
"""

from pathlib import Path

from evacusim.utils.logger import get_logger
from evacusim.jps.jupedsim_integration import (
    ConcordiaJuPedSimulation,
)
from evacusim.jps.multi_level_simulation import (
    MultiLevelJuPedSimulation,
)
from evacusim.jps.simulation_interface import PedestrianSimulation

logger = get_logger(__name__)


class JuPedSimSetup:
    """Handles JuPedSim simulation initialization."""

    @staticmethod
    def create_simulation(config: dict) -> PedestrianSimulation:
        """
        Create and configure a JuPedSim simulation instance.

        Creates either a single-level or multi-level simulation based on config.

        Args:
            config: Configuration dictionary containing simulation settings

        Returns:
            Initialized simulation instance (ConcordiaJuPedSimulation or MultiLevelJuPedSimulation)
        """
        sim_config = config.get("simulation", {})
        dt = sim_config.get("dt", 0.05)
        network_path = Path(sim_config.get("network_path", "scenarios/station_sim/network"))

        # Collect exits that should be blocked before t=0.
        # Timed block_exit events are handled by EventManager when their
        # scheduled time is reached, so only explicit startup blocks (or
        # block_exit events at t<=0) are removed from navmesh at init.
        initially_blocked_exits: set[str] = set(
            sim_config.get("initially_blocked_exits", []) or []
        )
        for event in config.get("events", []):
            if event.get("type") != "block_exit":
                continue
            event_time = float(event.get("time", 0.0))
            if event_time > 0.0:
                continue
            exits = event.get("exits", [])
            if isinstance(exits, str):
                exits = [exits]
            for exit_name in exits:
                initially_blocked_exits.add(exit_name)
        if initially_blocked_exits:
            logger.info(f"Pre-blocking exits at simulation start: {initially_blocked_exits}")

        # Check if multi-level mode is enabled
        multi_level = sim_config.get("multi_level", False)
        levels = sim_config.get("levels", ["0", "-1"])

        if multi_level:
            logger.info(
                f"Loading multi-level station geometry from {network_path} "
                f"(levels: {', '.join(levels)})..."
            )
            escalator_belt_speed = sim_config.get("escalator_belt_speed", 0.5)
            level_arrival_waypoints = sim_config.get("level_arrival_waypoints", {})
            transfer_random_waypoint_min_distance_m = sim_config.get(
                "transfer_random_waypoint_min_distance_m", 10.0
            )
            jps_sim = MultiLevelJuPedSimulation(
                network_path=network_path,
                dt=dt,
                exit_radius=10.0,
                levels=levels,
                escalator_belt_speed=escalator_belt_speed,
                level_arrival_waypoints=level_arrival_waypoints,
                transfer_random_waypoint_min_distance_m=transfer_random_waypoint_min_distance_m,
                initially_blocked_exits=initially_blocked_exits,
            )
            logger.info("Multi-level JuPedSim simulation created successfully")
        else:
            # Single-level mode (backward compatible)
            level_id = sim_config.get("level_id", 0)
            logger.info(f"Loading station geometry from {network_path} (level {level_id})...")
            jps_sim = ConcordiaJuPedSimulation(
                network_path=network_path,
                dt=dt,
                exit_radius=10.0,
                level_id=level_id,
            )
            logger.info("JuPedSim simulation created successfully")

        return jps_sim
