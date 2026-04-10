"""
Event management for Station Concordia simulations.

Handles:
- Event triggering based on test scenarios
- Exit blocking (physical obstacles + agent discovery)
- Event broadcasting to agents
- Event history tracking
"""

from typing import Any

from evacusim.utils.logger import get_logger
from evacusim.jps.simulation_interface import PedestrianSimulation

logger = get_logger(__name__)


class EventManager:
    """
    Manages simulation events like exit blocking and announcements.

    Responsible for:
    - Triggering scheduled events (exit blocking at specific times)
    - Broadcasting announcements to all agents
    - Tracking event history
    - Managing blocked exits state
    """

    def __init__(self, station_layout: dict[str, Any], jps_sim: PedestrianSimulation):
        """
        Initialize the event manager.

        Args:
            station_layout: Station geometry and exit information
            jps_sim: Pedestrian simulation instance (implements PedestrianSimulation)
        """
        self.station_layout = station_layout
        self.jps_sim = jps_sim

        # Event state
        self.event_history: list[dict[str, Any]] = []
        self.blocked_exits: set[str] = set()

        # Test scenario tracking
        self.test_block_exit_time: float | None = None
        self.test_block_exit_name: str | None = None
        self._test_exit_blocked: bool = False

    def setup_test_scenario(self, test_scenarios: dict[str, Any] | None) -> None:
        """
        Setup test scenarios for event triggering.

        Args:
            test_scenarios: Dictionary with test scenario configuration
                e.g., {"block_exit": {"exit_name": "north", "time": 30.0}}
        """
        if not test_scenarios:
            return

        # Phase 4.2: Exit blocking test
        if "blocked_exit" in test_scenarios:
            block_config = test_scenarios["blocked_exit"]
            # Check if scenario is enabled
            if not block_config.get("enabled", False):
                return

            self.test_block_exit_name = block_config.get("blocked_exit")
            self.test_block_exit_time = block_config.get("block_time")
            if self.test_block_exit_name and self.test_block_exit_time:
                logger.info(
                    f"Test scenario configured: Block '{self.test_block_exit_name}' "
                    f"exit at t={self.test_block_exit_time}s"
                )

    def check_and_trigger_events(self, current_sim_time: float) -> None:
        """
        Check for and trigger simulation events.

        Args:
            current_sim_time: Current simulation time in seconds
        """
        # Phase 4.2: Check for exit blocking test scenario
        if self.test_block_exit_time and not self._test_exit_blocked:
            # Check if we've reached the blocking time (within one timestep)
            if current_sim_time >= self.test_block_exit_time:
                # Trigger the blocking
                self.block_exit(self.test_block_exit_name)
                self._test_exit_blocked = True  # Flag to prevent repeated blocking

    def block_exit(self, exit_name: str) -> None:
        """
        Block an exit by placing a physical obstacle in JuPedSim.

        Agents will discover the blockage when they get close (visual range ~20m)
        or observe others turning back from it.

        Args:
            exit_name: Name of the exit to block
        """
        if exit_name not in self.station_layout["exits"]:
            logger.warning(f"Cannot block unknown exit: {exit_name}")
            return

        exit_pos = self.station_layout["exits"][exit_name]

        # Add to blocked exits set (for observations)
        self.blocked_exits.add(exit_name)

        # Place physical obstacle in JuPedSim
        # Use a radius that blocks the entrance (typically 2-3m wide, so 3-4m radius covers it)
        # This makes the exit unreachable in pathfinding - agents cannot get close enough
        # to evacuate through it, and will naturally reroute when they observe the blockage
        try:
            # Obstacle radius sized for typical entrance width (2-3m)
            obstacle_radius = 4.0
            self.jps_sim.add_obstacle(exit_pos, radius=obstacle_radius)
            logger.info(
                f"🚧 Exit {exit_name} physically blocked at {exit_pos} "
                f"(obstacle radius: {obstacle_radius}m)"
            )
        except Exception as e:
            logger.warning(f"Failed to add physical obstacle at {exit_name}: {e}")

        # NO announcement - agents discover naturally through observation
        # NOTE: We do NOT immediately reroute agents heading to this exit.
        # Agents should be allowed to travel to the blocked exit and discover it naturally,
        # then they will observe the blockage and choose alternative routes in their next decision.
        logger.info(f"Exit {exit_name} blocked - agents will discover naturally")

    def broadcast_event(
        self, event_message: str, current_sim_time: float, agents: dict[str, Any]
    ) -> None:
        """
        Broadcast an event to all agents.

        Args:
            event_message: The event message to broadcast
            current_sim_time: Current simulation time
            agents: Dictionary of agent_id -> agent entity
        """
        logger.info(f"Broadcasting event: {event_message}")

        # Store event
        self.event_history.append(
            {
                "time": current_sim_time,
                "message": event_message,
            }
        )

        # Notify all agents
        for agent in agents.values():
            agent.observe(f"[ANNOUNCEMENT] {event_message}")
