"""
Exit Tracker

Manages tracking of agents who have exited the simulation through evacuation exits.
Monitors agent positions and detects when agents are no longer present in the
pedestrian simulation, marking them as successfully evacuated.
"""

from typing import Any

from evacusim.utils.logger import get_logger
from evacusim.jps.simulation_interface import PedestrianSimulation

logger = get_logger(__name__)


class ExitTracker:
    """Tracks agents who have exited the simulation."""

    def __init__(
        self,
        concordia_agents: dict[str, Any],
        exited_agents: set[str],
        agent_destinations: dict[str, str],
        jps_sim: PedestrianSimulation,
        station_layout: dict[str, Any] | None = None,
        exit_validation_radius: float = 15.0,
        exit_log: list[dict[str, Any]] | None = None,
    ):
        """
        Initialize exit tracker.

        Args:
            concordia_agents: Dict of agent_id -> Concordia entity
            exited_agents: Set of agent IDs who have exited
            agent_destinations: Dict of agent_id -> current exit name
            jps_sim: Pedestrian simulation instance (implements PedestrianSimulation)
            station_layout: Station geometry and exit information for validation
            exit_validation_radius: Radius within which agent must be near exit to count as evacuated
            exit_log: Shared list that each evacuation is appended to (owned by the runner)
        """
        self.concordia_agents = concordia_agents
        self.exited_agents = exited_agents
        self.agent_destinations = agent_destinations
        self.jps_sim = jps_sim
        self.station_layout = station_layout
        self.exit_validation_radius = exit_validation_radius
        # Shared with HybridSimulationRunner so the runner can serialise it.
        self.exit_log = exit_log if exit_log is not None else []

        # Track last known positions for validation
        self.last_known_positions: dict[str, tuple[float, float]] = {}
        # Last level each agent was seen on. jps_sim.agent_levels drops the
        # agent before check_exited_agents runs, so the level has to be
        # snapshotted while the agent is still present or it is lost.
        self.last_known_levels: dict[str, str] = {}

    def check_exited_agents(self, current_sim_time: float, current_step: int):
        """
        Check for agents who have reached exits and mark them as exited.

        Validates that agents were actually near their target exit before
        marking them as successfully evacuated.

        Args:
            current_sim_time: Current simulation time in seconds
            current_step: Current simulation step number
        """
        # Get current agent positions from JuPedSim
        current_positions = self.jps_sim.get_all_agent_positions()

        # Update last known positions for all currently active agents
        tracked_levels_now = getattr(self.jps_sim, "agent_levels", {})
        for agent_id, position in current_positions.items():
            self.last_known_positions[agent_id] = position
            if agent_id in tracked_levels_now:
                self.last_known_levels[agent_id] = str(tracked_levels_now[agent_id])

        # Log agent count for debugging
        total_agents = len(self.concordia_agents)
        active_agents = len(current_positions)
        exited_count = len(self.exited_agents)

        # Find agents that are no longer in JuPedSim (they've exited)
        newly_exited = []
        failed_validations = []

        for agent_id in list(self.concordia_agents.keys()):
            if agent_id not in self.exited_agents and agent_id not in current_positions:
                # Multi-level transfers can produce a one-step gap between an
                # agent leaving one JuPedSim instance and becoming visible in
                # the destination instance. agent_levels remains authoritative
                # during that gap. Do not classify the agent as evacuated or it
                # will be excluded from every later decision after transfer.
                tracked_levels = getattr(self.jps_sim, "agent_levels", {})
                if agent_id in tracked_levels:
                    logger.debug(
                        f"{agent_id} temporarily absent during level transfer "
                        f"(tracked on level {tracked_levels[agent_id]})"
                    )
                    continue

                # Agent has disappeared - validate they actually reached an exit
                exit_name = self.agent_destinations.get(agent_id, "unknown")
                last_position = self.last_known_positions.get(agent_id)

                # Escalator exits are level transfers, not final evacuations.
                # The agent will reappear on the target level — do NOT add them to
                # exited_agents or they'll be excluded from the decision loop.
                if exit_name.startswith("escalator_"):
                    logger.debug(
                        f"🔄 {agent_id} exited through escalator '{exit_name}' "
                        f"— skipping final-evacuation marking (level transfer)"
                    )
                    continue

                if self._validate_exit_reached(agent_id, exit_name, last_position):
                    # Valid evacuation
                    self.exited_agents.add(agent_id)
                    newly_exited.append((agent_id, exit_name))
                else:
                    # Invalid - agent disappeared but wasn't near exit
                    failed_validations.append((agent_id, exit_name, last_position))
                    # Still mark as exited to prevent repeated checks, but log warning
                    self.exited_agents.add(agent_id)

        # Log newly exited agents
        for agent_id, exit_name in newly_exited:
            logger.info(f"✅ {agent_id} has evacuated through {exit_name}")
            self._record_exit(agent_id, exit_name, current_sim_time, validated=True)

        # Log validation failures
        for agent_id, exit_name, last_pos in failed_validations:
            self._record_exit(agent_id, exit_name, current_sim_time, validated=False)
            logger.warning(
                f"⚠️ {agent_id} disappeared but was NOT near exit '{exit_name}' "
                f"(last position: {last_pos}). Possible simulation error or geometry issue."
            )

        # Periodic status update every 50 steps
        if current_step % 50 == 0 and current_step > 0:
            logger.info(
                f"📊 Agent status: {active_agents} active, {exited_count} exited, "
                f"{total_agents} total (t={current_sim_time:.1f}s)"
            )

    def _record_exit(
        self, agent_id: str, intended_exit: str, sim_time: float, validated: bool
    ) -> None:
        """
        Append one evacuation to the shared exit log.

        ``intended_exit`` comes from ``agent_destinations``, which holds the
        agent's live *intent* and is cleared whenever it decides to wait — so it
        is frequently ``"unknown"`` and must not be treated as the outcome.  The
        exit actually used is resolved from geometry instead, which is what the
        agent's final position actually tells us.
        """
        last_position = self.last_known_positions.get(agent_id)
        resolved, distance = self._nearest_exit(last_position)
        self.exit_log.append(
            {
                "agent_id": agent_id,
                "exit_name": resolved if resolved is not None else intended_exit,
                "intended_exit": intended_exit,
                "exit_distance_m": round(distance, 2) if resolved is not None else "",
                "time_s": round(sim_time, 2),
                "level": self.last_known_levels.get(agent_id, ""),
                "x": round(last_position[0], 3) if last_position else "",
                "y": round(last_position[1], 3) if last_position else "",
                "validated": validated,
            }
        )

    def _nearest_exit(
        self, position: tuple[float, float] | None
    ) -> tuple[str | None, float]:
        """Return (exit_name, distance) for the exit nearest `position`."""
        if position is None or self.station_layout is None:
            return None, float("inf")
        best_name, best_dist = None, float("inf")
        for name, exit_position in (self.station_layout.get("exits") or {}).items():
            dx = position[0] - exit_position[0]
            dy = position[1] - exit_position[1]
            distance = (dx**2 + dy**2) ** 0.5
            if distance < best_dist:
                best_name, best_dist = name, distance
        return best_name, best_dist

    def _validate_exit_reached(
        self, agent_id: str, exit_name: str, last_position: tuple[float, float] | None
    ) -> bool:
        """
        Validate that an agent actually reached their target exit.

        Args:
            agent_id: ID of the agent
            exit_name: Name of the exit they were heading to
            last_position: Last known position of the agent

        Returns:
            True if validation passes, False if suspicious
        """
        # If we don't have last position or station layout, can't validate
        if last_position is None or self.station_layout is None:
            logger.debug(
                f"Cannot validate exit for {agent_id}: missing last_position or station_layout"
            )
            return True  # Assume valid if we can't validate

        # Get exit coordinates from station layout
        exits = self.station_layout.get("exits", {})
        if exit_name not in exits:
            logger.debug(f"Cannot validate exit for {agent_id}: exit '{exit_name}' not in layout")
            return True  # Assume valid if exit not in layout

        exit_position = exits[exit_name]

        # Calculate distance from last position to exit
        dx = last_position[0] - exit_position[0]
        dy = last_position[1] - exit_position[1]
        distance = (dx**2 + dy**2) ** 0.5

        # Agent should be within validation radius of the exit
        if distance <= self.exit_validation_radius:
            return True
        else:
            logger.debug(
                f"{agent_id} was {distance:.1f}m from exit '{exit_name}' "
                f"(threshold: {self.exit_validation_radius}m)"
            )
            return False
