"""
Multi-level JuPedSim simulation for Monument Station.

Manages multiple levels (concourse + platforms) and agent transfers between them
via escalators. Each level has its own JuPedSim simulation instance.
"""

from pathlib import Path
from typing import Any

from evacusim.conventions import platform_zone
from evacusim.escalators.spec_loader import build_specs
from evacusim.escalators.system import EscalatorSystem
from evacusim.jps.jupedsim_integration import (
    ConcordiaJuPedSimulation,
)
from evacusim.utils.logger import get_logger

logger = get_logger(__name__)


class MultiLevelJuPedSimulation:
    """
    Manages multiple JuPedSim simulations for multi-level stations.

    Each level has its own simulation instance, and agents can transfer
    between levels via escalators/stairs.
    """

    def __init__(
        self,
        network_path: Path,
        dt: float = 0.05,
        exit_radius: float = 10.0,
        levels: list[str] | None = None,
        initially_blocked_exits: set[str] | None = None,
        escalator_config: dict[str, Any] | None = None,
        escalator_seed: int = 0,
        platform_level: str = "-1",
    ):
        """
        Initialize multi-level simulation.

        Args:
            network_path: Path to network directory containing level_*.xml files
            dt: Timestep in seconds
            exit_radius: Radius of circular exits in meters
            levels: List of level IDs to load (default: ["0", "-1"])
            initially_blocked_exits: Exits that are blocked from simulation start.
                Escalators among them are built closed and not offered as exits.
            escalator_config: ``simulation.escalators`` config (defaults and
                per-escalator overrides) for the conveyor model.
            escalator_seed: Seed for lane choice and stander step gaps.
            platform_level: Level id of the train platforms.
        """
        self.dt = dt
        self.exit_radius = exit_radius
        self.network_path = Path(network_path)
        self.current_step = 0
        self.is_complete = False
        # Absolute time of step 0 (runs may start mid-day); set by the factory.
        self.clock_offset_s = 0.0

        if levels is None:
            levels = ["0", "-1"]
        self.levels = levels
        self.platform_level = platform_level

        _initially_blocked = set(initially_blocked_exits or [])

        # Create simulation instance for each level
        self.simulations: dict[str, ConcordiaJuPedSimulation] = {}
        for level_id in levels:
            logger.info(f"Initializing level {level_id}...")
            self.simulations[level_id] = ConcordiaJuPedSimulation(
                network_path=network_path,
                dt=dt,
                exit_radius=exit_radius,
                level_id=level_id,
                initially_blocked_exits=_initially_blocked,
            )

        # Level of every agent on a floor. Agents riding an escalator are on
        # no floor and are absent here; see is_agent_in_transit().
        self.agent_levels: dict[str, str] = {}
        # Each agent's own walking speed, kept across escalator rides.
        self.agent_base_speed: dict[str, float] = {}
        self.recently_transferred_agents: set[str] = set()

        # Post-transfer escape waypoints: the escalator's egress target assigned
        # when a rider steps off, so they clear the landing before deciding.
        # The decision processor defers prompts until the agent reaches it.
        self.transfer_escape_waypoints: dict[str, tuple[float, float]] = {}
        # Boarders continue from the local egress to their assigned platform
        # before the decision engine is allowed to choose "wait for train".
        self.transfer_platform_waypoints: dict[str, tuple[float, float]] = {}

        # Exits currently blocked by scenario events.
        self.blocked_exits: set[str] = set(_initially_blocked)
        # Agents forced to re-decide this step (e.g. released from a closed
        # escalator's queue). hybrid_simulation consumes this set.
        self.agents_needing_redecision: set[str] = set()

        specs = build_specs(self.network_path, self.levels, escalator_config)
        self.escalator_system = EscalatorSystem(
            self, specs, seed=escalator_seed, closed_exits=_initially_blocked
        )

        logger.info(
            f"Multi-level simulation initialized with {len(self.simulations)} levels: "
            f"{', '.join(levels)}"
        )

    # ------------------------------------------------------------------

    def get_agent_speed(self, agent_id: str) -> float | None:
        """
        Get an agent's current desired walking speed.

        Args:
            agent_id: Concordia agent ID

        Returns:
            Agent's desired speed in m/s, or None if agent has exited / unknown level.
        """
        if agent_id not in self.agent_levels:
            return None
        level_id = self.agent_levels[agent_id]
        return self.simulations[level_id].get_agent_speed(agent_id)

    # ------------------------------------------------------------------

    @property
    def geometry_manager(self):
        """
        Get geometry manager from level 0 (concourse level).

        For multi-level simulations, this exposes the concourse geometry
        which contains the street exits and main walkable areas.

        Returns:
            Geometry manager from level 0
        """
        return self.simulations["0"].geometry_manager

    def add_agent(
        self,
        agent_id: str,
        position: tuple[float, float],
        walking_speed: float = 1.34,
        level_id: str = "0",
        assign_default_destination: bool = True,
    ) -> None:
        """
        Add an agent to a specific level.

        Args:
            agent_id: Concordia agent ID
            position: Initial (x, y) position
            walking_speed: Desired walking speed in m/s
            level_id: Level to spawn on (default: "0")
            assign_default_destination: If True, spawn with the level's default
                destination exit. If False, spawn with no implicit destination.
        """
        if level_id not in self.simulations:
            raise ValueError(f"Level {level_id} not loaded")

        self.simulations[level_id].add_agent(
            agent_id,
            position,
            walking_speed,
            assign_default_destination=assign_default_destination,
        )
        self.agent_levels[agent_id] = level_id
        self.agent_base_speed.setdefault(agent_id, walking_speed)

        # Adding an agent means the simulation is no longer complete.  The
        # per-level step() latches is_complete when a level empties (and stays
        # latched); the aggregate flag latches likewise.  Reset the aggregate
        # here so a run that had briefly emptied (e.g. calibration between
        # sparse arrivals) resumes stepping once re-populated.  The per-level
        # flag is reset inside the level's own add_agent.
        self.is_complete = False

        logger.info(f"Added agent {agent_id} to level {level_id} at {position}")

    def step(self) -> bool:
        """
        Advance all levels and escalators by one timestep.

        Returns:
            True if simulation should continue, False if complete
        """
        if self.is_complete:
            return False

        now = self.current_time_s()

        # 1. Agents JuPedSim removed last step: street exits leave the station,
        #    escalator boarding strips put the agent on a conveyor.
        self._process_escalator_exits()

        # 2. Conveyors move; riders reaching the far comb step onto the other
        #    level; queues advance and admit the next boarders.
        self.escalator_system.step(self.dt, now)

        # 3. Step each level's floor simulation.
        any_active = False
        for sim in self.simulations.values():
            if sim.step():
                any_active = True

        self.current_step += 1

        # Count agents still registered on a floor, not just those JuPedSim is
        # stepping: an agent removed at a boarding strip this step is boarded
        # at the start of the next one, so the run is not over yet.
        total_agents = sum(len(sim.agent_tracker.agent_ids) for sim in self.simulations.values())
        if total_agents == 0 and not self.escalator_system.riding:
            logger.info("All agents have exited the simulation")
            self.is_complete = not any_active
            return False

        return True

    def _process_escalator_exits(self):
        """Hand agents removed at an escalator boarding strip to the escalator system."""
        now = self.current_time_s()
        for level_id, sim in self.simulations.items():
            exited_agents = sim.check_exits()
            if exited_agents:
                logger.info(
                    f"Level {level_id}: {len(exited_agents)} agents exited - {exited_agents}"
                )
            for agent_id, exit_name in exited_agents.items():
                if self.escalator_system.handles(exit_name):
                    self.escalator_system.board(agent_id, exit_name, now)
                    continue
                logger.info(f"Agent {agent_id} exited station through {exit_name}")
                self.agent_levels.pop(agent_id, None)

    def current_time_s(self) -> float:
        """Absolute simulation clock (seconds since midnight for calibration runs)."""
        return self.clock_offset_s + self.current_step * self.dt

    def is_agent_in_transit(self, agent_id: str) -> bool:
        """True while an agent is on an escalator (on no floor, but not gone)."""
        return self.escalator_system.is_in_transit(agent_id)

    def block_escalator(self, exit_name: str) -> None:
        """Close an escalator mid-run: no boarding, queue released, riders finish."""
        if not self.escalator_system.handles(exit_name):
            return
        self.blocked_exits.add(exit_name)
        self.escalator_system.close(exit_name)
        entry = self.escalator_system.escalators[exit_name]
        gm = self.simulations[entry.spec.from_level].geometry_manager
        gm.blocked_exit_positions[exit_name] = entry.landing_point

    def consume_recently_transferred_agents(self) -> set[str]:
        """Return and clear agents transferred since last consume call."""
        transferred = set(self.recently_transferred_agents)
        self.recently_transferred_agents.clear()
        return transferred

    def get_agent_position(self, agent_id: str) -> tuple[float, float] | None:
        """
        Get agent's current position.

        Args:
            agent_id: Concordia agent ID

        Returns:
            Agent's (x, y) position, or None if agent has exited
        """
        if agent_id not in self.agent_levels:
            return None

        level_id = self.agent_levels[agent_id]
        return self.simulations[level_id].get_agent_position(agent_id)

    def get_agent_level(self, agent_id: str) -> str | None:
        """Get the level an agent is currently on."""
        return self.agent_levels.get(agent_id)

    def set_agent_target(self, agent_id: str, target: tuple[float, float]) -> None:
        """Set an agent's movement target on their current level."""
        if agent_id not in self.agent_levels:
            return

        level_id = self.agent_levels[agent_id]
        self.simulations[level_id].set_agent_target(agent_id, target)

    def get_route_distance(
        self,
        agent_id: str,
        start: tuple[float, float],
        target: tuple[float, float],
    ) -> float | None:
        """Return navigable distance on the agent's current level."""
        level_id = self.agent_levels.get(agent_id)
        if level_id is None:
            return None
        return self.simulations[level_id].get_route_distance(start, target)

    def set_agent_destination_exit(self, agent_id: str, exit_name: str) -> None:
        """
        Direct an agent to a named exit on their current level.

        Escalator exits are handled by the escalator system: the agent joins
        that escalator's queue on this level. Escalators only board from their
        entry level, so a request for an escalator that does not board here
        (e.g. an up-escalator from the concourse) is refused with an error and
        the agent keeps its current journey.

        Args:
            agent_id: ID of the agent
            exit_name: Name of the exit - must exist on (or board from) the current level
        """
        if agent_id not in self.agent_levels:
            return

        level_id = self.agent_levels[agent_id]
        level_sim = self.simulations[level_id]

        if self.escalator_system.handles(exit_name):
            if exit_name not in self.escalator_system.exits_from_level(level_id):
                logger.error(
                    f"[DIRECTION VIOLATION] {agent_id} on level {level_id} requested "
                    f"escalator '{exit_name}', which does not board here. Request refused."
                )
                return
            self.escalator_system.assign(agent_id, exit_name, self.current_time_s())
            return

        if exit_name not in level_sim.exit_manager.evacuation_exits:
            # Raise KeyError so callers must surface and handle invalid exit
            # choices explicitly rather than silently rerouting.
            raise KeyError(
                f"Agent {agent_id} on level {level_id} tried to route to exit '{exit_name}' "
                f"which doesn't exist on this level. Available exits: "
                f"{list(level_sim.exit_manager.evacuation_exits.keys())}"
            )

        level_sim.set_agent_destination_exit(agent_id, exit_name)

    def set_agent_speed(self, agent_id: str, speed: float) -> None:
        """Set an agent's walking speed."""
        if agent_id not in self.agent_levels:
            return

        level_id = self.agent_levels[agent_id]
        self.simulations[level_id].set_agent_speed(agent_id, speed)

    def get_nearby_agents(self, agent_id: str, radius: float) -> list[dict[str, Any]]:
        """Get information about agents within radius on the same level."""
        if agent_id not in self.agent_levels:
            return []

        level_id = self.agent_levels[agent_id]
        return self.simulations[level_id].get_nearby_agents(agent_id, radius)

    def get_all_nearby_agents_bulk(self, radius: float) -> dict[str, list[dict[str, Any]]]:
        """
        Return nearby-agent lists for ALL agents in a single pass per level.

        Agents on different levels cannot see each other, so the bulk computation
        is run independently per level and results are merged.

        Args:
            radius: Search radius in metres

        Returns:
            Mapping agent_id -> list of nearby-agent info dicts
        """
        result: dict[str, list[dict[str, Any]]] = {}
        for sim in self.simulations.values():
            result.update(sim.get_all_nearby_agents_bulk(radius))
        return result

    def get_all_agent_positions(self) -> dict[str, tuple[float, float]]:
        """
        Get positions of all agents across all levels.

        Returns:
            Dictionary mapping agent IDs to (x, y) positions
        """
        all_positions = {}
        for sim in self.simulations.values():
            positions = sim.get_all_agent_positions()
            all_positions.update(positions)
        return all_positions

    def get_geometry(self, level_id: str | None = None) -> dict[str, Any]:
        """
        Get geometry information for visualization.

        Args:
            level_id: Specific level to get geometry for, or None for all levels

        Returns:
            Geometry data for the requested level(s)
        """
        if level_id is not None:
            return self.simulations[level_id].get_geometry()

        # Return all levels
        all_geometry = {}
        for lid, sim in self.simulations.items():
            all_geometry[f"level_{lid}"] = sim.get_geometry()
        return all_geometry

    def board_agents_on_platform(
        self,
        exit_name: str,
        agent_destinations: dict[str, str] | None = None,
        eligible_ids: "set[str] | None" = None,
    ) -> list[str]:
        """
        Board agents standing on the platform when a train dwells there.

        An agent inside the platform polygon is boarded when either:

        * it has **explicitly committed** to this train — its
          ``agent_destinations`` entry equals *exit_name*; or
        * it is a **waiting boarder** — its id is in *eligible_ids*, the set of
          agents whose goal is to board a train and who are not actively routing
          away (computed by the caller from agent goals/destinations).

        Agents actively routing away (a non-train destination such as an
        escalator or street exit) and agents whose goal is to *leave* the
        station — e.g. alighters who have just stepped off onto the platform —
        are left untouched, so they are never re-boarded onto the train they
        just left.

        If *agent_destinations* is not provided every agent inside the platform
        polygon is boarded (legacy fallback).

        Uses Shapely containment against the full platform walkable area so that
        agents board from any point on the platform, not just from the small
        train-entrance marker at one end.

        Args:
            exit_name: Canonical exit name, e.g. ``"train_platform_3"``.
            agent_destinations: Live dict of agent_id -> current exit name.
            eligible_ids: Ids of waiting boarders eligible to board a dwelling
                train even without an explicit ``train_platform_*`` destination.

        Returns:
            List of Concordia IDs marked for removal this step.
        """
        platform_name = platform_zone(exit_name)
        level_sim = self.simulations.get(self.platform_level)
        if level_sim is None:
            return []

        platform_poly = level_sim.geometry_manager.walkable_areas.get(platform_name)
        if platform_poly is None:
            logger.debug(
                f"board_agents_on_platform: no walkable area '{platform_name}' "
                f"found for exit '{exit_name}'."
            )
            return []

        boarded: list[str] = []

        # JuPedSim's agents_in_polygon requires a convex polygon, which the
        # platform walkable areas are not (they can be L-shaped etc.).  Instead,
        # use Shapely containment directly on all agent positions tracked by
        # this level's agent_tracker.
        from shapely.geometry import Point

        current_positions = level_sim.agent_tracker.get_all_positions()
        eligible = eligible_ids or set()
        for concordia_id, pos in current_positions.items():
            # Board an agent when it has explicitly committed to this exact train
            # exit, or when it is a waiting boarder standing on the platform as a
            # train dwells.  Agents routing away via an escalator/street exit and
            # non-boarders (e.g. alighters leaving the station) are left alone —
            # they are excluded from ``eligible_ids`` and never re-boarded.
            if agent_destinations is not None:
                dest = agent_destinations.get(concordia_id, "")
                if dest != exit_name and concordia_id not in eligible:
                    continue
            if not platform_poly.contains(Point(pos)):
                continue
            jps_id = level_sim.agent_tracker.agent_ids.get(concordia_id)
            if jps_id is None:
                continue
            level_sim.agent_assigned_exits[concordia_id] = exit_name
            level_sim.simulation.mark_agent_for_removal(jps_id)
            boarded.append(concordia_id)

        if boarded:
            logger.info(f"🚂 {len(boarded)} agent(s) boarding '{exit_name}': {boarded}")
        return boarded

    def generate_spawn_positions(
        self, num_agents: int, seed: int = 42
    ) -> list[tuple[float, float, str]]:
        """
        Generate spawn positions distributed across all levels.

        Returns list of (x, y, level_id) tuples so agents can be spawned on correct level.
        Distribution is proportional to walkable area on each level.

        Args:
            num_agents: Total number of agents to spawn
            seed: Random seed for reproducibility

        Returns:
            List of (x, y, level_id) tuples
        """
        import random

        random.seed(seed)

        # Calculate total walkable area per level
        level_areas = {}
        for level_id, sim in self.simulations.items():
            total_area = sum(
                poly.area for poly in sim.geometry_manager.walkable_areas_with_obstacles.values()
            )
            level_areas[level_id] = total_area

        total_area = sum(level_areas.values())

        # Distribute agents proportionally by area
        spawn_positions = []
        agents_placed = 0

        for idx, (level_id, area) in enumerate(sorted(level_areas.items())):
            # Calculate proportional number of agents for this level
            if idx == len(level_areas) - 1:
                # Last level gets remainder to ensure exact count
                level_agents = num_agents - agents_placed
            else:
                level_agents = int(num_agents * (area / total_area))

            if level_agents > 0:
                # Generate positions on this level
                positions = self.simulations[level_id].generate_spawn_positions(
                    level_agents, seed + idx
                )

                # Add level_id to each position
                for x, y in positions:
                    spawn_positions.append((x, y, level_id))

                agents_placed += len(positions)
                logger.info(f"Spawning {len(positions)} agents on level {level_id}")

        return spawn_positions
