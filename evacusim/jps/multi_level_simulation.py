"""
Multi-level JuPedSim simulation for Monument Station.

Manages multiple levels (concourse + platforms) and agent transfers between them
via escalators. Each level has its own JuPedSim simulation instance.
"""

import math
import random
import re
from pathlib import Path
from typing import Any

from shapely.geometry import Point

from evacusim.utils.logger import get_logger
from evacusim.coordination.level_transfer_manager import LevelTransferManager
from evacusim.jps.jupedsim_integration import (
    ConcordiaJuPedSimulation,
)

logger = get_logger(__name__)


class MultiLevelJuPedSimulation:
    """
    Manages multiple JuPedSim simulations for multi-level stations.

    Each level has its own simulation instance, and agents can transfer
    between levels via escalators/stairs.
    """

    #: Pattern used to parse escalator zone names.
    _ESCALATOR_ZONE_RE = re.compile(r"^L([^_]+)_esc_([a-f])_(up|down)$")
    #: Pattern used to extract level and escalator letter from corridor names.
    _CORRIDOR_NAME_RE = re.compile(r"^L([^_]+)_esc_corridor_([a-f])$")

    def __init__(
        self,
        network_path: Path,
        dt: float = 0.05,
        exit_radius: float = 10.0,
        levels: list[str] | None = None,
        escalator_belt_speed: float = 0.5,
        level_arrival_waypoints: dict[str, tuple[float, float]] | None = None,
        initially_blocked_exits: set[str] | None = None,
    ):
        """
        Initialize multi-level simulation.

        Args:
            network_path: Path to network directory containing level_*.xml files
            dt: Timestep in seconds
            exit_radius: Radius of circular exits in meters
            levels: List of level IDs to load (default: ["0", "-1"])
            escalator_belt_speed: Speed of the escalator belt in m/s.  Agents
                inside an escalator zone will have their desired speed raised to
                at least this value so they are never stationary relative to the
                surrounding floor.  Agents that drift into an arrival-only zone
                (wrong direction for their level) are redirected to the nearest
                valid exit.  Default 0.5 m/s (standard commercial escalator).
            initially_blocked_exits: Exits that are blocked from simulation start.
                Their corridor geometry and exit stages are omitted so agents
                physically cannot enter them.
        """
        self.dt = dt
        self.exit_radius = exit_radius
        self.network_path = Path(network_path)
        self.current_step = 0
        self.is_complete = False
        self.escalator_belt_speed = escalator_belt_speed
        # Keyed by level_id string; values are (x, y) waypoints assigned to
        # agents immediately after a level transfer, keeping them moving until
        # the next LLM decision cycle.
        self.level_arrival_waypoints: dict[str, tuple[float, float]] = {
            str(k): (float(v[0]), float(v[1]))
            for k, v in (level_arrival_waypoints or {}).items()
        }

        if levels is None:
            levels = ["0", "-1"]
        self.levels = levels

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

        # Track which level each agent is on
        self.agent_levels: dict[str, str] = {}  # agent_id -> level_id
        self.recently_transferred_agents: set[str] = set()
        # Positions used for transfers in the current step (cleared each step).
        # Prevents same-step transfers from landing on top of each other.
        self._pending_spawn_positions: list[tuple[float, float]] = []
        # Transfers deferred because the landing zone was too crowded.
        # Retried at the start of the next step.
        self._deferred_transfers: list[tuple[str, str, str]] = (
            []
        )  # (agent_id, current_level, exit_name)
        # Cooldown: minimum steps between consecutive transfers for the same agent.
        # At dt=0.05 s, 100 steps = 5 seconds — enough time to walk clear of the
        # arrival zone before a return trip could be triggered accidentally.
        self._transfer_cooldown_steps: int = 100
        self._last_transfer_step: dict[str, int] = {}  # agent_id -> step number

        # Exits that are currently blocked (set by hybrid_simulation / EventManager).
        # When an agent physically reaches a blocked escalator exit on a level
        # where no geometry obstacle could be placed, they are returned to the
        # platform floor and flagged for an immediate re-decision.
        self.blocked_exits: set[str] = set()
        # Agents returned from a blocked escalator this step; hybrid_simulation
        # should trigger an immediate LLM re-decision for these.
        self.agents_needing_redecision: set[str] = set()

        # Setup level transfer manager
        self.transfer_manager = LevelTransferManager(network_path, levels)

        # Pre-classify every known escalator zone as a *departure* zone (has a
        # registered JuPedSim exit on its level) or an *arrival* zone (no exit —
        # agents spawn here after transferring and must walk out).
        # Built once after both transfer_manager and simulations are ready.
        self._zone_is_departure: dict[str, bool] = self._classify_escalator_zones()

        # Track which exit each agent has been routed to while inside a corridor.
        # Used to avoid redundant journey switches every step — only re-route when
        # the agent first enters the corridor or their assigned exit changes.
        self._corridor_routed_exit: dict[str, str] = {}  # agent_id -> exit_name

        # Agent IDs that should never be routed to evacuation exits by the
        # escalator-zone handler.  Register director / staff agents here so that
        # accidental entry into an arrival-only escalator zone does not redirect
        # them to a down/up exit and override their patrol or hold targets.
        self.non_evacuating_agents: set[str] = set()

        logger.info(
            f"Multi-level simulation initialized with {len(self.simulations)} levels: "
            f"{', '.join(levels)}"
        )
        logger.info(f"Transfer info: {self.transfer_manager.get_transfer_info()}")
        logger.info(
            f"Escalator belt speed: {self.escalator_belt_speed} m/s  "
            f"Departure zones: "
            f"{[z for z, dep in self._zone_is_departure.items() if dep]}  "
            f"Arrival zones: "
            f"{[z for z, dep in self._zone_is_departure.items() if not dep]}"
        )

    # ------------------------------------------------------------------
    # Escalator zone helpers
    # ------------------------------------------------------------------

    def _classify_escalator_zones(self) -> dict[str, bool]:
        """
        Classify every escalator zone as a departure (True) or arrival (False) zone.

        A zone is a *departure* zone if a JuPedSim exit stage is registered for the
        corresponding canonical exit name on that level.  Arrival zones have no exit
        stage — they only exist so transferred agents can be spawned inside them.

        Returns:
            Mapping of zone_name -> is_departure_zone.
        """
        result: dict[str, bool] = {}
        for zone_name in self.transfer_manager.escalator_zones:
            m = self._ESCALATOR_ZONE_RE.match(zone_name)
            if not m:
                result[zone_name] = False
                continue
            zone_level, esc_letter, direction = m.groups()
            exit_name = f"escalator_{esc_letter}_{direction}"
            level_sim = self.simulations.get(zone_level)
            if level_sim is None:
                result[zone_name] = False
                continue
            result[zone_name] = exit_name in level_sim.exit_manager.evacuation_exits
        return result

    def _enforce_escalator_constraints(self) -> None:
        """
        Per-step escalator physics enforcement — called every simulation step.

        Checks two classes of escalator zone:

        **Exit boxes** (the small ~2 m² terminal zones from which JuPedSim removes
        agents to trigger a level transfer):
        - Departure boxes: enforce minimum speed.
        - Arrival boxes: redirect any agent who hasn't just been transferred (direction
          correction), plus enforce minimum speed.

        **Corridor zones** (``jupedsim.escalator`` polygons in the XML) covering the
        full navigable corridor between the railing obstacles:
        - Enforce minimum speed only.  Direction is already controlled by routing;
          corridor zones exist to prevent agents appearing stationary mid-escalator.
        """
        for level_id, sim in self.simulations.items():
            escalator_corridors = getattr(sim.geometry_manager, "escalator_corridors", {})
            agent_ids = list(sim.agent_tracker.agent_ids.keys())
            for agent_id in agent_ids:
                pos = sim.get_agent_position(agent_id)
                if pos is None:
                    continue

                point = Point(pos)
                handled = False

                # --- Check escalator EXIT BOXES ---
                for zone_name, zone_poly in self.transfer_manager.escalator_zones.items():
                    m = self._ESCALATOR_ZONE_RE.match(zone_name)
                    if not m or m.group(1) != level_id:
                        continue
                    if not zone_poly.contains(point):
                        continue

                    handled = True

                    # Enforce minimum speed (applies regardless of direction).
                    current_speed = sim.get_agent_speed(agent_id)
                    if current_speed is not None and current_speed < self.escalator_belt_speed:
                        sim.set_agent_speed(agent_id, self.escalator_belt_speed)
                        logger.debug(
                            f"[ESCALATOR] {agent_id} in {zone_name}: raised speed from "
                            f"{current_speed:.2f} to {self.escalator_belt_speed:.2f} m/s"
                        )

                    is_departure = self._zone_is_departure.get(zone_name, False)
                    if not is_departure:
                        # Arrival-only zone.  Only redirect if the agent was NOT
                        # recently transferred here, AND is not a non-evacuating
                        # agent (e.g. a director / staff agent on patrol).
                        if agent_id in self.non_evacuating_agents:
                            break  # let them pass through without redirection
                        steps_since = self.current_step - self._last_transfer_step.get(
                            agent_id, -(self._transfer_cooldown_steps + 1)
                        )
                        if steps_since > self._transfer_cooldown_steps:
                            nearest_exit = self._find_nearest_valid_exit_for_level(pos, level_id)
                            if nearest_exit:
                                logger.warning(
                                    f"[ESCALATOR] Direction violation: {agent_id} on level "
                                    f"{level_id} entered arrival-only zone '{zone_name}'. "
                                    f"Redirecting to '{nearest_exit}'."
                                )
                                sim.set_agent_evacuation_exit(agent_id, nearest_exit)

                    # Each agent can only be in one zone at a time.
                    break

                if handled:
                    continue

                # --- Check escalator CORRIDORS ---
                in_any_corridor = False
                for corridor_name, corridor_poly in escalator_corridors.items():
                    if not corridor_poly.contains(point):
                        continue
                    in_any_corridor = True

                    # Enforce minimum speed.
                    current_speed = sim.get_agent_speed(agent_id)
                    if current_speed is not None and current_speed < self.escalator_belt_speed:
                        sim.set_agent_speed(agent_id, self.escalator_belt_speed)
                        logger.debug(
                            f"[ESCALATOR CORRIDOR] {agent_id} in {corridor_name}: raised speed "
                            f"from {current_speed:.2f} to {self.escalator_belt_speed:.2f} m/s"
                        )

                    # Route the agent toward the departure exit for this corridor
                    # letter on this level.  If no departure exit exists (arrival
                    # side of the escalator), leave them alone — they have already
                    # been given a waypoint into the concourse/platform and should
                    # not be sent back the way they came.
                    m = self._CORRIDOR_NAME_RE.match(corridor_name)
                    if m:
                        esc_letter = m.group(2)
                        departure_exit = next(
                            (
                                name
                                for name in sim.exit_manager.evacuation_exits
                                if name == f"escalator_{esc_letter}_down"
                                or name == f"escalator_{esc_letter}_up"
                            ),
                            None,
                        )
                        if (
                            departure_exit
                            and self._corridor_routed_exit.get(agent_id) != departure_exit
                            and agent_id not in self.non_evacuating_agents
                        ):
                            logger.debug(
                                f"[ESCALATOR CORRIDOR] {agent_id} in '{corridor_name}' "
                                f"— routing to '{departure_exit}'"
                            )
                            sim.set_agent_evacuation_exit(agent_id, departure_exit)
                            self._corridor_routed_exit[agent_id] = departure_exit
                    break

                if not in_any_corridor:
                    # Agent left the corridor — clear the cached route so it is
                    # re-asserted if they re-enter.
                    self._corridor_routed_exit.pop(agent_id, None)

    def _find_nearest_valid_exit_for_level(
        self, pos: tuple[float, float], level_id: str
    ) -> str | None:
        """
        Return the name of the nearest registered exit on *level_id* to *pos*.

        Street exits (non-escalator) are preferred so that an agent ejected from
        an arrival-only escalator zone is never redirected straight back down
        (or up) via another escalator.

        Returns None if no exits are available.
        """
        level_sim = self.simulations.get(level_id)
        if level_sim is None:
            return None
        exits = level_sim.exit_manager.exit_coordinates
        if not exits:
            return None
        # Prefer street exits (no "escalator_" prefix) so arriving agents are
        # not bounced back through a different escalator.
        street_exits = {name: coords for name, coords in exits.items()
                        if not name.startswith("escalator_")}
        exits_to_search = street_exits if street_exits else exits
        nearest_name = min(
            exits_to_search,
            key=lambda name: math.hypot(pos[0] - exits_to_search[name][0], pos[1] - exits_to_search[name][1]),
        )
        return nearest_name

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
    ) -> None:
        """
        Add an agent to a specific level.

        Args:
            agent_id: Concordia agent ID
            position: Initial (x, y) position
            walking_speed: Desired walking speed in m/s
            level_id: Level to spawn on (default: "0")
        """
        if level_id not in self.simulations:
            raise ValueError(f"Level {level_id} not loaded")

        self.simulations[level_id].add_agent(agent_id, position, walking_speed)
        self.agent_levels[agent_id] = level_id

        logger.info(f"Added agent {agent_id} to level {level_id} at {position}")

    def step(self) -> bool:
        """
        Advance all level simulations and process agent transfers between levels.

        Returns:
            True if simulation should continue, False if complete
        """
        if self.is_complete:
            return False

        # Step 1: Check for agents that exited through escalators and transfer them
        self._process_escalator_exits(self.blocked_exits)

        # Step 2: Step each level's simulation
        any_active = False
        for sim in self.simulations.values():
            if sim.step():
                any_active = True

        # Step 3: Enforce escalator physics (minimum speed + direction correction).
        # Done after JuPedSim has advanced so position data is fresh.
        self._enforce_escalator_constraints()

        self.current_step += 1

        # Check if simulation is complete (no agents left anywhere)
        total_agents = sum(sim.simulation.agent_count() for sim in self.simulations.values())
        if total_agents == 0:
            logger.info("All agents have exited the simulation")
            self.is_complete = not any_active
            return False

        return True

    def _process_escalator_exits(self, blocked_exits: set[str] | None = None):
        """
        Check each level for agents that have exited through escalators.

        Escalators are exits that connect two levels. When an agent exits through
        an escalator on one level, they are spawned into the target level.

        If *blocked_exits* is provided, agents that reach a blocked escalator exit
        are returned to a safe position on their current level and flagged for
        immediate re-decision rather than being transferred.
        """
        if blocked_exits is None:
            blocked_exits = set()
        # Reset same-step spawn tracking so each step starts fresh.
        self._pending_spawn_positions.clear()

        # Retry any transfers that were deferred last step because the zone was crowded.
        if self._deferred_transfers:
            deferred = self._deferred_transfers[:]
            self._deferred_transfers.clear()
            for agent_id, from_level, esc_name in deferred:
                logger.info(f"Retrying deferred transfer for {agent_id} via {esc_name}")
                self._transfer_agent_through_escalator(agent_id, from_level, esc_name)

        # Check each level for agent exits
        for level_id, sim in self.simulations.items():
            exited_agents = sim.check_exits()

            if exited_agents:
                logger.info(
                    f"Level {level_id}: {len(exited_agents)} agents exited - {exited_agents}"
                )
            else:
                logger.debug(f"Level {level_id}: No agents exited")

            # Process each exited agent
            for agent_id, exit_name in exited_agents.items():
                # Check if exit is an escalator (starts with "escalator_")
                if not exit_name.startswith("escalator_"):
                    logger.info(f"Agent {agent_id} exited station through street exit {exit_name}")
                    # Remove from level tracking - agent has truly exited station
                    if agent_id in self.agent_levels:
                        del self.agent_levels[agent_id]
                    continue

                # --- Blocked exit interception (general) ---
                # If this exit is blocked, the agent physically reached a barrier.
                # Re-spawn them at their last known position so they are back
                # just in front of the barrier, and flag for immediate re-decision.
                # This mechanism is exit-type agnostic — it works for any blocked
                # exit (escalator, door, collapsed-person blockage, etc.).
                if exit_name in blocked_exits:
                    last_pos = self.simulations[level_id].last_known_positions.get(agent_id)
                    sim = self.simulations[level_id]
                    if last_pos is not None:
                        try:
                            sim.add_agent(agent_id, last_pos, walking_speed=1.34)
                            self.agent_levels[agent_id] = level_id
                            self.agents_needing_redecision.add(agent_id)
                            logger.info(
                                f"Agent {agent_id} reached blocked exit {exit_name} on level "
                                f"{level_id} — returned to last position {last_pos} for re-decision"
                            )
                        except Exception as e:
                            logger.warning(
                                f"Could not return agent {agent_id} after blocked exit: {e}"
                            )
                    else:
                        logger.warning(
                            f"Agent {agent_id} reached blocked exit {exit_name} "
                            f"but no last position recorded — agent lost"
                        )
                    continue

                # Enforce cooldown to prevent immediate bounce-back transfers.
                last_step = self._last_transfer_step.get(agent_id, -self._transfer_cooldown_steps)
                steps_since = self.current_step - last_step
                if steps_since < self._transfer_cooldown_steps:
                    remaining_s = (self._transfer_cooldown_steps - steps_since) * self.dt
                    logger.warning(
                        f"Agent {agent_id} tried to transfer again via {exit_name} only "
                        f"{steps_since} steps after last transfer (cooldown: "
                        f"{self._transfer_cooldown_steps} steps). "
                        f"Ignoring for {remaining_s:.1f}s more."
                    )
                    continue

                # This is an escalator exit - transfer to target level
                logger.info(
                    f"Agent {agent_id} reached escalator exit {exit_name} on level {level_id} - initiating transfer"
                )
                self._transfer_agent_through_escalator(agent_id, level_id, exit_name)

    def _transfer_agent_through_escalator(self, agent_id: str, current_level: str, exit_name: str):
        """
        Transfer an agent from one level to another through an escalator.

        The up/down suffix in the exit name is for agent decision-making only.
        For the physical transfer, only the escalator letter (a-f) matters:
        an agent exiting through any escalator_X_* on level N spawns at the
        escalator_X zone on the other level, regardless of direction.

        Args:
            agent_id: Concordia agent ID
            current_level: Current level ID (e.g., "0" or "-1")
            exit_name: Name of the escalator exit (e.g., "escalator_a_down")
        """
        # Two levels only: flip between them
        target_level = "-1" if current_level == "0" else "0"

        if target_level not in self.simulations:
            logger.warning(f"Target level {target_level} not in simulation")
            return

        # Extract escalator letter — only this matters for locating the arrival zone.
        # exit_name format: "escalator_{letter}_{direction}" e.g. "escalator_a_down"
        parts = exit_name.split("_")
        if len(parts) < 2:
            logger.error(f"Cannot parse escalator letter from '{exit_name}'")
            return
        esc_letter = parts[1]  # 'a', 'b', 'c', etc.

        # Find whatever zone exists for this letter on the target level.
        # Zone naming: L{level}_esc_{letter}_{direction}
        target_zone_name = next(
            (
                k
                for k in self.transfer_manager.escalator_zones
                if k.startswith(f"L{target_level}_esc_{esc_letter}_")
            ),
            None,
        )

        if target_zone_name is None:
            logger.error(
                f"No arrival zone found for escalator '{esc_letter}' on level {target_level}. "
                f"Available: {list(self.transfer_manager.escalator_zones.keys())}"
            )
            return

        target_zone_poly = self.transfer_manager.escalator_zones[target_zone_name]
        centroid = target_zone_poly.centroid

        # Erode the polygon by JuPedSim's minimum boundary clearance (0.2 m) plus a
        # small margin so random candidates are never too close to walls.
        BOUNDARY_MARGIN = 0.3
        safe_zone = target_zone_poly.buffer(-BOUNDARY_MARGIN)
        if safe_zone.is_empty:
            # Polygon too small to erode — fall back to centroid only
            safe_zone = target_zone_poly

        # Choose a spawn position that doesn't collide with:
        #   (a) agents already present on the target level, and
        #   (b) other agents being transferred to this level in the same step.
        MIN_AGENT_SEP = 0.4
        existing_positions = (
            list(self.simulations[target_level].get_all_agent_positions().values())
            + self._pending_spawn_positions
        )

        spawn_pos = None
        for _attempt in range(60):
            angle = random.uniform(0, 2 * math.pi)
            radius = random.uniform(0.0, 0.7)
            candidate = (
                centroid.x + math.cos(angle) * radius,
                centroid.y + math.sin(angle) * radius,
            )
            if not safe_zone.contains(Point(candidate)):
                continue
            if any(
                math.hypot(candidate[0] - p[0], candidate[1] - p[1]) < MIN_AGENT_SEP
                for p in existing_positions
            ):
                continue
            spawn_pos = candidate
            break

        if spawn_pos is None:
            # Escalator zone is too crowded — wait until next step rather than crash.
            logger.warning(
                f"Cannot find free spawn point for {agent_id} in {target_zone_name} "
                f"({len(existing_positions)} agents nearby). Deferring transfer."
            )
            self._deferred_transfers.append((agent_id, current_level, exit_name))
            return

        self._pending_spawn_positions.append(spawn_pos)

        # Clear cached corridor route so the arriving agent is re-routed immediately
        # if they land inside a corridor on the target level.
        self._corridor_routed_exit.pop(agent_id, None)

        # Spawn agent in target level
        try:
            self.simulations[target_level].add_agent(agent_id, spawn_pos)
            self.agent_levels[agent_id] = target_level
            self.recently_transferred_agents.add(agent_id)
            self._last_transfer_step[agent_id] = self.current_step

            logger.info(
                f"Transferred agent {agent_id} from level {current_level} to {target_level} "
                f"through {exit_name} at {spawn_pos}"
            )

            # Assign a temporary waypoint inside the main walkable area of the
            # target level so the agent keeps moving until the next decision
            # cycle gives them a real goal.  We do NOT route to an exit here
            # because the agent may have transferred to catch a train, not to
            # leave the building.
            level_sim = self.simulations[target_level]
            try:
                # For level 0 (concourse), use a fixed waypoint in the open
                # concourse floor — near the Blackett Street corridor mouth
                # and clear of all escalator exits, so agents walk into the
                # visible area before their next LLM decision fires.
                if str(target_level) in self.level_arrival_waypoints:
                    temp_pos = self.level_arrival_waypoints[str(target_level)]
                else:
                    walkable = level_sim.geometry_manager.walkable_areas_with_obstacles
                    main_poly = max(walkable.values(), key=lambda p: p.area)
                    safe_main = main_poly.buffer(-0.3)
                    if safe_main.is_empty:
                        safe_main = main_poly
                    anchor = safe_main.representative_point()
                    temp_pos = (anchor.x, anchor.y)
                level_sim.set_agent_target(agent_id, temp_pos)
                logger.debug(
                    f"Assigned temporary waypoint {temp_pos} to "
                    f"transferred agent {agent_id} on level {target_level}"
                )
            except Exception as dest_err:
                logger.warning(f"Could not assign temporary waypoint to {agent_id}: {dest_err}")

        except Exception as e:
            logger.error(f"Failed to transfer agent {agent_id} to level {target_level}: {e}")
            # Remove agent from tracking if transfer failed
            if agent_id in self.agent_levels:
                del self.agent_levels[agent_id]

    def add_geometry_obstacle_for_exit(self, exit_name: str) -> None:
        """Punch a geometry obstacle at the entrance of *exit_name* on every level.

        Each level is attempted independently so a failure on one level (e.g. the
        platform level whose topology doesn't support runtime corridor removal)
        doesn't prevent the blocker being applied on other levels.
        """
        for level_id, sim in self.simulations.items():
            try:
                sim.add_geometry_obstacle_for_exit(exit_name)
            except Exception as e:
                from evacusim.utils.logger import get_logger as _gl
                _gl(__name__).debug(
                    f"Geometry obstacle skipped for '{exit_name}' on level {level_id}: {e}"
                )

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

    def set_agent_evacuation_exit(self, agent_id: str, exit_name: str) -> None:
        """
        Direct an agent to a specific evacuation exit on their current level.

        The exit must exist on the agent's current level. For multi-level evacuations:
        - On platform levels: Agents route to escalator exits (e.g., "escalator_a_up")
        - On concourse levels: Agents route to street exits (e.g., "eldon_square")

        Direction is enforced programmatically: UP escalators are not valid exits
        from the concourse (level 0), and DOWN escalators are not valid exits from
        the platform level (level -1).  Any such request is rejected with an error
        log and the agent keeps its current journey.

        Args:
            agent_id: ID of the agent
            exit_name: Name of the exit to route - must exist on current level
        """
        if agent_id not in self.agent_levels:
            return

        level_id = self.agent_levels[agent_id]
        level_sim = self.simulations[level_id]

        # Explicit direction guard: detect wrong-way escalator requests and log
        # a clear error before the generic existence check catches them silently.
        esc_m = re.match(r"^escalator_([a-f])_(up|down)$", exit_name)
        if esc_m:
            direction = esc_m.group(2)
            if level_id == "0" and direction == "up":
                logger.error(
                    f"[DIRECTION VIOLATION] {agent_id} on concourse (level 0) requested "
                    f"UP escalator '{exit_name}'.  UP escalators are arrival zones on "
                    f"the concourse — agents must use a DOWN escalator to reach platforms, "
                    f"or a street exit to leave the building.  Request refused."
                )
                return
            if level_id == "-1" and direction == "down":
                logger.error(
                    f"[DIRECTION VIOLATION] {agent_id} on platform level (-1) requested "
                    f"DOWN escalator '{exit_name}'.  DOWN escalators are arrival zones on "
                    f"the platform — agents must use an UP escalator to reach the concourse.  "
                    f"Request refused."
                )
                return

        # Check if exit exists on this level
        if exit_name not in level_sim.exit_manager.evacuation_exits:
            # Raise KeyError so callers (e.g. ActionExecutor._handle_move_action)
            # can fall back to waypoint navigation toward the blocked position,
            # allowing the agent to walk up to the exit and discover the blockage
            # through observation.
            raise KeyError(
                f"Agent {agent_id} on level {level_id} tried to route to exit '{exit_name}' "
                f"which doesn't exist on this level. Available exits: "
                f"{list(level_sim.exit_manager.evacuation_exits.keys())}"
            )

        # Route to the exit on this level
        level_sim.set_agent_evacuation_exit(agent_id, exit_name)

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

    def get_simulation_time(self) -> float:
        """Get current simulation time in seconds."""
        return self.current_step * self.dt

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
    ) -> list[str]:
        """
        Board agents that have committed to boarding this train.

        Only agents whose ``agent_destinations`` entry matches *exit_name* are
        boarded — agents merely crossing the platform toward an escalator are
        left alone.  If *agent_destinations* is not provided every agent inside
        the platform polygon is boarded (legacy fallback).

        Uses Shapely containment against the full platform walkable area so that
        agents board from any point on the platform, not just from the small
        train-entrance marker at one end.

        Args:
            exit_name: Canonical exit name, e.g. ``"train_platform_3"``.
            agent_destinations: Live dict of agent_id -> current exit name.

        Returns:
            List of Concordia IDs marked for removal this step.
        """
        platform_name = "platform_" + exit_name.rsplit("_", 1)[-1]
        level_sim = self.simulations.get("-1")
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
        for concordia_id, pos in current_positions.items():
            # Only board agents who have explicitly committed to this exact
            # train exit.  Agents with no destination (waiting) or heading to
            # an escalator/street exit are left alone — they haven't chosen
            # to board and should not be silently removed.
            if agent_destinations is not None:
                dest = agent_destinations.get(concordia_id, "")
                if dest != exit_name:
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
            logger.info(
                f"🚂 {len(boarded)} agent(s) boarding '{exit_name}': {boarded}"
            )
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
