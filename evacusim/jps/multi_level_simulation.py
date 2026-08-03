"""
Multi-level JuPedSim simulation for Monument Station.

Manages multiple levels (concourse + platforms) and agent transfers between them
via escalators. Each level has its own JuPedSim simulation instance.
"""

import math
import random
from pathlib import Path
from typing import Any

from shapely.geometry import Point
from shapely.ops import nearest_points

from evacusim.utils.logger import get_logger
from evacusim.coordination.level_transfer_manager import LevelTransferManager
from evacusim.jps.escalator_controller import EscalatorController
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

    def __init__(
        self,
        network_path: Path,
        dt: float = 0.05,
        exit_radius: float = 10.0,
        levels: list[str] | None = None,
        escalator_belt_speed: float = 0.5,
        level_arrival_waypoints: dict[str, tuple[float, float]] | None = None,
        transfer_random_waypoint_min_distance_m: float = 10.0,
        initially_blocked_exits: set[str] | None = None,
    ):
        """
        Initialize multi-level simulation.

        Args:
            network_path: Path to network directory containing level_*.xml files
            dt: Timestep in seconds
            exit_radius: Radius of circular exits in meters
            levels: List of level IDs to load (default: ["0", "-1"])
            escalator_belt_speed: Speed floor for agents inside escalator zones
                and corridors. Default 0.5 m/s (standard commercial escalator).
            transfer_random_waypoint_min_distance_m: Minimum distance from the
                escalator landing that the random temporary destination must be.
                The agent walks there under JuPedSim routing while the LLM fires
                its scheduled decision; the LLM decision then overrides the waypoint.
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
        self.transfer_random_waypoint_min_distance_m = max(
            2.0, float(transfer_random_waypoint_min_distance_m)
        )

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

        # Phase A: Build explicit escalator registry + startup validation.
        # Runtime transfer behavior remains unchanged in this phase.
        self.escalator_controller = EscalatorController(
            network_path=self.network_path,
            levels=self.levels,
            simulations=self.simulations,
        )
        registry_summary = self.escalator_controller.summary()
        logger.info(
            "Escalator registry initialized: "
            f"endpoints={registry_summary['endpoint_count']}, "
            f"edges={registry_summary['edge_count']}, "
            f"ids={registry_summary['escalator_ids']}"
        )
        for issue in self.escalator_controller.validate_registry():
            logger.warning(f"[ESCALATOR REGISTRY] {issue}")

        logger.info(
            f"Multi-level simulation initialized with {len(self.simulations)} levels: "
            f"{', '.join(levels)}"
        )
        logger.info(f"Transfer info: {self.transfer_manager.get_transfer_info()}")
        logger.info(f"Escalator belt speed: {self.escalator_belt_speed} m/s")
        logger.info(
            "Transfer random waypoint: "
            f"min_distance={self.transfer_random_waypoint_min_distance_m:.1f}m "
            "(agent walks to a random level point; LLM fires during transit)"
        )

    # ------------------------------------------------------------------
    # Escalator zone helpers
    # ------------------------------------------------------------------

    def _enforce_escalator_constraints(self) -> None:
        """
        Per-step escalator physics enforcement — called every simulation step.

        Enforces the speed floor for agents physically inside escalator zones and
        corridors. Direction control for departure-role zones is handled by the
        escalator controller; arrival-role zones are not overridden here so agents
        keep whatever destination was assigned at transfer time.
        """
        for level_id, sim in self.simulations.items():
            agent_ids = list(sim.agent_tracker.agent_ids.keys())
            for agent_id in agent_ids:
                pos = sim.get_agent_position(agent_id)
                if pos is None:
                    continue

                in_escalator_geometry = self.escalator_controller.enforce_motion_for_agent(
                    agent_id=agent_id,
                    level_id=level_id,
                    position=pos,
                    level_sim=sim,
                )

                if not in_escalator_geometry:
                    continue

                current_speed = sim.get_agent_speed(agent_id)
                if current_speed is not None and current_speed < self.escalator_belt_speed:
                    sim.set_agent_speed(agent_id, self.escalator_belt_speed)
                    logger.debug(
                        f"[ESCALATOR] {agent_id} raised speed from "
                        f"{current_speed:.2f} to {self.escalator_belt_speed:.2f} m/s"
                    )

    def _pick_random_level_waypoint(
        self,
        level_id: str,
        away_from: tuple[float, float],
    ) -> tuple[float, float] | None:
        """Return a random walkable point on *level_id* that is outside escalator zones.

        The point is at least ``transfer_random_waypoint_min_distance_m`` from
        *away_from* (the escalator landing position) and is not inside any
        escalator transfer zone or corridor. JuPedSim will path-find through the
        escalator corridor naturally to reach the point, so the agent clears the
        landing area without the simulation needing to manage their direction.

        The LLM decision (scheduled immediately on transfer) fires while the agent
        is en route and overwrites this waypoint with the agent's real intention.
        If the agent reaches the point before the LLM fires, they stop briefly
        but the next scheduled decision cycle will pick them up — they are now
        well clear of the escalator mouth so they do not block it.
        """
        sim = self.simulations.get(level_id)
        if sim is None:
            return None

        combined = getattr(sim.geometry_manager, "_combined_geometry", None)
        if combined is None or combined.is_empty:
            return None

        inner = combined.buffer(-0.3)
        if inner.is_empty:
            inner = combined

        # Collect escalator zones and corridors to avoid.
        exclusion_polys: list = []
        for zone_name in self.escalator_controller.get_level_zone_names(level_id):
            poly = self.escalator_controller.get_zone_polygon(zone_name)
            if poly is not None:
                exclusion_polys.append(poly)
        level_sim = self.simulations.get(level_id)
        if level_sim is not None:
            corridors = getattr(level_sim.geometry_manager, "escalator_corridors", {})
            exclusion_polys.extend(corridors.values())

        minx, miny, maxx, maxy = inner.bounds
        min_dist = self.transfer_random_waypoint_min_distance_m

        for _ in range(60):
            x = random.uniform(minx, maxx)
            y = random.uniform(miny, maxy)
            if not inner.contains(Point((x, y))):
                continue
            if math.hypot(x - away_from[0], y - away_from[1]) < min_dist:
                continue
            if any(ep.contains(Point((x, y))) for ep in exclusion_polys):
                continue
            return (x, y)

        # Fallback: use the geometry representative point (always valid).
        rp = inner.representative_point()
        return (float(rp.x), float(rp.y))

    def _enforce_transfer_discharge(self) -> None:  # no-op: superseded by random-waypoint
        """No-op retained for call-site compatibility. Logic removed."""

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

        # Step 3: Enforce escalator physics (speed floor for departure zones/corridors).
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
                    self._restore_agent_to_source_level(
                        agent_id=agent_id,
                        source_level=level_id,
                        reason=(
                            f"Agent reached blocked exit {exit_name} on level {level_id}; "
                            "forcing immediate re-decision"
                        ),
                        force_redecision=True,
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

        Args:
            agent_id: Concordia agent ID
            current_level: Current level ID (e.g., "0" or "-1")
            exit_name: Name of the escalator exit (e.g., "escalator_a_down")
        """
        edge = self.escalator_controller.get_edge_for_exit(current_level, exit_name)
        if edge is None:
            available = [
                (e.from_level, e.escalator_id, e.direction)
                for e in self.escalator_controller.registry.edges
            ]
            logger.error(
                f"No transfer edge found for level {current_level} exit '{exit_name}'. "
                f"Available signatures: {available}"
            )
            self._restore_agent_to_source_level(
                agent_id=agent_id,
                source_level=current_level,
                reason=(
                    f"No transfer edge for {exit_name} on level {current_level}; "
                    "restoring agent to source level"
                ),
                force_redecision=True,
            )
            return

        target_level = edge.to_level
        if target_level not in self.simulations:
            logger.error(
                f"Transfer edge target level {target_level} is not loaded "
                f"for edge {edge.from_zone_name} -> {edge.to_zone_name}"
            )
            self._restore_agent_to_source_level(
                agent_id=agent_id,
                source_level=current_level,
                reason=(
                    f"Target level {target_level} not loaded for {exit_name}; "
                    "restoring to source level"
                ),
                force_redecision=True,
            )
            return

        target_zone_name = edge.to_zone_name

        target_zone_poly = self.escalator_controller.get_zone_polygon(target_zone_name)
        if target_zone_poly is None:
            logger.error(
                f"Transfer target zone '{target_zone_name}' not found in zone registry"
            )
            self._restore_agent_to_source_level(
                agent_id=agent_id,
                source_level=current_level,
                reason=(
                    f"Target zone {target_zone_name} missing for {exit_name}; "
                    "restoring to source level"
                ),
                force_redecision=True,
            )
            return

        base_spawn = self.escalator_controller.get_spawn_point_for_edge(edge)

        # Erode the polygon by JuPedSim's minimum boundary clearance (0.2 m) plus a
        # small margin so random candidates are never too close to walls.
        BOUNDARY_MARGIN = 0.3
        safe_zone = target_zone_poly.buffer(-BOUNDARY_MARGIN)
        if safe_zone.is_empty or not safe_zone.contains(Point(base_spawn)):
            logger.error(
                f"Invalid transfer metadata for edge {edge.from_endpoint_id} -> {edge.to_endpoint_id}: "
                f"spawn point {base_spawn} is outside safe arrival zone '{target_zone_name}'."
            )
            self._restore_agent_to_source_level(
                agent_id=agent_id,
                source_level=current_level,
                reason=(
                    f"Invalid transfer spawn metadata for {exit_name}; restoring to source level"
                ),
                force_redecision=True,
            )
            return

        # Choose a spawn position that doesn't collide with:
        #   (a) agents already present on the target level, and
        #   (b) other agents being transferred to this level in the same step.
        MIN_AGENT_SEP = 0.4
        existing_positions = (
            list(self.simulations[target_level].get_all_agent_positions().values())
            + self._pending_spawn_positions
        )

        spawn_pos = None
        if not any(
            math.hypot(base_spawn[0] - p[0], base_spawn[1] - p[1]) < MIN_AGENT_SEP
            for p in existing_positions
        ):
            spawn_pos = base_spawn

        for _attempt in range(60):
            if spawn_pos is not None:
                break
            angle = random.uniform(0, 2 * math.pi)
            radius = random.uniform(0.0, 0.9)
            candidate = (
                base_spawn[0] + math.cos(angle) * radius,
                base_spawn[1] + math.sin(angle) * radius,
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
            # Escalator landing zone is too crowded. Never leave the agent orphaned:
            # put them back on the source level and force a re-decision.
            logger.warning(
                f"Cannot find free spawn point for {agent_id} in {target_zone_name} "
                f"({len(existing_positions)} agents nearby). Restoring to source level."
            )
            self._restore_agent_to_source_level(
                agent_id=agent_id,
                source_level=current_level,
                reason=(
                    f"Transfer landing zone crowded for {exit_name}; restored to source level"
                ),
                force_redecision=True,
            )
            return

        self._pending_spawn_positions.append(spawn_pos)

        # Spawn agent in target level
        try:
            self.simulations[target_level].add_agent(
                agent_id,
                spawn_pos,
                assign_default_destination=False,
            )
            self.agent_levels[agent_id] = target_level
            self.recently_transferred_agents.add(agent_id)
            self._last_transfer_step[agent_id] = self.current_step

            logger.info(
                f"Transferred agent {agent_id} from level {current_level} to {target_level} "
                f"through {exit_name} at {spawn_pos}"
            )

            # Give the agent a random temporary destination that is well clear of
            # all escalator zones. JuPedSim paths through the corridor naturally;
            # the LLM decision (queued immediately via consume_recently_transferred_agents)
            # fires during transit and replaces this waypoint with the agent's real choice.
            landing_origin = (float(edge.to_spawn_point[0]), float(edge.to_spawn_point[1]))
            random_wp = self._pick_random_level_waypoint(
                level_id=target_level,
                away_from=landing_origin,
            )
            if random_wp is not None:
                try:
                    self.simulations[target_level].set_agent_target(agent_id, random_wp)
                    logger.debug(
                        f"[TRANSFER] {agent_id} → random waypoint {random_wp} "
                        f"on level {target_level} (LLM fires during transit)"
                    )
                except Exception as e:
                    logger.debug(f"Could not set transfer waypoint for {agent_id}: {e}")

        except Exception as e:
            logger.error(f"Failed to transfer agent {agent_id} to level {target_level}: {e}")
            self._restore_agent_to_source_level(
                agent_id=agent_id,
                source_level=current_level,
                reason=(
                    f"Exception while transferring via {exit_name}: {e}. "
                    "Restoring to source level"
                ),
                force_redecision=True,
            )

    def _restore_agent_to_source_level(
        self,
        agent_id: str,
        source_level: str,
        reason: str,
        force_redecision: bool = True,
    ) -> None:
        """Re-spawn an agent on the source level after a failed transfer path.

        This guarantees an agent cannot silently disappear if transfer mapping,
        target-zone metadata, or landing-space placement fails.
        """
        source_sim = self.simulations.get(source_level)
        if source_sim is None:
            logger.error(
                f"Cannot restore {agent_id}: source level {source_level} not loaded"
            )
            return

        restore_pos = self._resolve_valid_restore_position(
            source_sim=source_sim,
            preferred_pos=source_sim.last_known_positions.get(agent_id),
        )
        if restore_pos is None:
            logger.error(
                f"Cannot restore {agent_id} on level {source_level}: no fallback position available"
            )
            return

        try:
            source_sim.add_agent(
                agent_id,
                restore_pos,
                walking_speed=1.34,
                assign_default_destination=False,
            )
            self.agent_levels[agent_id] = source_level
            if force_redecision:
                self.agents_needing_redecision.add(agent_id)
            logger.warning(
                f"Restored {agent_id} to level {source_level} at {restore_pos}. Reason: {reason}"
            )
        except Exception as e:
            retry_pos = self._resolve_valid_restore_position(source_sim=source_sim, preferred_pos=None)
            if retry_pos is not None and retry_pos != restore_pos:
                try:
                    source_sim.add_agent(
                        agent_id,
                        retry_pos,
                        walking_speed=1.34,
                        assign_default_destination=False,
                    )
                    self.agent_levels[agent_id] = source_level
                    if force_redecision:
                        self.agents_needing_redecision.add(agent_id)
                    logger.warning(
                        f"Restored {agent_id} to level {source_level} at {retry_pos} after retry. "
                        f"Reason: {reason}"
                    )
                    return
                except Exception as retry_e:
                    logger.error(
                        f"Failed retry restore for {agent_id} on level {source_level}: {retry_e}"
                    )

            logger.error(
                f"Failed to restore {agent_id} on level {source_level} after transfer issue: {e}"
            )

    def _resolve_valid_restore_position(
        self,
        source_sim: ConcordiaJuPedSimulation,
        preferred_pos: tuple[float, float] | None,
    ) -> tuple[float, float] | None:
        """Return a position guaranteed to be inside the current walkable geometry."""
        combined = getattr(source_sim.geometry_manager, "_combined_geometry", None)

        if combined is not None and not combined.is_empty:
            # Keep a small margin from boundaries to avoid "on-edge" insertion failures.
            inner = combined.buffer(-0.08)
            if inner.is_empty:
                inner = combined

            def _inside(pos: tuple[float, float]) -> bool:
                return bool(inner.contains(Point(pos)))

            if preferred_pos is not None and _inside(preferred_pos):
                return preferred_pos

            if preferred_pos is not None:
                try:
                    nearest_on_walkable, _ = nearest_points(inner, Point(preferred_pos))
                    base = (float(nearest_on_walkable.x), float(nearest_on_walkable.y))
                    samples = [
                        base,
                        (base[0] + 0.15, base[1]),
                        (base[0] - 0.15, base[1]),
                        (base[0], base[1] + 0.15),
                        (base[0], base[1] - 0.15),
                        (base[0] + 0.25, base[1] + 0.25),
                        (base[0] + 0.25, base[1] - 0.25),
                        (base[0] - 0.25, base[1] + 0.25),
                        (base[0] - 0.25, base[1] - 0.25),
                    ]
                    for candidate in samples:
                        if _inside(candidate):
                            return candidate
                except Exception:
                    pass

            rp = inner.representative_point()
            return (float(rp.x), float(rp.y))

        for poly in source_sim.geometry_manager.walkable_areas.values():
            rp = poly.representative_point()
            return (float(rp.x), float(rp.y))

        return None

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

    def set_agent_destination_exit(self, agent_id: str, exit_name: str) -> None:
        """
        Direct an agent to a specific named exit on their current level.

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

        # Geometry-registry direction guard: if an escalator exit exists but has
        # no transfer edge from this level, it is an arrival-only endpoint here.
        if exit_name.startswith("escalator_"):
            edge = self.escalator_controller.get_edge_for_exit(level_id, exit_name)
            if edge is None:
                if any(e.from_exit_name == exit_name for e in self.escalator_controller.registry.edges):
                    logger.error(
                        f"[DIRECTION VIOLATION] {agent_id} on level {level_id} requested "
                        f"arrival-only escalator '{exit_name}' on this level. Request refused."
                    )
                    return

        # Check if exit exists on this level
        if exit_name not in level_sim.exit_manager.evacuation_exits:
            # Raise KeyError so callers must surface and handle invalid exit
            # choices explicitly rather than silently rerouting.
            raise KeyError(
                f"Agent {agent_id} on level {level_id} tried to route to exit '{exit_name}' "
                f"which doesn't exist on this level. Available exits: "
                f"{list(level_sim.exit_manager.evacuation_exits.keys())}"
            )

        # Route to the exit on this level
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
