"""
Action Executor

Executes translated agent actions in the JuPedSim simulation, including:
- Speed adjustments
- Movement to exits, waypoints, or toward other agents
- Waiting behavior (standing still or seeking information)
"""

import math
import random
from typing import Any

from evacusim.utils.logger import get_logger
from evacusim.decision.action_utils import extract_exit_name
from evacusim.utils.speed_utils import convert_speed_to_ms

logger = get_logger(__name__)

# TODO: Calibrate these from empirical evacuation speed distributions.
# These are not intended as free tuning parameters during scenario runs.
HURRYING_MULTIPLIER: float | None = None
RUNNING_MULTIPLIER: float | None = None


class ActionExecutor:
    """Applies translated actions to the JuPedSim simulation."""

    def __init__(
        self,
        jps_sim,
        state_queries,
        event_manager,
        station_layout: dict[str, Any],
        agent_injured: set[str],
        agent_action: dict[str, str],
        agent_last_decision: dict[str, dict],
        agent_destinations: dict[str, str],
        wait_events: list[dict[str, Any]],
        agent_configs: list[dict[str, Any]],
        agent_roles: dict[str, str] | None = None,
        pace_multipliers: dict[str, Any] | None = None,
    ):
        """
        Initialize action executor.

        Args:
            jps_sim: JuPedSim simulation instance
            state_queries: Simulation state query interface
            event_manager: Event manager for blocked exits
            station_layout: Station geometry and exit information
            agent_injured: Set of agent IDs who are injured/slow
            agent_action: agent_id -> "moving"|"waiting"
            agent_last_decision: agent_id -> last translated_action dict (for memory)
            agent_destinations: agent_id -> current exit name
            wait_events: List tracking all wait decisions with reasons
            agent_configs: List of agent configuration dictionaries
        """
        self.jps_sim = jps_sim
        self.state_queries = state_queries
        self.event_manager = event_manager
        self.station_layout = station_layout
        self.agent_injured = agent_injured
        self.agent_action = agent_action
        self.agent_last_decision = agent_last_decision
        self.agent_destinations = agent_destinations
        self.wait_events = wait_events
        self.agent_configs = agent_configs
        self.agent_roles = agent_roles or {}
        self._zone_adjacency: dict[str, list[str]] = station_layout.get("zone_adjacency", {})
        self._indicator_boards: list[dict[str, Any]] = station_layout.get("indicator_boards", [])
        self._configure_pace_multipliers(pace_multipliers or {})

    @staticmethod
    def _coerce_multiplier(value: Any, key: str) -> float:
        """Coerce a configured multiplier and validate positivity."""
        try:
            coerced = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"performance.pace_multipliers.{key} must be a number") from exc
        if coerced <= 0.0:
            raise ValueError(f"performance.pace_multipliers.{key} must be > 0")
        return coerced

    def _configure_pace_multipliers(self, pace_multipliers: dict[str, Any]) -> None:
        """Configure module-level pace multipliers from scenario config."""
        global HURRYING_MULTIPLIER, RUNNING_MULTIPLIER

        if not isinstance(pace_multipliers, dict):
            raise ValueError("performance.pace_multipliers must be a mapping when provided")

        hurrying = pace_multipliers.get("hurrying")
        running = pace_multipliers.get("running")

        HURRYING_MULTIPLIER = (
            self._coerce_multiplier(hurrying, "hurrying") if hurrying is not None else None
        )
        RUNNING_MULTIPLIER = (
            self._coerce_multiplier(running, "running") if running is not None else None
        )

        if HURRYING_MULTIPLIER is not None and HURRYING_MULTIPLIER < 1.0:
            logger.warning(
                "Configured hurrying multiplier %.3f is < 1.0; this will slow agents.",
                HURRYING_MULTIPLIER,
            )
        if RUNNING_MULTIPLIER is not None and RUNNING_MULTIPLIER < 1.0:
            logger.warning(
                "Configured running multiplier %.3f is < 1.0; this will slow agents.",
                RUNNING_MULTIPLIER,
            )
        if (
            HURRYING_MULTIPLIER is not None
            and RUNNING_MULTIPLIER is not None
            and RUNNING_MULTIPLIER < HURRYING_MULTIPLIER
        ):
            logger.warning(
                "Configured running multiplier %.3f is below hurrying %.3f.",
                RUNNING_MULTIPLIER,
                HURRYING_MULTIPLIER,
            )

    def execute_action(
        self, agent_id: str, translated_action: dict[str, Any], current_sim_time: float
    ):
        """
        Apply a translated action to the JuPedSim simulation.

        Args:
            agent_id: ID of the agent performing the action
            translated_action: Dict containing action details from ActionTranslator
            current_sim_time: Current simulation time in seconds
        """
        action_type = translated_action["action_type"]
        target = translated_action["target"]

        # Store time in translated_action for downstream use
        translated_action["time"] = current_sim_time

        logger.info(
            f"Agent {agent_id}: {action_type} to {target} "
            f"(confidence: {translated_action['confidence']:.2f}) - {translated_action['reasoning']}"
        )

        try:
            # Apply pace-based speed modulation from the cognitive layer.
            pace = translated_action.get("pace")
            if isinstance(pace, str):
                self._apply_pace_speed(agent_id, pace)

            # Backward-compatible speed handling for legacy prompt schemas.
            speed_str = translated_action.get("speed")
            if speed_str:
                speed_ms = convert_speed_to_ms(speed_str)
                if speed_ms:
                    self.jps_sim.set_agent_speed(agent_id, speed_ms)
                    logger.debug(f"Set {agent_id} speed to {speed_ms:.2f} m/s ({speed_str})")

            # Resolve target_agent to actual position if specified
            target_agent_id = translated_action.get("target_agent")
            target_type = translated_action.get("target_type", "")

            if target_agent_id and action_type == "move":
                target_position = self.state_queries.get_agent_position(target_agent_id)
                if target_position is not None:
                    if target_type == "agent":
                        agent_pos = self.state_queries.get_agent_position(agent_id)
                        target_position = self._safe_follow_target(
                            agent_id, agent_pos, target_position
                        )
                    target = target_position
                    translated_action["target"] = target

                    # For "following" type, optionally match speed for coordinated movement
                    if target_type == "agent":
                        logger.debug(f"{agent_id} following {target_agent_id} at {target}")

                        if agent_pos is not None:
                            distance = (
                                (agent_pos[0] - target_position[0]) ** 2
                                + (agent_pos[1] - target_position[1]) ** 2
                            ) ** 0.5

                            # Match speed if close enough and no explicit speed set
                            if distance < 10.0 and not speed_str:
                                try:
                                    target_speed = self.jps_sim.get_agent_speed(target_agent_id)
                                    if target_speed:
                                        self.jps_sim.set_agent_speed(agent_id, target_speed)
                                        logger.debug(
                                            f"{agent_id} matching {target_agent_id}'s speed: "
                                            f"{target_speed:.2f} m/s (distance: {distance:.1f}m)"
                                        )
                                except Exception:
                                    pass  # If speed matching fails, continue with default
                    else:
                        # Regular agent approach (not following)
                        logger.debug(f"{agent_id} moving toward {target_agent_id} at {target}")

            if action_type == "continue":
                self._handle_continue_action(agent_id)
            elif action_type == "seek_information":
                self._handle_seek_information_action(agent_id, translated_action)
            elif action_type == "move" and target:
                self._handle_move_action(agent_id, translated_action, target)
            elif action_type == "wait":
                self._handle_wait_action(agent_id, translated_action, current_sim_time)

            # Store decision for agent memory (Concordia natural language context)
            # This allows agents to reference their previous commitment in future decisions
            self.agent_last_decision[agent_id] = {
                "time": current_sim_time,
                "action_type": action_type,
                "target_type": translated_action.get("target_type"),
                "target_agent": translated_action.get("target_agent"),
                "target_exit": translated_action.get("target_exit"),
                "zone_name": translated_action.get("zone_name"),
                "speed": translated_action.get("speed"),
                "reasoning": translated_action.get("reasoning"),
                "wait_reason": translated_action.get("wait_reason"),
            }

        except Exception as e:
            logger.error(
                f"Failed to apply action for {agent_id}: {e}\n"
                f"  Action type: {action_type}\n"
                f"  Target: {target}\n"
                f"  Translated action: {translated_action}"
            )
            import traceback

            logger.debug(f"Traceback: {traceback.format_exc()}")

    def _apply_pace_speed(self, agent_id: str, pace: str) -> None:
        """Apply pace multiplier to the agent's current desired speed."""
        if pace == "normal_pace":
            return
        if pace == "hurrying":
            if HURRYING_MULTIPLIER is None:
                raise ValueError("HURRYING_MULTIPLIER is unset for pace='hurrying'")
            multiplier = HURRYING_MULTIPLIER
        elif pace == "running":
            if RUNNING_MULTIPLIER is None:
                raise ValueError("RUNNING_MULTIPLIER is unset for pace='running'")
            multiplier = RUNNING_MULTIPLIER
        else:
            return

        try:
            current_speed = self.jps_sim.get_agent_speed(agent_id)
            if current_speed is None:
                return
            self.jps_sim.set_agent_speed(agent_id, float(current_speed) * multiplier)
        except Exception:
            # Keep execution robust if speed APIs are unavailable in some backends.
            return

    def _handle_continue_action(self, agent_id: str) -> None:
        """Continue along the currently assigned journey without changing target."""
        if self.agent_destinations.get(agent_id):
            self.agent_action[agent_id] = "moving"
            logger.debug(f"[CONTINUE] {agent_id} preserving current journey")
        else:
            # No known destination to continue toward; hold current state safely.
            self.agent_action[agent_id] = "waiting"
            logger.debug(f"[CONTINUE] {agent_id} had no active destination; holding state")

    def _handle_seek_information_action(
        self,
        agent_id: str,
        translated_action: dict[str, Any],
    ) -> None:
        """Route to a deterministic nearby information source when available."""
        current_position = self.state_queries.get_agent_position(agent_id)
        if current_position is None:
            return

        zone_id = self._identify_zone(current_position)
        source = self._resolve_information_source(agent_id, zone_id)
        if source is not None:
            snapped = self._safe_follow_target(agent_id, current_position, source["position"])
            self.jps_sim.set_agent_target(agent_id, snapped)
            self.agent_action[agent_id] = "moving"
            translated_action["resolved_info_source"] = source["id"]
            logger.debug(f"[SEEK_INFO] {agent_id} routing to {source['id']}")
            return

        self.agent_action[agent_id] = "waiting"
        self.jps_sim.set_agent_target(agent_id, current_position)
        translated_action["resolved_info_source"] = None
        logger.debug(f"[SEEK_INFO] {agent_id} no reachable source found; waiting in place")

    def has_reachable_information_source(
        self,
        agent_id: str,
        position: tuple[float, float],
        zone_id: str | None,
    ) -> bool:
        """Return True when deterministic seek-information resolver has a target."""
        _ = position  # position kept for future geometric checks.
        return self._resolve_information_source(agent_id, zone_id) is not None

    def _identify_zone(self, position: tuple[float, float]) -> str | None:
        """Return the finest-grained named zone covering this position."""
        zones_polygons = self.station_layout.get("zones_polygons", {})
        if not zones_polygons:
            return None
        from shapely.geometry import Point

        pt = Point(position)
        best_zone = None
        best_area = float("inf")
        for zone_name, poly in zones_polygons.items():
            try:
                if poly.covers(pt) or poly.contains(pt):
                    area = poly.area
                    if area < best_area:
                        best_area = area
                        best_zone = zone_name
            except Exception:
                continue
        return best_zone

    def _staff_candidates(self) -> list[str]:
        """Return IDs of likely information-providing staff/responder agents."""
        out: list[str] = []
        for aid, role in self.agent_roles.items():
            role_l = str(role).lower()
            if any(k in role_l for k in ("staff", "rci", "responder", "fire")):
                out.append(aid)
        return out

    def _resolve_information_source(
        self,
        agent_id: str,
        zone_id: str | None,
    ) -> dict[str, Any] | None:
        """Deterministically resolve seek_information source with v1.1 priority."""
        if zone_id is None:
            return None

        adjacent = set(self._zone_adjacency.get(zone_id, []))
        staff_ids = self._staff_candidates()

        staff_current: list[dict[str, Any]] = []
        staff_adjacent: list[dict[str, Any]] = []
        for sid in staff_ids:
            if sid == agent_id:
                continue
            pos = self.state_queries.get_agent_position(sid)
            if pos is None:
                continue
            s_zone = self._identify_zone(pos)
            if s_zone == zone_id:
                staff_current.append({"id": f"staff:{sid}", "position": pos})
            elif s_zone in adjacent:
                staff_adjacent.append({"id": f"staff:{sid}", "position": pos})

        boards_current: list[dict[str, Any]] = []
        boards_adjacent: list[dict[str, Any]] = []
        for board in self._indicator_boards:
            b_zone = board.get("zone_id")
            b_pos = board.get("position")
            if not b_zone or not b_pos:
                continue
            try:
                b_position = (float(b_pos[0]), float(b_pos[1]))
            except Exception:
                continue
            b_id = str(board.get("id", f"board:{b_zone}"))
            if b_zone == zone_id:
                boards_current.append({"id": b_id, "position": b_position})
            elif b_zone in adjacent:
                boards_adjacent.append({"id": b_id, "position": b_position})

        if staff_current:
            return staff_current[0]
        if boards_current:
            return boards_current[0]
        if staff_adjacent:
            return staff_adjacent[0]
        if boards_adjacent:
            return boards_adjacent[0]
        return None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_walkable_areas_for_agent(self, agent_id: str) -> dict:
        """Return the walkable-area polygons for the level the agent is on."""
        if hasattr(self.jps_sim, "simulations"):
            agent_level = self.jps_sim.agent_levels.get(agent_id, "0")
            sim = self.jps_sim.simulations.get(agent_level)
            if sim is not None:
                return sim.geometry_manager.walkable_areas_with_obstacles
        elif hasattr(self.jps_sim, "geometry_manager"):
            return self.jps_sim.geometry_manager.walkable_areas_with_obstacles
        return {}

    def _safe_follow_target(
        self,
        agent_id: str,
        agent_pos: tuple[float, float] | None,
        target_pos: tuple[float, float],
    ) -> tuple[float, float]:
        """
        Return a waypoint toward *target_pos* that lies strictly inside a walkable
        polygon.  If the raw target is already valid it is returned unchanged.

        Otherwise, the nearest point on the boundary of the closest walkable
        polygon is found and nudged 0.1 m inward toward that polygon's centroid.
        Falls back to the agent's own position if everything else fails.
        """
        from shapely.geometry import Point
        from shapely.ops import nearest_points as shapely_nearest_points

        walkable_areas = self._get_walkable_areas_for_agent(agent_id)
        if not walkable_areas:
            return target_pos

        p = Point(target_pos)

        # Fast path: already inside
        for poly in walkable_areas.values():
            if poly.contains(p):
                return target_pos

        # Find the nearest point on the boundary of the closest polygon
        best_poly = None
        best_boundary_pt = None
        best_dist = float("inf")
        for poly in walkable_areas.values():
            boundary_pt = shapely_nearest_points(p, poly)[1]
            d = p.distance(boundary_pt)
            if d < best_dist:
                best_dist = d
                best_boundary_pt = boundary_pt
                best_poly = poly

        if best_poly is None or best_boundary_pt is None:
            return agent_pos if agent_pos is not None else target_pos

        # Nudge the boundary point slightly inward toward the polygon centroid
        cx, cy = best_poly.centroid.x, best_poly.centroid.y
        bx, by = best_boundary_pt.x, best_boundary_pt.y
        dx, dy = cx - bx, cy - by
        length = (dx * dx + dy * dy) ** 0.5
        if length > 0:
            _nudge = 0.1  # metres inward
            candidate = (bx + dx / length * _nudge, by + dy / length * _nudge)
            if best_poly.contains(Point(candidate)):
                logger.debug(
                    f"[FOLLOW] {agent_id}: target {target_pos} outside walkable area; "
                    f"snapped to nearest interior point {candidate}"
                )
                return candidate

        # Centroid nudge landed outside (very thin polygon edge case) - stay put
        fallback = agent_pos if agent_pos is not None else target_pos
        logger.debug(f"[FOLLOW] {agent_id}: nearest-point snap failed; using fallback {fallback}")
        return fallback

    def _is_in_escalator_context(
        self, agent_id: str, position: tuple[float, float] | None
    ) -> bool:
        """Return True when an agent is inside an escalator corridor or transfer zone."""
        if position is None:
            return False

        from shapely.geometry import Point

        p = Point(position)

        # Multi-level: check per-level corridors + named transfer zones
        if hasattr(self.jps_sim, "simulations") and hasattr(self.jps_sim, "agent_levels"):
            level_id = self.jps_sim.agent_levels.get(agent_id)
            if level_id is None:
                return False

            level_sim = self.jps_sim.simulations.get(level_id)
            if level_sim is None:
                return False

            corridors = getattr(level_sim.geometry_manager, "escalator_corridors", {})
            for poly in corridors.values():
                if poly.covers(p) or poly.contains(p):
                    return True

            controller = getattr(self.jps_sim, "escalator_controller", None)
            if controller is not None:
                for zone_name in controller.get_level_zone_names(level_id):
                    zone_poly = controller.get_zone_polygon(zone_name)
                    if zone_poly is not None and (zone_poly.covers(p) or zone_poly.contains(p)):
                        return True
            else:
                transfer_manager = getattr(self.jps_sim, "transfer_manager", None)
                zones = getattr(transfer_manager, "escalator_zones", {}) if transfer_manager else {}
                for zone_poly in zones.values():
                    if zone_poly.covers(p) or zone_poly.contains(p):
                        return True
            return False

        # Single-level fallback: only corridor detection is available
        geometry_manager = getattr(self.jps_sim, "geometry_manager", None)
        corridors = getattr(geometry_manager, "escalator_corridors", {}) if geometry_manager else {}
        for poly in corridors.values():
            if poly.covers(p) or poly.contains(p):
                return True
        return False

    def _handle_move_action(self, agent_id: str, translated_action: dict[str, Any], target):
        """Handle move action: agent moving to exit, waypoint, or toward another agent."""
        # Update action state
        self.agent_action[agent_id] = "moving"

        target_agent = translated_action.get("target_agent")
        target_type = translated_action.get("target_type", "")

        if target_agent and target_type == "agent":
            # Agent is following someone - this is coordinated movement, not helping detection
            logger.debug(f"{agent_id} following {target_agent}")
        elif target_agent:
            # Regular agent approach (not following)
            logger.debug(f"{agent_id} moving toward {target_agent}")

        # Log current agent position and target for debugging
        agent_pos = self.state_queries.get_agent_position(agent_id)
        logger.debug(f"[MOVE] {agent_id} at {agent_pos} → target {target}")

        # Extract the NEW exit name from this action (if moving to an exit)
        new_exit_name = extract_exit_name(translated_action, self.station_layout)

        if new_exit_name:
            # Skip redundant switch_agent_journey calls when the destination is
            # unchanged.  Re-issuing the same JuPedSim journey for an agent that
            # is already routing to this exit (e.g. a queued agent whose cached
            # decision is re-applied each cycle) triggers a crowd-force
            # recalculation burst that can push agents sideways inside narrow
            # escalator corridors and eject agents from transfer zones.
            current_dest = self.agent_destinations.get(agent_id)
            if current_dest == new_exit_name:
                logger.debug(
                    f"[MOVE] {agent_id} already routing to '{new_exit_name}' — skip journey reset"
                )
                return
            # Track destination and commit the selected exit directly.
            # Do not substitute with fallback waypoint routing on failure.
            self.agent_destinations[agent_id] = new_exit_name
            logger.debug(
                f"[MOVE] {agent_id} set_agent_destination_exit({agent_id}, {new_exit_name})"
            )
            self.jps_sim.set_agent_destination_exit(agent_id, new_exit_name)
            logger.debug(f"Switched {agent_id} to journey for {new_exit_name}")
        else:
            # Not moving to an exit, just a waypoint — snap to walkable area first so
            # that zone centroids landing on obstacles or outside the geometry don't
            # cause "WayPoint not inside walkable area" errors.
            snapped = self._safe_follow_target(agent_id, agent_pos, target)
            if snapped != target:
                logger.debug(
                    f"[MOVE] {agent_id} zone target {target} outside walkable area; "
                    f"snapped to {snapped}"
                )
            logger.debug(f"[MOVE] {agent_id} set_agent_target({agent_id}, {snapped}) - waypoint")
            self.jps_sim.set_agent_target(agent_id, snapped)

    def _handle_wait_action(
        self, agent_id: str, translated_action: dict[str, Any], current_sim_time: float
    ):
        """Handle wait action: agent staying in place or seeking information."""
        wait_reason = translated_action.get("wait_reason", "unspecified")

        current_position = self.state_queries.get_agent_position(agent_id)

        if self._is_in_escalator_context(agent_id, current_position):
            logger.warning(
                f"[WAIT] {agent_id} chose wait in escalator context; no automatic reroute applied"
            )

        # Update action state (normal wait handling outside escalator contexts)
        self.agent_action[agent_id] = "waiting"

        # Different behavior based on wait reason
        if wait_reason == "seeking_information":
            # Seeking information: move slowly in a small random direction (looking around)
            # Generate a random nearby point within 3-5 meters
            from shapely.geometry import Point

            distance = random.uniform(3.0, 5.0)
            angle = random.uniform(0, 2 * math.pi)
            target_x = current_position[0] + distance * math.cos(angle)
            target_y = current_position[1] + distance * math.sin(angle)

            target_pos = (target_x, target_y)

            # Validate that target is inside a walkable area
            target_point = Point(target_x, target_y)
            valid_target = False

            # Try to get walkable areas - handle both single and multi-level simulations
            walkable_areas = {}
            if hasattr(self.jps_sim, "simulations"):
                # Multi-level: get walkable areas from current agent's level
                agent_level = self.jps_sim.agent_levels.get(agent_id, "0")
                if agent_level in self.jps_sim.simulations:
                    walkable_areas = self.jps_sim.simulations[
                        agent_level
                    ].geometry_manager.walkable_areas_with_obstacles
            elif hasattr(self.jps_sim, "geometry_manager"):
                # Single-level: use geometry manager directly
                walkable_areas = self.jps_sim.geometry_manager.walkable_areas_with_obstacles

            # Also try station_layout as fallback
            if not walkable_areas:
                walkable_areas = self.station_layout.get("walkable_areas", {})

            # Check if target is in walkable area
            for poly in walkable_areas.values():
                if poly.contains(target_point):
                    valid_target = True
                    break

            if not valid_target:
                # Generated point is outside walkable area - use current position instead
                logger.warning(
                    f"[WAIT] {agent_id} seeking_information waypoint {target_pos} is outside walkable area, "
                    f"using current position {current_position} instead"
                )
                target_pos = current_position

            logger.debug(
                f"[WAIT] {agent_id} at {current_position} seeking information → random waypoint {target_pos} ({distance:.1f}m away)"
            )
            self.jps_sim.set_agent_target(agent_id, target_pos)
            logger.debug(
                f"{agent_id} seeking information - moving slowly to nearby point "
                f"({distance:.1f}m away)"
            )
        else:
            # If the agent is already committed to boarding a train, preserve that
            # routing rather than switching to a stand-still waypoint.  The "wait"
            # semantically means the agent is heading to the platform doors and
            # waiting for the train — they should keep walking there.
            current_dest = self.agent_destinations.get(agent_id, "")
            if current_dest.startswith("train_platform_"):
                logger.debug(
                    f"[WAIT] {agent_id} routed to '{current_dest}' — preserving train boarding route"
                )
            else:
                # All other wait types: stand still at current position
                logger.debug(f"[WAIT] {agent_id} at {current_position} staying still")
                self.jps_sim.set_agent_target(agent_id, current_position)

        # Record wait event
        agent_config = next((c for c in self.agent_configs if c["id"] == agent_id), {})
        self.wait_events.append(
            {
                "time": current_sim_time,
                "agent": agent_id,
                "personality": agent_config.get("personality_type", "UNKNOWN"),
                "wait_reason": wait_reason if wait_reason else "unspecified",
                "location": current_position,
            }
        )
