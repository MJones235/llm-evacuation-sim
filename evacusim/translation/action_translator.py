"""
Translates natural language actions from Concordia agents into JuPedSim waypoints and goals.

Examples:
    "I will evacuate through the north exit" → waypoint at north exit
    "I will wait here for more information" → stay in current position
    "I will help the person nearby" → move toward nearest agent
"""

import json
import math
import re
from typing import Any

from concordia.language_model import language_model

from evacusim.utils.logger import get_logger
from evacusim.translation.exit_name_registry import (
    build_registry_from_station_layout,
)

logger = get_logger(__name__)


# Precompiled regex to normalise geometry zone names (e.g. "L0_esc_a_down") to
# canonical escalator exit IDs ("escalator_a_down") at translation time.
_ESC_ZONE_RE = re.compile(r"^L[^_]+_esc_([a-f])_(up|down)$")


class ActionTranslator:
    """
    Translates natural language actions from Concordia agents into
    JuPedSim waypoints and goals.
    """

    def __init__(
        self,
        station_layout: dict[str, Any],
        model: language_model.LanguageModel | None = None,
        jps_sim=None,
    ):
        """
        Initialize the action translator.

        Args:
            station_layout: Dictionary with station geometry info (exits, zones, etc.)
            model: Optional LLM for ambiguous action parsing
            jps_sim: JuPedSim simulation instance (for multi-level exit lookup)
        """
        self.station_layout = station_layout
        self.model = model
        self.jps_sim = jps_sim

        # Define exit locations from layout (street-level exits)
        self.exits = station_layout.get("exits", {})
        self.zones = station_layout.get("zones", {})
        self.zones_polygons = station_layout.get("zones_polygons", {})

        # Build exit name registry for natural language resolution
        self.exit_registry = build_registry_from_station_layout(station_layout, jps_sim)

        # Down-access escalator zones on the concourse (level 0) that lead to platforms.
        # Agents can now explicitly choose these as exits; this dict provides their
        # coordinates for translation.
        self._level0_down_esc_centroids: dict[str, tuple[float, float]] = station_layout.get(
            "down_access_exits", {}
        )
        # Keep a mapping from platform zone name -> down escalator zone for the
        # safety-net redirect (catches LLM hallucinations of platform zones on level 0).
        self._platform_down_exits: dict[str, list[str]] = station_layout.get(
            "platform_down_exits", {}
        )

    def translate(
        self, agent_id: str, action: str, current_position: tuple[float, float]
    ) -> dict[str, Any]:
        """
        Translate a JSON action response to a concrete goal.

        Args:
            agent_id: ID of the acting agent
            action: JSON action from Concordia agent
            current_position: Agent's current (x, y) position

        Returns:
            Dictionary with:
                - action_type: "move", "wait", "help", "follow"
                - target: Target coordinates (x, y) or agent ID
                - confidence: Parsing confidence (0-1)
                - reasoning: Explanation of translation
        """
        # Get agent's level for multi-level exit lookup
        agent_level = None
        if self.jps_sim and hasattr(self.jps_sim, "agent_levels"):
            agent_level = self.jps_sim.agent_levels.get(agent_id)
        # Try parsing as JSON first
        try:
            # Strip agent name prefix (e.g., "Agent 0 {" -> "{")
            json_start = action.find("{")
            if json_start > 0:
                action = action[json_start:]

            data = json.loads(action)
            action_type = data.get("action_type")
            target_type = data.get("target_type")
            exit_name = data.get("exit_name")
            zone_name = data.get("zone_name")

            # Reject placeholder/invalid exit names that the LLM sometimes returns
            _INVALID_EXIT_NAMES = {
                "none",
                "null",
                "n/a",
                "nearest exit",
                "nearest",
                "concourse",
                "street exit",
                "unknown",
                "exit",
                "",
            }
            if exit_name and str(exit_name).lower().strip() in _INVALID_EXIT_NAMES:
                exit_name = None
            target_agent = data.get("target_agent")  # Agent ID to move toward or follow
            wait_reason = data.get("wait_reason")  # Phase 4.3: Information seeking
            speed = data.get("speed")  # Phase 4.3: Dynamic speed selection

            # Move toward another agent (helping, following, or approaching)
            # ONLY if target_type explicitly says so (not if just mentioned in context)
            if action_type == "move" and target_agent and target_type == "agent":
                return {
                    "action_type": "move",
                    "target": None,  # Will be resolved to agent position later
                    "target_agent": target_agent,
                    "target_type": "agent",
                    "confidence": 0.9,
                    "reasoning": f"Moving toward {target_agent}",
                    "speed": speed,
                }

            if action_type == "wait" or target_type == "current_position":
                # Phase 4.3: Include wait reason if provided
                wait_reason_str = f" ({wait_reason})" if wait_reason else ""
                # Default speed for wait actions: slow_walk for seeking_information, otherwise null (keep current)
                default_speed = "slow_walk" if wait_reason == "seeking_information" else None
                return {
                    "action_type": "wait",
                    "target": current_position,
                    "confidence": 0.9,
                    "reasoning": f"Agent chose to wait at current position{wait_reason_str}",
                    "wait_reason": wait_reason,  # Pass through for tracking
                    "speed": speed or default_speed,  # Use LLM speed if provided, else default
                }

            if action_type == "move" and target_type == "exit":
                if exit_name:
                    exit_coords = self._get_exit_coordinates(exit_name, agent_level)
                    resolved_id = self.exit_registry.resolve_to_id(exit_name)
                else:
                    exit_coords = None
                    resolved_id = None

                if exit_coords:
                    # Normalise the resolved ID: zone-form names like "L0_esc_a_down"
                    # must become "escalator_a_down" so set_agent_evacuation_exit
                    # finds them in the level's evacuation_exits dict.
                    _raw_resolved = resolved_id or exit_name
                    _m = re.match(r"^L[^_]+_esc_([a-f])_(up|down)$", _raw_resolved)
                    _canonical_resolved = (
                        f"escalator_{_m.group(1)}_{_m.group(2)}" if _m else _raw_resolved
                    )
                    # Log successful resolution if display name was converted
                    if _canonical_resolved and _canonical_resolved != exit_name:
                        logger.debug(
                            f"Agent {agent_id} exit name '{exit_name}' → resolved to '{_canonical_resolved}'"
                        )
                    return {
                        "action_type": "move",
                        "target": exit_coords,
                        "target_type": "exit",
                        "exit_name": exit_name,
                        "resolved_exit_id": _canonical_resolved,
                        "confidence": 0.95,
                        "reasoning": f"Moving to known exit {exit_name}",
                        "speed": speed,  # Phase 4.3: Dynamic speed
                    }
                else:
                    # Exit not available on this level.  If the resolved exit
                    # lives on a different (higher) level, transparently redirect
                    # the agent to the nearest UP escalator on their current level
                    # so they make progress toward the requested exit.
                    if resolved_id and agent_level:
                        redirect = self._redirect_to_next_level_exit(
                            agent_id, resolved_id, agent_level, current_position
                        )
                        if redirect is not None:
                            redirect["speed"] = redirect.get("speed") or speed
                            return redirect
                    if resolved_id:
                        logger.warning(
                            f"Agent {agent_id} exit name '{exit_name}' resolved to '{resolved_id}' "
                            f"but no coordinates found (may not be on level {agent_level})"
                        )
                    else:
                        known_names = self.exit_registry.get_all_display_names()[:5]
                        logger.warning(
                            f"Agent {agent_id} requested unknown exit '{exit_name}'. "
                            f"Could not resolve to any exit ID. Examples of known exits: {known_names}"
                        )

            if action_type == "move" and target_type == "zone" and zone_name:
                zone_target = self._find_zone_target(zone_name.lower())
                if zone_target:
                    zone_name_resolved, zone_coords = zone_target
                    # Safety net: if a level-0 agent somehow requests a platform zone
                    # (which is physically on level -1), redirect to the appropriate
                    # down-access escalator rather than sending them to invalid coords.
                    import re as _re

                    if agent_level == "0" and _re.match(r"^platform_", zone_name_resolved.lower()):
                        # Look up the correct escalator for this specific platform
                        down_zones = self._platform_down_exits.get(zone_name_resolved.lower(), [])
                        if not down_zones:
                            # Fall back to any available down escalator
                            down_zones = list(self._level0_down_esc_centroids.keys())[:1]
                        for esc_zone in down_zones:
                            esc_coords = self._level0_down_esc_centroids.get(esc_zone)
                            if esc_coords:
                                logger.warning(
                                    f"{agent_id} (level 0) requested platform zone "
                                    f"'{zone_name_resolved}' — redirecting to {esc_zone}"
                                )
                                return {
                                    "action_type": "move",
                                    "target": esc_coords,
                                    "target_type": "zone",
                                    "zone_name": zone_name_resolved,
                                    "confidence": 0.6,
                                    "reasoning": "Redirected from platform zone to down escalator",
                                    "speed": speed,
                                }
                    return {
                        "action_type": "move",
                        "target": zone_coords,
                        "target_type": "zone",
                        "zone_name": zone_name_resolved,
                        "confidence": 0.9,
                        "reasoning": f"Moving to zone {zone_name_resolved}",
                        "speed": speed,  # Phase 4.3: Dynamic speed
                    }

        except json.JSONDecodeError:
            logger.warning(f"Failed to parse action as JSON: {action[:100]}")

        return {
            "action_type": "wait",
            "target": current_position,
            "target_type": "current_position",
            "confidence": 0.3,
            "reasoning": f"Parse failed, defaulting to wait: {action[:100]}",
        }

    def _redirect_to_next_level_exit(
        self,
        agent_id: str,
        resolved_exit_id: str,
        agent_level: str,
        current_position: tuple[float, float],
    ) -> dict[str, Any] | None:
        """
        When an agent requests an exit that doesn't exist on their current level,
        find the nearest exit on the current level that leads toward it.

        For platform agents (level -1) requesting a street exit: redirect to the
        nearest UP escalator on level -1.
        For concourse agents (level 0) requesting a platform escalator: redirect
        to the nearest DOWN escalator on level 0.

        Returns a translated-action dict, or None if no redirect is possible.
        """
        if not (self.jps_sim and hasattr(self.jps_sim, "simulations")):
            return None

        level_sim = self.jps_sim.simulations.get(agent_level)
        if level_sim is None:
            return None

        level_exits = level_sim.exit_manager.exit_coordinates

        # Determine which direction this agent needs to travel.
        # Street exits (no "escalator_" prefix) are on level 0 → agents on level -1
        # must go UP.  Down-escalator exits (escalator_*_down) are used to reach
        # level -1 → agents on level 0 must use those.
        is_street_exit = not resolved_exit_id.startswith("escalator_")
        is_down_escalator = resolved_exit_id.endswith("_down")

        if is_street_exit:
            # Agent needs to go up: find UP escalators on current level.
            candidates = {
                name: coords for name, coords in level_exits.items()
                if name.startswith("escalator_") and name.endswith("_up")
            }
            direction_label = "up escalator"
        elif is_down_escalator:
            # Agent wants to go down: find DOWN escalators on current level.
            candidates = {
                name: coords for name, coords in level_exits.items()
                if name.startswith("escalator_") and name.endswith("_down")
            }
            direction_label = "down escalator"
            # If no down escalators exist on this level (e.g. agent is already on
            # the lowest level), fall back to UP escalators — the agent is confused
            # about their location and needs to go up to reach the concourse first.
            if not candidates:
                candidates = {
                    name: coords for name, coords in level_exits.items()
                    if name.startswith("escalator_") and name.endswith("_up")
                }
                direction_label = "up escalator (redirected from down request)"
        else:
            return None

        if not candidates:
            return None

        nearest_name = min(
            candidates,
            key=lambda name: math.hypot(
                current_position[0] - candidates[name][0],
                current_position[1] - candidates[name][1],
            ),
        )
        nearest_coords = candidates[nearest_name]

        _m = re.match(r"^L[^_]+_esc_([a-f])_(up|down)$", nearest_name)
        canonical_name = (
            f"escalator_{_m.group(1)}_{_m.group(2)}" if _m else nearest_name
        )

        logger.info(
            f"Agent {agent_id} on level {agent_level} requested '{resolved_exit_id}' "
            f"(not on this level) — redirecting to {direction_label} '{canonical_name}'"
        )
        return {
            "action_type": "move",
            "target": nearest_coords,
            "target_type": "exit",
            "exit_name": self.exit_registry.get_display_name(canonical_name),
            "resolved_exit_id": canonical_name,
            "confidence": 0.8,
            "reasoning": (
                f"'{resolved_exit_id}' is not on level {agent_level}; "
                f"routing via {canonical_name} to reach it"
            ),
        }

    def _get_exit_coordinates(
        self, exit_name: str, agent_level: str | None = None
    ) -> tuple[float, float] | None:
        """
        Get exit coordinates, resolving natural language names to technical IDs.

        Handles exit name variations that may come from LLM:
        - "Blackett Street" -> "blackett_street"
        - "Grey Street Exit" -> "grey_street"
        - "Escalator B" -> "escalator_b_up"
        - "escalator b going up" -> "escalator_b_up"

        Args:
            exit_name: Name of the exit (natural language or technical ID)
            agent_level: Agent's current level (for multi-level lookup)

        Returns:
            (x, y) coordinates or None if not found
        """
        # Use registry to resolve natural language to technical ID
        resolved_id = self.exit_registry.resolve_to_id(exit_name)

        if resolved_id is None:
            # Log helpful message for debugging
            logger.debug(
                f"Could not resolve exit name '{exit_name}' to any known exit ID. "
                f"Known exits: {list(self.exit_registry.get_all_ids())[:10]}"
            )
            return None

        # For multi-level simulations, only return coordinates for exits on the
        # agent's current level.
        if agent_level and self.jps_sim and hasattr(self.jps_sim, "simulations"):
            level_sim = self.jps_sim.simulations.get(agent_level)
            level_exits = (
                level_sim.exit_manager.exit_coordinates
                if level_sim and hasattr(level_sim, "exit_manager")
                else {}
            )

            # 1. Direct lookup by resolved ID (covers street exits + exact escalator IDs)
            if resolved_id in level_exits:
                return level_exits[resolved_id]

            # 2. Escalator: only the letter matters.  The registry may have resolved to
            #    the wrong direction for this level (e.g. "Escalator B" → escalator_b_up
            #    but the agent is on level 0 where only _down escalators are exits).
            #    Also handles zone-name form: L0_esc_d_down → letter 'd'.
            #    Find any escalator with the same letter that IS valid on this level.
            import re as _re

            m = _re.match(r"^escalator_([a-f])_(?:up|down)$", resolved_id) or _re.match(
                r"^L[^_]+_esc_([a-f])_(?:up|down)$", resolved_id
            )
            if m:
                letter = m.group(1)
                for key, coords in level_exits.items():
                    if _re.match(rf"^escalator_{letter}_(?:up|down)$", key):
                        logger.debug(
                            f"Escalator '{resolved_id}' not on level {agent_level} "
                            f"— using '{key}' (same letter, valid on this level)"
                        )
                        return coords

            # 3. Pre-blocked exits: the TZ polygon was removed from the navmesh
            #    at startup, so it's absent from exit_coordinates.  Fall back to
            #    the centroid recorded by geometry_manager before removal so the
            #    agent can navigate to the nearest accessible point and then
            #    receive a "blocked" observation at close range.
            if level_sim and hasattr(level_sim, "geometry_manager"):
                gm = level_sim.geometry_manager
                blocked_pos = gm.blocked_exit_positions.get(resolved_id)
                if blocked_pos is not None:
                    logger.debug(
                        f"Exit '{resolved_id}' is pre-blocked; using stored centroid {blocked_pos}"
                    )
                    return blocked_pos

            # Exit not available on this level
            return None

        # Single-level simulation: check station_layout exits
        if resolved_id in self.exits:
            return self.exits[resolved_id]

        return None

    def _find_nearest_exit(
        self, position: tuple[float, float], agent_level: str | None = None
    ) -> dict[str, Any]:
        """Find the nearest exit to a given position (level-aware)."""
        # Get all exits available to this agent (from their current level)
        available_exits = dict(self.exits)  # Start with station_layout exits

        # Add level-specific exits (escalators, etc.) if multi-level
        if agent_level and self.jps_sim and hasattr(self.jps_sim, "simulations"):
            level_sim = self.jps_sim.simulations.get(agent_level)
            if level_sim and hasattr(level_sim, "exit_manager"):
                if hasattr(level_sim.exit_manager, "exit_coordinates"):
                    available_exits.update(level_sim.exit_manager.exit_coordinates)

        min_dist = float("inf")
        nearest = None

        for exit_name, exit_coords in available_exits.items():
            dist = (
                (position[0] - exit_coords[0]) ** 2 + (position[1] - exit_coords[1]) ** 2
            ) ** 0.5
            if dist < min_dist:
                min_dist = dist
                nearest = {"name": exit_name, "coords": exit_coords}

        return nearest if nearest else {"name": "default", "coords": (0, 0)}

    def _find_zone_target(self, text: str) -> tuple[str, tuple[float, float]] | None:
        """Find zone coordinates from zone name in text.

        Uses ``representative_point()`` (guaranteed inside the polygon) instead of
        ``centroid`` (which can fall outside concave shapes such as L- or U-shaped
        concourses).  The executor further snaps the point to the walkable area in
        case the zone polygon extends beyond the actual navigable geometry.
        """
        for zone_name in self.zones_polygons.keys():
            if zone_name.lower() in text:
                polygon = self.zones_polygons[zone_name]
                # representative_point() always lies inside the polygon; centroid does not.
                pt = polygon.representative_point()
                return zone_name, (pt.x, pt.y)

        for zone_name, zone_bounds in self.zones.items():
            if zone_name.lower() in text:
                x_center = (zone_bounds["x_min"] + zone_bounds["x_max"]) / 2
                y_center = (zone_bounds["y_min"] + zone_bounds["y_max"]) / 2
                return zone_name, (x_center, y_center)

        return None
