"""Translate decision payloads into pedestrian-simulation commands.

``evacuate`` becomes a move to the chosen exit's position on the agent's
level, ``leave_by_train`` a move to the nearest train, and ``wait``,
``continue_activity`` and ``seek_information`` keep their meaning; the
action executor applies the result.
"""

import math
import re
from typing import Any

from evacusim.conventions import is_train_exit
from evacusim.translation.exit_name_registry import (
    build_registry_from_station_layout,
)
from evacusim.utils.logger import get_logger

logger = get_logger(__name__)


_ESC_ZONE_DOTTED_RE = re.compile(r"^esc\.([A-F])\.zone\.(concourse|platform)\.(departure|arrival)$")


class ActionTranslator:
    """Locates exits and turns decision payloads into simulation commands."""

    def __init__(
        self,
        station_layout: dict[str, Any],
        jps_sim=None,
    ):
        """
        Initialize the action translator.

        Args:
            station_layout: Dictionary with station geometry info (exits, zones, etc.)
            jps_sim: JuPedSim simulation instance (for multi-level exit lookup)
        """
        self.station_layout = station_layout
        self.jps_sim = jps_sim

        # Define exit locations from layout (street-level exits)
        self.exits = station_layout.get("exits", {})
        self.zones = station_layout.get("zones", {})
        self.zones_polygons = station_layout.get("zones_polygons", {})

        # Build exit name registry for natural language resolution
        self.exit_registry = build_registry_from_station_layout(station_layout, jps_sim)

    def translate(
        self, agent_id: str, decision: dict[str, Any], current_position: tuple[float, float]
    ) -> dict[str, Any]:
        """Turn a validated decision payload into a pedestrian-simulation command.

        Args:
            agent_id: The deciding agent.
            decision: A payload valid for the agent's offered set
                (:mod:`evacusim.decision.payload`).
            current_position: The agent's (x, y) position.

        Returns:
            A command dict whose ``action_type`` is ``continue``, ``wait``,
            ``seek_information`` or ``move`` (with a ``target`` position and,
            for exits, ``exit_name``), plus the decision's ``pace``. If the
            chosen exit or train cannot be located, the agent waits.
        """
        agent_level = None
        if self.jps_sim and hasattr(self.jps_sim, "agent_levels"):
            agent_level = self.jps_sim.agent_levels.get(agent_id)

        verb = decision.get("action")
        pace = decision.get("pace")

        if verb == "continue_activity":
            return {
                "action_type": "continue",
                "target": None,
                "target_type": "journey",
                "confidence": 0.95,
                "reasoning": "Continuing assigned journey",
                "pace": pace,
            }

        if verb == "wait":
            return {
                "action_type": "wait",
                "target": current_position,
                "target_type": "current_position",
                "confidence": 0.95,
                "reasoning": "Waiting at current position",
                "wait_reason": decision.get("wait_reason"),
                "pace": pace,
            }

        if verb == "seek_information":
            return {
                "action_type": "seek_information",
                "target": current_position,
                "target_type": "information_source",
                "confidence": 0.9,
                "reasoning": "Seeking information from deterministic resolver",
                "pace": pace,
            }

        if verb == "evacuate":
            exit_id = decision.get("exit_id")
            exit_coords = self._get_exit_coordinates(exit_id, agent_level) if exit_id else None
            if exit_coords:
                return {
                    "action_type": "move",
                    "target": exit_coords,
                    "target_type": "exit",
                    "exit_name": exit_id,
                    "resolved_exit_id": exit_id,
                    "confidence": 0.95,
                    "reasoning": f"Evacuating via {exit_id}",
                    "pace": pace,
                }

        if verb == "leave_by_train":
            nearest_train = self._find_nearest_train_exit(current_position, agent_level)
            if nearest_train is not None:
                return {
                    "action_type": "move",
                    "target": nearest_train["coords"],
                    "target_type": "exit",
                    "exit_name": nearest_train["name"],
                    "resolved_exit_id": nearest_train["name"],
                    "confidence": 0.9,
                    "reasoning": "Heading to nearest train platform for boarding",
                    "pace": pace,
                }

        # The chosen exit or train could not be located: wait.
        return {
            "action_type": "wait",
            "target": current_position,
            "target_type": "current_position",
            "confidence": 0.4,
            "reasoning": "Invalid v1.1 action payload; defaulting to wait",
            "wait_reason": "awaiting_information",
            "pace": None,
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
            if m is None:
                m2 = _ESC_ZONE_DOTTED_RE.match(resolved_id)
                if m2:
                    m = re.match(r"^([a-f])$", m2.group(1).lower())
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

    def _find_nearest_train_exit(
        self,
        position: tuple[float, float],
        agent_level: str | None = None,
    ) -> dict[str, Any] | None:
        """Return nearest train-platform exit currently defined on the agent's level."""
        candidates: dict[str, tuple[float, float]] = {}

        if agent_level and self.jps_sim and hasattr(self.jps_sim, "simulations"):
            level_sim = self.jps_sim.simulations.get(agent_level)
            if level_sim and hasattr(level_sim, "exit_manager"):
                for name, coords in level_sim.exit_manager.exit_coordinates.items():
                    if is_train_exit(name):
                        candidates[name] = coords

        if not candidates:
            for name, coords in self.exits.items():
                if is_train_exit(name):
                    candidates[name] = coords

        if not candidates:
            return None

        nearest_name = min(
            candidates,
            key=lambda name: math.hypot(
                position[0] - candidates[name][0],
                position[1] - candidates[name][1],
            ),
        )
        return {"name": nearest_name, "coords": candidates[nearest_name]}
