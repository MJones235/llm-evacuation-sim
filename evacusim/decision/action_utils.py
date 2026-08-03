"""
Action utilities for parsing and extracting information from agent actions.

This module provides utilities for working with translated actions and
extracting specific information like exit names.
"""

from typing import Any


def extract_exit_name(
    translated_action: dict[str, Any], station_layout: dict[str, Any]
) -> str | None:
    """
    Extract the target exit name from a translated action.

    Args:
        translated_action: Translated action dictionary with action_type and target
        station_layout: Station layout dict with "exits" mapping exit names to coords

    Returns:
        Exit name if action is moving to an exit, None otherwise
    """
    if translated_action["action_type"] != "move":
        return None

    target_coords = translated_action.get("target")
    if not target_coords:
        return None

    # When the translator already resolved an exit (e.g. escalator exits whose
    # coordinates are NOT in station_layout["exits"]), trust its resolved ID
    # directly.  This prevents escalator exits from falling through to the
    # waypoint path in _handle_move_action, which caused agents to congregate
    # at the escalator centroid and never exit the level.
    if translated_action.get("target_type") == "exit":
        resolved = translated_action.get("resolved_exit_id")
        if resolved:
            return resolved

    # Match coordinates to exit name
    for exit_name, exit_coords in station_layout["exits"].items():
        # Check if coordinates match (within 1m tolerance)
        if (
            abs(target_coords[0] - exit_coords[0]) < 1.0
            and abs(target_coords[1] - exit_coords[1]) < 1.0
        ):
            return exit_name

    # Also check down-access escalator zones (concourse -> platform transfers).
    # These are stored in station_layout["down_access_exits"] keyed by transfer
    # zone names and must be normalized to JuPedSim exit IDs.
    import re as _re

    for zone_key, exit_coords in station_layout.get("down_access_exits", {}).items():
        if (
            abs(target_coords[0] - exit_coords[0]) < 1.0
            and abs(target_coords[1] - exit_coords[1]) < 1.0
        ):
            # Convert legacy and current zone keys to canonical exit IDs.
            m = _re.match(r"^L[^_]+_esc_([a-f])_(up|down)$", zone_key)
            if m:
                return f"escalator_{m.group(1)}_{m.group(2)}"
            m2 = _re.match(
                r"^esc\.([A-F])\.zone\.(concourse|platform)\.(departure|arrival)$",
                zone_key,
            )
            if m2:
                letter, location, role = m2.groups()
                direction = "down" if (location == "concourse" and role == "departure") else "up"
                return f"escalator_{letter.lower()}_{direction}"
            return zone_key

    return None
