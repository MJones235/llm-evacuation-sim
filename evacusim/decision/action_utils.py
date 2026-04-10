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

    # Match coordinates to exit name
    for exit_name, exit_coords in station_layout["exits"].items():
        # Check if coordinates match (within 1m tolerance)
        if (
            abs(target_coords[0] - exit_coords[0]) < 1.0
            and abs(target_coords[1] - exit_coords[1]) < 1.0
        ):
            return exit_name

    # Also check down-access escalator zones (level-0 → level -1 transfers).
    # These are stored in station_layout["down_access_exits"] keyed by their zone
    # name (e.g. "L0_esc_a_down"), but the JPS evacuation exit is registered as
    # "escalator_a_down" — convert the key before returning.
    import re as _re

    for zone_key, exit_coords in station_layout.get("down_access_exits", {}).items():
        if (
            abs(target_coords[0] - exit_coords[0]) < 1.0
            and abs(target_coords[1] - exit_coords[1]) < 1.0
        ):
            # Convert "L0_esc_a_down" → "escalator_a_down"
            m = _re.match(r"^L[^_]+_esc_([a-f])_(up|down)$", zone_key)
            if m:
                return f"escalator_{m.group(1)}_{m.group(2)}"
            return zone_key

    return None
