"""
Exit Name Registry - Bidirectional mapping between technical IDs and display names.

Provides clean separation between:
- Technical IDs (used in geometry/routing): "blackett_street", "escalator_b_up"
- Display names (shown to LLMs): "Blackett Street", "Escalator B (going up)"

This allows LLMs to work in natural language while the system internally
uses technical identifiers.
"""

import re
from typing import Any

from evacusim.utils.logger import get_logger

logger = get_logger(__name__)


class ExitNameRegistry:
    """
    Central registry for exit name mappings.

    Handles bidirectional translation between:
    - Technical IDs (geometry/code): "blackett_street", "escalator_b_up"
    - Display names (LLM-facing): "Blackett Street", "Escalator B (going up)"

    Provides fuzzy matching to handle natural language variations.
    """

    def __init__(self):
        """Initialize the registry with mappings."""
        # Map technical_id -> display_name
        self._id_to_display: dict[str, str] = {}
        # Map normalized_name -> technical_id (for reverse lookup)
        self._normalized_to_id: dict[str, str] = {}
        # Resolution cache: raw user input -> resolved technical ID (or None)
        # This avoids repeated regex/string work for identical LLM outputs.
        self._resolve_cache: dict[str, str | None] = {}

    def register_exit(self, exit_id: str, display_name: str | None = None):
        """
        Register an exit with its display name.

        Args:
            exit_id: Technical identifier (e.g., "blackett_street", "escalator_b_up")
            display_name: Human-readable name (e.g., "Blackett Street")
                         If None, auto-generates from exit_id
        """
        if display_name is None:
            display_name = self._generate_display_name(exit_id)

        self._id_to_display[exit_id] = display_name

        # Store normalized version for fuzzy lookup
        normalized = self._normalize(display_name)
        self._normalized_to_id[normalized] = exit_id

        # Also store ID normalized (so "blackett_street" matches itself)
        self._normalized_to_id[self._normalize(exit_id)] = exit_id

        # Invalidate resolution cache — new registrations change reachable IDs
        self._resolve_cache.clear()

    def get_display_name(self, exit_id: str) -> str:
        """
        Get display name for a technical ID.

        Args:
            exit_id: Technical identifier

        Returns:
            Human-readable display name
        """
        return self._id_to_display.get(exit_id, exit_id)

    def resolve_to_id(self, user_input: str) -> str | None:
        """
        Resolve user input (potentially natural language) to technical ID.

        Results are cached per-instance so repeated identical LLM outputs
        (which are the common case) pay only a single dict lookup.
        """
        cached = self._resolve_cache.get(user_input)
        if cached is not None:
            return cached
        # Sentinel: distinguish "not cached" from "cached as None"
        if user_input in self._resolve_cache:
            return None

        result = self._resolve_to_id_uncached(user_input)
        self._resolve_cache[user_input] = result
        return result

    def _resolve_to_id_uncached(self, user_input: str) -> str | None:
        """
        Internal resolver (no caching layer).

        Handles variations like:
        - "Blackett Street" -> "blackett_street"
        - "blackett street" -> "blackett_street"
        - "Escalator B" -> "escalator_b_up"
        - "escalator b going up" -> "escalator_b_up"

        Args:
            user_input: LLM output or user-provided exit name

        Returns:
            Technical ID if match found, None otherwise
        """
        # First try exact match on ID
        if user_input in self._id_to_display:
            return user_input

        # Try normalized lookup
        normalized = self._normalize(user_input)
        if normalized in self._normalized_to_id:
            return self._normalized_to_id[normalized]

        # Try fuzzy partial matches for escalators
        if "escalator" in normalized:
            return self._fuzzy_match_escalator(normalized)

        # Try partial street name match
        return self._fuzzy_match_street(normalized)

    def _normalize(self, name: str) -> str:
        """
        Normalize a name for matching.

        - Lowercase
        - Remove punctuation
        - Collapse spaces/underscores
        - Trim special suffixes

        Args:
            name: Input name

        Returns:
            Normalized string
        """
        # Lowercase
        name = name.lower()
        # Remove common suffixes
        name = re.sub(r"\s+(exit|entrance|street|station)$", "", name)
        # Remove punctuation
        name = re.sub(r"[^\w\s]", "", name)
        # Convert underscores and multiple spaces to single space
        name = re.sub(r"[_\s]+", " ", name)
        # Trim
        return name.strip()

    def _generate_display_name(self, exit_id: str) -> str:
        """
        Auto-generate display name from technical ID.

        Examples:
        - "blackett_street" -> "Blackett Street"
        - "grey_street" -> "Grey Street"
        - "escalator_b_up" -> "Escalator B (going up)"
        - "escalator_c_down" -> "Escalator C (going down)"

        Args:
            exit_id: Technical identifier

        Returns:
            Human-readable display name
        """
        # Handle escalators
        escalator_match = re.match(r"escalator_([a-z])_(up|down)", exit_id)
        if escalator_match:
            letter, direction = escalator_match.groups()
            direction_display = "going up" if direction == "up" else "going down"
            return f"Escalator {letter.upper()} ({direction_display})"

        # Handle street names (replace underscores, title case)
        words = exit_id.replace("_", " ").split()
        return " ".join(word.capitalize() for word in words)

    def _fuzzy_match_escalator(self, normalized_input: str) -> str | None:
        """
        Fuzzy match escalator references.

        Handles:
        - "escalator b" -> "escalator_b_up" or "escalator_b_down"
        - "escalator b going up" -> "escalator_b_up"
        - "b escalator" -> "escalator_b_up"

        Args:
            normalized_input: Normalized user input

        Returns:
            Matched exit ID or None
        """
        # Extract letter (a-f typically)
        letter_match = re.search(r"\b([a-f])\b", normalized_input)
        if not letter_match:
            return None

        letter = letter_match.group(1)

        # Check for direction hints
        has_up = any(word in normalized_input for word in ["up", "going up", "upward", "upwards"])
        has_down = any(
            word in normalized_input for word in ["down", "going down", "downward", "downwards"]
        )

        # Try to match with direction
        if has_up:
            candidate = f"escalator_{letter}_up"
            if candidate in self._id_to_display:
                return candidate
        elif has_down:
            candidate = f"escalator_{letter}_down"
            if candidate in self._id_to_display:
                return candidate

        # If no direction or both, try to find any escalator with this letter
        for exit_id in self._id_to_display:
            if exit_id.startswith(f"escalator_{letter}_"):
                return exit_id

        return None

    def _fuzzy_match_street(self, normalized_input: str) -> str | None:
        """
        Fuzzy match street names.

        Handles partial matches:
        - "blackett" -> "blackett_street"
        - "grey" -> "grey_street"

        Args:
            normalized_input: Normalized user input

        Returns:
            Matched exit ID or None
        """
        # Try partial word match
        input_words = set(normalized_input.split())

        best_match = None
        best_score = 0

        for normalized_name, exit_id in self._normalized_to_id.items():
            name_words = set(normalized_name.split())
            # Count matching words
            matching = len(input_words & name_words)
            if matching > best_score:
                best_score = matching
                best_match = exit_id

        return best_match if best_score > 0 else None

    def get_all_display_names(self) -> list[str]:
        """Get list of all registered display names."""
        return list(self._id_to_display.values())

    def get_all_ids(self) -> list[str]:
        """Get list of all registered technical IDs."""
        return list(self._id_to_display.keys())

    def __len__(self) -> int:
        """Get number of registered exits."""
        return len(self._id_to_display)

    def __repr__(self) -> str:
        """String representation."""
        return f"<ExitNameRegistry: {len(self)} exits registered>"


def build_registry_from_station_layout(
    station_layout: dict[str, Any], jps_sim=None
) -> ExitNameRegistry:
    """
    Build exit name registry from station layout and simulation.

    Args:
        station_layout: Station geometry dictionary
        jps_sim: JuPedSim simulation (for multi-level exits)

    Returns:
        Populated ExitNameRegistry
    """
    registry = ExitNameRegistry()
    custom_names: dict[str, str] = station_layout.get("custom_exit_display_names", {})

    # Register all exits from station_layout
    for exit_id in station_layout.get("exits", {}).keys():
        registry.register_exit(exit_id, custom_names.get(exit_id))

    # Register down-access exits (concourse escalators leading to platforms).
    # Keys are geometry zone names like "L0_esc_a_down"; normalise them to the
    # canonical exit ID form ("escalator_a_down") so that display names like
    # "Escalator A (down to Platform 3 & Platform 4)" resolve to the exit name
    # that JuPedSim actually has registered, preventing routing failures.
    _esc_zone_re = re.compile(r"^L[^_]+_esc_([a-f])_(up|down)$")
    for zone_key in station_layout.get("down_access_exits", {}).keys():
        m = _esc_zone_re.match(zone_key)
        canonical_id = f"escalator_{m.group(1)}_{m.group(2)}" if m else zone_key
        if canonical_id not in registry._id_to_display:
            # Prefer the custom display name keyed by zone_key, then by canonical_id.
            display = custom_names.get(zone_key) or custom_names.get(canonical_id)
            registry.register_exit(canonical_id, display)

    # Register multi-level exits (escalators)
    if jps_sim and hasattr(jps_sim, "simulations"):
        for _, level_sim in jps_sim.simulations.items():
            if hasattr(level_sim, "exit_manager"):
                for exit_id in level_sim.exit_manager.evacuation_exits.keys():
                    if exit_id not in registry._id_to_display:
                        registry.register_exit(exit_id, custom_names.get(exit_id))

    logger.info(f"Built exit name registry with {len(registry)} exits")
    return registry
