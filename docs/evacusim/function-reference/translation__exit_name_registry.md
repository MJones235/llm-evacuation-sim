# Function Reference: evacusim/translation/exit_name_registry.py

- Source file: [evacusim/translation/exit_name_registry.py](../../../evacusim/translation/exit_name_registry.py)
- Top-level classes: 1
- Top-level functions: 1

## Module Summary

Exit Name Registry - Bidirectional mapping between technical IDs and display names.

## Top-Level Functions

### build_registry_from_station_layout

- Signature: def build_registry_from_station_layout(station_layout, jps_sim=None) -> ExitNameRegistry
- Location: [evacusim/translation/exit_name_registry.py#L272](../../../evacusim/translation/exit_name_registry.py#L272)
- Summary: Build exit name registry from station layout and simulation.
- Key calls: ExitNameRegistry, station_layout.get, keys, re.compile, zone_known.values, logger.info, registry.register_exit, _esc_zone_re.match, hasattr, jps_sim.simulations.items

## Classes and Methods

### Class ExitNameRegistry

- Location: [evacusim/translation/exit_name_registry.py#L20](../../../evacusim/translation/exit_name_registry.py#L20)
- Summary: Central registry for exit name mappings.
- Method count: 13

#### __init__

- Signature: def __init__(self)
- Location: [evacusim/translation/exit_name_registry.py#L31](../../../evacusim/translation/exit_name_registry.py#L31)
- Summary: Initialize the registry with mappings.

#### register_exit

- Signature: def register_exit(self, exit_id, display_name=None)
- Location: [evacusim/translation/exit_name_registry.py#L41](../../../evacusim/translation/exit_name_registry.py#L41)
- Summary: Register an exit with its display name.
- Key calls: self._normalize, self._resolve_cache.clear, self._generate_display_name

#### get_display_name

- Signature: def get_display_name(self, exit_id) -> str
- Location: [evacusim/translation/exit_name_registry.py#L65](../../../evacusim/translation/exit_name_registry.py#L65)
- Summary: Get display name for a technical ID.
- Key calls: self._id_to_display.get

#### resolve_to_id

- Signature: def resolve_to_id(self, user_input) -> str | None
- Location: [evacusim/translation/exit_name_registry.py#L77](../../../evacusim/translation/exit_name_registry.py#L77)
- Summary: Resolve user input (potentially natural language) to technical ID.
- Key calls: self._resolve_cache.get, self._resolve_to_id_uncached

#### _resolve_to_id_uncached

- Signature: def _resolve_to_id_uncached(self, user_input) -> str | None
- Location: [evacusim/translation/exit_name_registry.py#L95](../../../evacusim/translation/exit_name_registry.py#L95)
- Summary: Internal resolver (no caching layer).
- Key calls: self._normalize, self._fuzzy_match_street, self._fuzzy_match_escalator

#### _normalize

- Signature: def _normalize(self, name) -> str
- Location: [evacusim/translation/exit_name_registry.py#L127](../../../evacusim/translation/exit_name_registry.py#L127)
- Summary: Normalize a name for matching.
- Key calls: name.lower, re.sub, name.strip

#### _generate_display_name

- Signature: def _generate_display_name(self, exit_id) -> str
- Location: [evacusim/translation/exit_name_registry.py#L153](../../../evacusim/translation/exit_name_registry.py#L153)
- Summary: Auto-generate display name from technical ID.
- Key calls: re.match, split, join, escalator_match.groups, exit_id.replace, word.capitalize, letter.upper

#### _fuzzy_match_escalator

- Signature: def _fuzzy_match_escalator(self, normalized_input) -> str | None
- Location: [evacusim/translation/exit_name_registry.py#L180](../../../evacusim/translation/exit_name_registry.py#L180)
- Summary: Fuzzy match escalator references.
- Key calls: re.search, letter_match.group, any, exit_id.startswith

#### _fuzzy_match_street

- Signature: def _fuzzy_match_street(self, normalized_input) -> str | None
- Location: [evacusim/translation/exit_name_registry.py#L225](../../../evacusim/translation/exit_name_registry.py#L225)
- Summary: Fuzzy match street names.
- Key calls: set, self._normalized_to_id.items, normalized_input.split, len, normalized_name.split

#### get_all_display_names

- Signature: def get_all_display_names(self) -> list[str]
- Location: [evacusim/translation/exit_name_registry.py#L255](../../../evacusim/translation/exit_name_registry.py#L255)
- Summary: Get list of all registered display names.
- Key calls: list, self._id_to_display.values

#### get_all_ids

- Signature: def get_all_ids(self) -> list[str]
- Location: [evacusim/translation/exit_name_registry.py#L259](../../../evacusim/translation/exit_name_registry.py#L259)
- Summary: Get list of all registered technical IDs.
- Key calls: list, self._id_to_display.keys

#### __len__

- Signature: def __len__(self) -> int
- Location: [evacusim/translation/exit_name_registry.py#L263](../../../evacusim/translation/exit_name_registry.py#L263)
- Summary: Get number of registered exits.
- Key calls: len

#### __repr__

- Signature: def __repr__(self) -> str
- Location: [evacusim/translation/exit_name_registry.py#L267](../../../evacusim/translation/exit_name_registry.py#L267)
- Summary: String representation.
- Key calls: len

## Coverage Notes

- Nested helper functions documented in this file: 0
