# Function Reference: evacusim/translation/action_translator.py

- Source file: [evacusim/translation/action_translator.py](../../../evacusim/translation/action_translator.py)
- Top-level classes: 1
- Top-level functions: 0

## Module Summary

Translates natural language actions from Concordia agents into JuPedSim waypoints and goals.

## Classes and Methods

### Class ActionTranslator

- Location: [evacusim/translation/action_translator.py#L30](../../../evacusim/translation/action_translator.py#L30)
- Summary: Translates natural language actions from Concordia agents into JuPedSim waypoints and goals.
- Method count: 6

#### __init__

- Signature: def __init__(self, station_layout, model=None, jps_sim=None)
- Location: [evacusim/translation/action_translator.py#L36](../../../evacusim/translation/action_translator.py#L36)
- Summary: Initialize the action translator.
- Key calls: station_layout.get, build_registry_from_station_layout

#### translate

- Signature: def translate(self, agent_id, action, current_position) -> dict[str, Any]
- Location: [evacusim/translation/action_translator.py#L74](../../../evacusim/translation/action_translator.py#L74)
- Summary: Translate a JSON action response to a concrete goal.
- Key calls: hasattr, self.jps_sim.agent_levels.get, action.find, json.loads, data.get, self._find_zone_target, logger.warning, strip, self._get_exit_coordinates, self.exit_registry.resolve_to_id

#### _redirect_to_next_level_exit

- Signature: def _redirect_to_next_level_exit(self, agent_id, resolved_exit_id, agent_level, current_position) -> dict[str, Any] | None
- Location: [evacusim/translation/action_translator.py#L263](../../../evacusim/translation/action_translator.py#L263)
- Summary: When an agent requests an exit that doesn't exist on their current level, find the nearest exit on the current level that leads toward it.
- Key calls: self.jps_sim.simulations.get, resolved_exit_id.endswith, min, re.match, logger.info, resolved_exit_id.startswith, self.exit_registry.get_display_name, hasattr, level_exits.items, math.hypot

#### _get_exit_coordinates

- Signature: def _get_exit_coordinates(self, exit_name, agent_level=None) -> tuple[float, float] | None
- Location: [evacusim/translation/action_translator.py#L357](../../../evacusim/translation/action_translator.py#L357)
- Summary: Get exit coordinates, resolving natural language names to technical IDs.
- Key calls: self.exit_registry.resolve_to_id, logger.debug, hasattr, self.jps_sim.simulations.get, _re.match, m.group, level_exits.items, gm.blocked_exit_positions.get, list, self.exit_registry.get_all_ids

#### _find_nearest_exit

- Signature: def _find_nearest_exit(self, position, agent_level=None) -> dict[str, Any]
- Location: [evacusim/translation/action_translator.py#L444](../../../evacusim/translation/action_translator.py#L444)
- Summary: Find the nearest exit to a given position (level-aware).
- Key calls: dict, float, available_exits.items, hasattr, self.jps_sim.simulations.get, available_exits.update

#### _find_zone_target

- Signature: def _find_zone_target(self, text) -> tuple[str, tuple[float, float]] | None
- Location: [evacusim/translation/action_translator.py#L471](../../../evacusim/translation/action_translator.py#L471)
- Summary: Find zone coordinates from zone name in text.
- Key calls: self.zones_polygons.keys, self.zones.items, zone_name.lower, polygon.representative_point

## Coverage Notes

- Nested helper functions documented in this file: 0
