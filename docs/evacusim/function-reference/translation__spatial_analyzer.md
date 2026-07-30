# Function Reference: evacusim/translation/spatial_analyzer.py

- Source file: [evacusim/translation/spatial_analyzer.py](../../../evacusim/translation/spatial_analyzer.py)
- Top-level classes: 1
- Top-level functions: 0

## Module Summary

Spatial analysis for observation generation.

## Classes and Methods

### Class SpatialAnalyzer

- Location: [evacusim/translation/spatial_analyzer.py#L15](../../../evacusim/translation/spatial_analyzer.py#L15)
- Summary: Analyzes spatial relationships in the station environment.
- Method count: 12

#### __init__

- Signature: def __init__(self, station_layout, exit_registry=None)
- Location: [evacusim/translation/spatial_analyzer.py#L26](../../../evacusim/translation/spatial_analyzer.py#L26)
- Summary: Initialize spatial analyzer.
- Key calls: station_layout.get

#### _is_platform_level

- Signature: def _is_platform_level(level_id) -> bool
- Location: [evacusim/translation/spatial_analyzer.py#L51](../../../evacusim/translation/spatial_analyzer.py#L51)
- Summary: No docstring summary available.
- Key calls: bool, startswith, str

#### _infer_platform_side

- Signature: def _infer_platform_side(self, position, agent_level, agent_zone) -> str | None
- Location: [evacusim/translation/spatial_analyzer.py#L54](../../../evacusim/translation/spatial_analyzer.py#L54)
- Summary: Infer whether the agent is on platform_abc or platform_def side.
- Key calls: lower, self.zone_boundaries.get, str, lvl_bounds.get, cfg.get, isinstance, get, float

#### _belongs_to_platform_side

- Signature: def _belongs_to_platform_side(canonical_exit_id, platform_side) -> bool
- Location: [evacusim/translation/spatial_analyzer.py#L104](../../../evacusim/translation/spatial_analyzer.py#L104)
- Summary: Return True if a canonical exit logically belongs to the given platform side.
- Key calls: canonical_exit_id.startswith

#### identify_zone

- Signature: def identify_zone(self, position) -> str
- Location: [evacusim/translation/spatial_analyzer.py#L129](../../../evacusim/translation/spatial_analyzer.py#L129)
- Summary: Identify which zone a position is in.
- Key calls: self.zones.items, zone_name.lower, self._point_in_bounds, polygon.covers, polygon.contains, Point, self.zones_polygons.items, self.walkable_areas.items, _is_main_footbridge, _is_platform_zone
- Nested function count: 4

##### Nested helpers inside identify_zone

###### identify_zone._covers_or_contains

- Signature: def _covers_or_contains(polygon, point)
- Location: [evacusim/translation/spatial_analyzer.py#L146](../../../evacusim/translation/spatial_analyzer.py#L146)
- Summary: No nested-function docstring summary available.
- Key calls: polygon.covers, polygon.contains

###### identify_zone._is_main_footbridge

- Signature: def _is_main_footbridge(zone_name) -> bool
- Location: [evacusim/translation/spatial_analyzer.py#L157](../../../evacusim/translation/spatial_analyzer.py#L157)
- Summary: Check if this is the main footbridge zone.
- Key calls: zone_name.lower

###### identify_zone._is_platform_zone

- Signature: def _is_platform_zone(zone_name) -> bool
- Location: [evacusim/translation/spatial_analyzer.py#L162](../../../evacusim/translation/spatial_analyzer.py#L162)
- Summary: Check if this is a platform zone.
- Key calls: zone_name.lower

###### identify_zone._is_connector_zone

- Signature: def _is_connector_zone(zone_name) -> bool
- Location: [evacusim/translation/spatial_analyzer.py#L167](../../../evacusim/translation/spatial_analyzer.py#L167)
- Summary: Check if this is a connector zone between platforms.
- Key calls: zone_name.lower

#### get_nearest_exit_info

- Signature: def get_nearest_exit_info(self, position, agent_level=None, jps_sim=None) -> str
- Location: [evacusim/translation/spatial_analyzer.py#L225](../../../evacusim/translation/spatial_analyzer.py#L225)
- Summary: Get information about the nearest exit on the agent's current level.
- Key calls: float, exits_to_use.items, nearest_name.startswith, hasattr, jps_sim.simulations.get, split, level_sim.exit_manager.evacuation_exits.keys, len, exit_name.startswith, nearest_name.replace

#### get_visible_exits

- Signature: def get_visible_exits(self, position, agent_level=None, agent_zone=None, jps_sim=None, inactive_exits=None, blocked_exits=None) -> list[dict[str, str]]
- Location: [evacusim/translation/spatial_analyzer.py#L305](../../../evacusim/translation/spatial_analyzer.py#L305)
- Summary: Get all exits visible from the agent's position.
- Key calls: self._is_platform_level, exits_to_check.items, sorted, hasattr, jps_sim.simulations.get, self._infer_platform_side, self._has_line_of_sight, visible_by_key.items, visible_exits.append, exits_to_check.update

#### _has_line_of_sight

- Signature: def _has_line_of_sight(self, start, end, obstacles=None, walkable_geom=None) -> bool
- Location: [evacusim/translation/spatial_analyzer.py#L491](../../../evacusim/translation/spatial_analyzer.py#L491)
- Summary: Return True if the segment from start to end is not blocked.
- Key calls: LineString, isinstance, obstacles_to_check.values, sight_line.intersects, walkable_geom.covers, sight_line.crosses, sight_line.within, sight_line.intersection, inter.touches

#### _canonical_visible_exit_key

- Signature: def _canonical_visible_exit_key(self, exit_name) -> str
- Location: [evacusim/translation/spatial_analyzer.py#L547](../../../evacusim/translation/spatial_analyzer.py#L547)
- Summary: Map equivalent exit IDs to one canonical key for display deduplication.
- Key calls: re.match, escalator_match.groups, zone_escalator_match.groups

#### _prefer_exit_label

- Signature: def _prefer_exit_label(self, current_label, candidate_label) -> str
- Location: [evacusim/translation/spatial_analyzer.py#L561](../../../evacusim/translation/spatial_analyzer.py#L561)
- Summary: Prefer the more descriptive label when equivalent exits collide.
- Key calls: current_label.lower, candidate_label.lower, len

#### get_visible_blocked_exits

- Signature: def get_visible_blocked_exits(self, position, blocked_exits, agent_level=None, jps_sim=None) -> list[dict[str, Any]]
- Location: [evacusim/translation/spatial_analyzer.py#L575](../../../evacusim/translation/spatial_analyzer.py#L575)
- Summary: Get blocked exits that an agent can perceive from their current position.
- Key calls: hasattr, jps_sim.simulations.get, exits_to_check.get, visible_blocked.append, str, self.exits.get, self.exit_registry.get_display_name, dict, getattr, self._walkable_union_cache.get

#### _point_in_bounds

- Signature: def _point_in_bounds(self, point, bounds) -> bool
- Location: [evacusim/translation/spatial_analyzer.py#L686](../../../evacusim/translation/spatial_analyzer.py#L686)
- Summary: Check if a point is within rectangular bounds.

## Coverage Notes

- Nested helper functions documented in this file: 4
