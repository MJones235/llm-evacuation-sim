# Function Reference: evacusim/jps/exit_manager.py

- Source file: [evacusim/jps/exit_manager.py](../../../evacusim/jps/exit_manager.py)
- Top-level classes: 1
- Top-level functions: 0

## Module Summary

Evacuation exit management for JuPedSim simulation.

## Classes and Methods

### Class ExitManager

- Location: [evacusim/jps/exit_manager.py#L18](../../../evacusim/jps/exit_manager.py#L18)
- Summary: Manages evacuation exits and journey routing.
- Method count: 9

#### __init__

- Signature: def __init__(self, stage_manager, entrance_areas, walkable_areas_with_obstacles, walkable_areas=None, level_id='0', exit_thresholds=None, train_entrance_areas=None, initially_blocked_exits=None)
- Location: [evacusim/jps/exit_manager.py#L29](../../../evacusim/jps/exit_manager.py#L29)
- Summary: Initialize exit manager and create evacuation exits.
- Key calls: str, set, logger.info, GeometryProcessor.combine_geometry, self._setup_evacuation_exits, self._populate_exit_coordinates, self._populate_train_exit_coordinates, list, p.split, keys

#### _setup_evacuation_exits

- Signature: def _setup_evacuation_exits(self, walkable_geometry) -> tuple[dict[str, int], dict[str, int]]
- Location: [evacusim/jps/exit_manager.py#L97](../../../evacusim/jps/exit_manager.py#L97)
- Summary: Create evacuation exit stages at entrance locations or escalators.
- Key calls: self.entrance_areas.items, re.compile, self.walkable_areas.items, logger.info, self._create_escalator_exits, self._create_train_exits, escalator_exits.update, escalator_journeys.update, self._create_convex_exit_from_polygon, self.stage_manager.create_simple_exit_journey

#### _create_escalator_exits

- Signature: def _create_escalator_exits(self, walkable_geometry) -> tuple[dict[str, int], dict[str, int]]
- Location: [evacusim/jps/exit_manager.py#L212](../../../evacusim/jps/exit_manager.py#L212)
- Summary: Create real terminal exits at escalator locations for levels without street exits.
- Key calls: re.compile, self.walkable_areas.items, escalator_pattern.match, match.groups, logger.debug, logger.info, self.stage_manager.create_exit_at_coordinates, self.stage_manager.create_simple_exit_journey, list, len

#### _create_train_exits

- Signature: def _create_train_exits(self) -> tuple[dict[str, int], dict[str, int]]
- Location: [evacusim/jps/exit_manager.py#L299](../../../evacusim/jps/exit_manager.py#L299)
- Summary: Create JuPedSim exit stages for train boarding areas.
- Key calls: self.train_entrance_areas.items, list, len, logger.warning, self.stage_manager.create_exit_at_coordinates, self.stage_manager.create_simple_exit_journey, self.train_exits.add, logger.info

#### _populate_exit_coordinates

- Signature: def _populate_exit_coordinates(self)
- Location: [evacusim/jps/exit_manager.py#L340](../../../evacusim/jps/exit_manager.py#L340)
- Summary: Populate exit_coordinates dict with exit center positions.
- Key calls: self.entrance_areas.items, re.compile, self.walkable_areas.items, logger.info, escalator_pattern.match, match.groups, logger.debug, len, list, self.exit_coordinates.keys

#### _populate_train_exit_coordinates

- Signature: def _populate_train_exit_coordinates(self) -> None
- Location: [evacusim/jps/exit_manager.py#L402](../../../evacusim/jps/exit_manager.py#L402)
- Summary: Add centroid coordinates for train boarding exits to exit_coordinates.
- Key calls: self.train_entrance_areas.items, logger.debug

#### _create_convex_exit_from_polygon

- Signature: def _create_convex_exit_from_polygon(self, exit_name, polygon, walkable_geometry) -> int | None
- Location: [evacusim/jps/exit_manager.py#L413](../../../evacusim/jps/exit_manager.py#L413)
- Summary: Create a convex rectangular exit centered on polygon centroid.
- Key calls: self.stage_manager.create_exit_at_coordinates, logger.info, walkable_geometry.contains, polygon.intersection, intersection.representative_point, logger.error

#### get_default_exit

- Signature: def get_default_exit(self) -> tuple[str, int, int]
- Location: [evacusim/jps/exit_manager.py#L458](../../../evacusim/jps/exit_manager.py#L458)
- Summary: Get default exit for agents.
- Key calls: logger.error, RuntimeError, list, self.evacuation_exits.keys

#### get_exit_ids

- Signature: def get_exit_ids(self, exit_name) -> tuple[int, int]
- Location: [evacusim/jps/exit_manager.py#L482](../../../evacusim/jps/exit_manager.py#L482)
- Summary: Get stage and journey IDs for a specific exit.
- Key calls: KeyError, list, self.evacuation_journeys.keys

## Coverage Notes

- Nested helper functions documented in this file: 0
