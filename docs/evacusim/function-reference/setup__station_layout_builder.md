# Function Reference: evacusim/setup/station_layout_builder.py

- Source file: [evacusim/setup/station_layout_builder.py](../../../evacusim/setup/station_layout_builder.py)
- Top-level classes: 1
- Top-level functions: 0

## Module Summary

Station layout builder for Station Concordia simulations.

## Classes and Methods

### Class StationLayoutBuilder

- Location: [evacusim/setup/station_layout_builder.py#L21](../../../evacusim/setup/station_layout_builder.py#L21)
- Summary: Handles creation of station layout from simulation geometry.
- Method count: 5

#### build_layout

- Signature: def build_layout(jps_sim, config) -> dict[str, Any]
- Location: [evacusim/setup/station_layout_builder.py#L25](../../../evacusim/setup/station_layout_builder.py#L25)
- Summary: Build station layout dictionary from pedestrian simulation geometry.
- Key calls: hasattr, get, logger.info, sorted, jps_sim.simulations.values, StationLayoutBuilder._build_zones, StationLayoutBuilder._build_zone_polygons, platform_down_cfg.items, all_walkable_by_level.get, esc_to_platforms.items

#### _build_zones

- Signature: def _build_zones(jps_sim, platform_areas) -> dict[str, dict[str, float]]
- Location: [evacusim/setup/station_layout_builder.py#L214](../../../evacusim/setup/station_layout_builder.py#L214)
- Summary: Build zone boundary definitions.
- Key calls: StationLayoutBuilder._polygon_bounds, list, platform_areas.items, jps_sim.geometry_manager.walkable_areas.values

#### _build_zone_polygons

- Signature: def _build_zone_polygons(jps_sim, platform_areas) -> dict[str, Any]
- Location: [evacusim/setup/station_layout_builder.py#L238](../../../evacusim/setup/station_layout_builder.py#L238)
- Summary: Build zone polygon definitions.
- Key calls: list, jps_sim.geometry_manager.walkable_areas.values

#### _polygon_bounds

- Signature: def _polygon_bounds(polygon) -> dict[str, float]
- Location: [evacusim/setup/station_layout_builder.py#L257](../../../evacusim/setup/station_layout_builder.py#L257)
- Summary: Extract bounding box from a polygon.

#### _escalator_entrance_position

- Signature: def _escalator_entrance_position(corridor_poly, tz_poly, offset=1.5) -> tuple[float, float]
- Location: [evacusim/setup/station_layout_builder.py#L271](../../../evacusim/setup/station_layout_builder.py#L271)
- Summary: Find the concourse-side entrance of an escalator corridor and return a point *offset* metres outside it, suitable for positioning a fire marshal.
- Key calls: len, range, list, math.hypot

## Coverage Notes

- Nested helper functions documented in this file: 0
