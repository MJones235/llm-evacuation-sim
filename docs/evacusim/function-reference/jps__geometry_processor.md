# Function Reference: evacusim/jps/geometry_processor.py

- Source file: [evacusim/jps/geometry_processor.py](../../../evacusim/jps/geometry_processor.py)
- Top-level classes: 1
- Top-level functions: 0

## Module Summary

Geometry processing utilities for JuPedSim simulations.

## Classes and Methods

### Class GeometryProcessor

- Location: [evacusim/jps/geometry_processor.py#L12](../../../evacusim/jps/geometry_processor.py#L12)
- Summary: Processes geometry for JuPedSim simulations, handling obstacles and topology.
- Method count: 3

#### fix_topology

- Signature: def fix_topology(polygon) -> Polygon
- Location: [evacusim/jps/geometry_processor.py#L16](../../../evacusim/jps/geometry_processor.py#L16)
- Summary: Fix invalid polygon topology using make_valid() or buffer(0).
- Key calls: hasattr, shapely.make_valid, polygon.make_valid, polygon.buffer

#### integrate_obstacles

- Signature: def integrate_obstacles(zones, obstacles) -> tuple[dict[str, Polygon], list[Polygon]]
- Location: [evacusim/jps/geometry_processor.py#L39](../../../evacusim/jps/geometry_processor.py#L39)
- Summary: Integrate obstacles into walkable zones by subtracting them.
- Key calls: zones.items, GeometryProcessor.fix_topology, fixed_obstacles.append, zone_with_holes.intersects, isinstance, max, print, zone_with_holes.difference

#### combine_geometry

- Signature: def combine_geometry(polygons) -> Polygon
- Location: [evacusim/jps/geometry_processor.py#L109](../../../evacusim/jps/geometry_processor.py#L109)
- Summary: Combine multiple polygons into a single geometry for JuPedSim.
- Key calls: enumerate, unary_union, isinstance, ValueError, GeometryProcessor.fix_topology, valid_polygons.append, len, max, print

## Coverage Notes

- Nested helper functions documented in this file: 0
