# Function Reference: evacusim/jps/geometry_loader.py

- Source file: [evacusim/jps/geometry_loader.py](../../../evacusim/jps/geometry_loader.py)
- Top-level classes: 0
- Top-level functions: 10

## Module Summary

Load and convert SUMO walking areas to JuPedSim geometry.

## Top-Level Functions

### parse_shape_string

- Signature: def parse_shape_string(shape_str) -> list[tuple[float, float]]
- Location: [evacusim/jps/geometry_loader.py#L11](../../../evacusim/jps/geometry_loader.py#L11)
- Summary: Parse SUMO shape string into list of (x, y) coordinate tuples.
- Key calls: split, point_str.split, coords.append, shape_str.strip, float

### load_walkable_areas

- Signature: def load_walkable_areas(xml_path) -> dict[str, Polygon]
- Location: [evacusim/jps/geometry_loader.py#L28](../../../evacusim/jps/geometry_loader.py#L28)
- Summary: Load walkable areas from SUMO walking_areas.
- Key calls: ET.parse, tree.getroot, root.findall, poly.get, parse_shape_string, Polygon

### load_entrance_areas

- Signature: def load_entrance_areas(xml_path) -> dict[str, Polygon]
- Location: [evacusim/jps/geometry_loader.py#L56](../../../evacusim/jps/geometry_loader.py#L56)
- Summary: Load entrance areas from SUMO walking_areas.
- Key calls: ET.parse, tree.getroot, root.findall, poly.get, parse_shape_string, Polygon

### load_platform_areas

- Signature: def load_platform_areas(xml_path) -> dict[str, Polygon]
- Location: [evacusim/jps/geometry_loader.py#L85](../../../evacusim/jps/geometry_loader.py#L85)
- Summary: Load platform areas from SUMO walking_areas.
- Key calls: ET.parse, tree.getroot, root.findall, poly.get, parse_shape_string, Polygon

### load_obstacles

- Signature: def load_obstacles(xml_path) -> list[Polygon]
- Location: [evacusim/jps/geometry_loader.py#L114](../../../evacusim/jps/geometry_loader.py#L114)
- Summary: Load obstacle polygons from SUMO walking_areas.
- Key calls: ET.parse, tree.getroot, root.findall, poly.get, parse_shape_string, Polygon, obstacles.append

### load_train_entrance_areas

- Signature: def load_train_entrance_areas(xml_path) -> dict[str, Polygon]
- Location: [evacusim/jps/geometry_loader.py#L140](../../../evacusim/jps/geometry_loader.py#L140)
- Summary: Load train entrance areas from a level geometry XML file.
- Key calls: ET.parse, tree.getroot, root.findall, poly.get, parse_shape_string, Polygon

### load_train_track_sides

- Signature: def load_train_track_sides(xml_path) -> dict[str, str]
- Location: [evacusim/jps/geometry_loader.py#L172](../../../evacusim/jps/geometry_loader.py#L172)
- Summary: Load the ``track_side`` attribute from each ``jupedsim.
- Key calls: ET.parse, tree.getroot, root.findall, poly.get

### load_exit_thresholds

- Signature: def load_exit_thresholds(xml_path) -> dict[str, Polygon]
- Location: [evacusim/jps/geometry_loader.py#L202](../../../evacusim/jps/geometry_loader.py#L202)
- Summary: Load exit threshold markers from a level geometry file.
- Key calls: ET.parse, tree.getroot, root.findall, poly.get, parse_shape_string, Polygon

### load_escalator_corridors

- Signature: def load_escalator_corridors(xml_path) -> dict[str, Polygon]
- Location: [evacusim/jps/geometry_loader.py#L232](../../../evacusim/jps/geometry_loader.py#L232)
- Summary: Load escalator corridor zones from a SUMO geometry file.
- Key calls: ET.parse, tree.getroot, root.findall, poly.get, parse_shape_string, Polygon

### combine_walkable_geometry

- Signature: def combine_walkable_geometry(walkable_areas, obstacles=None) -> tuple
- Location: [evacusim/jps/geometry_loader.py#L263](../../../evacusim/jps/geometry_loader.py#L263)
- Summary: Combine all walkable areas into a single geometry for JuPedSim.
- Key calls: list, walkable_areas.values, len, shapely.GeometryCollection

## Coverage Notes

- Nested helper functions documented in this file: 0
