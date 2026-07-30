# Function Reference: evacusim/jps/geometry_manager.py

- Source file: [evacusim/jps/geometry_manager.py](../../../evacusim/jps/geometry_manager.py)
- Top-level classes: 1
- Top-level functions: 0

## Module Summary

Geometry management for JuPedSim simulation.

## Classes and Methods

### Class GeometryManager

- Location: [evacusim/jps/geometry_manager.py#L28](../../../evacusim/jps/geometry_manager.py#L28)
- Summary: Manages station geometry loading and JuPedSim simulation creation.
- Method count: 6

#### __init__

- Signature: def __init__(self, network_path, dt=0.05, level_id=0, initially_blocked_exits=None)
- Location: [evacusim/jps/geometry_manager.py#L39](../../../evacusim/jps/geometry_manager.py#L39)
- Summary: Initialize geometry manager and load station geometry.
- Key calls: set, logger.info, self._load_geometry, self._create_simulation, self._apply_initial_blockages, self.walkable_areas.items, _re_tz.match, len

#### _load_geometry

- Signature: def _load_geometry(self) -> tuple[dict, dict, dict, dict, list]
- Location: [evacusim/jps/geometry_manager.py#L106](../../../evacusim/jps/geometry_manager.py#L106)
- Summary: Load station geometry from SUMO network files.
- Key calls: level_file.exists, load_walkable_areas, load_entrance_areas, load_platform_areas, load_obstacles, load_escalator_corridors, load_exit_thresholds, load_train_entrance_areas, GeometryProcessor.integrate_obstacles, logger.info

#### _create_simulation

- Signature: def _create_simulation(self) -> jps.Simulation
- Location: [evacusim/jps/geometry_manager.py#L172](../../../evacusim/jps/geometry_manager.py#L172)
- Summary: Create JuPedSim simulation with loaded geometry.
- Key calls: list, GeometryProcessor.combine_geometry, jps.Simulation, logger.info, self.walkable_areas_with_obstacles.values, ValueError, jps.CollisionFreeSpeedModel

#### _apply_initial_blockages

- Signature: def _apply_initial_blockages(self) -> None
- Location: [evacusim/jps/geometry_manager.py#L205](../../../evacusim/jps/geometry_manager.py#L205)
- Summary: Remove corridor and transfer-zone polygons for pre-blocked escalators.
- Key calls: str, exit_name.split, next, buffer, list, logger.info, shapes.append, self.walkable_areas.pop, self.walkable_areas_with_obstacles.pop, logger.debug

#### add_obstacle_polygon

- Signature: def add_obstacle_polygon(self, obstacle_poly) -> None
- Location: [evacusim/jps/geometry_manager.py#L301](../../../evacusim/jps/geometry_manager.py#L301)
- Summary: Remove *obstacle_poly* from the walkable geometry and call switch_geometry.
- Key calls: GeometryProcessor.fix_topology, self.simulation.switch_geometry, logger.info, self._combined_geometry.difference, logger.warning

#### get_geometry_data

- Signature: def get_geometry_data(self) -> dict[str, Any]
- Location: [evacusim/jps/geometry_manager.py#L321](../../../evacusim/jps/geometry_manager.py#L321)
- Summary: Get geometry information for visualization or analysis.

## Coverage Notes

- Nested helper functions documented in this file: 0
