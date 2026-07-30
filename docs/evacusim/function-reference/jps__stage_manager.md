# Function Reference: evacusim/jps/stage_manager.py

- Source file: [evacusim/jps/stage_manager.py](../../../evacusim/jps/stage_manager.py)
- Top-level classes: 1
- Top-level functions: 0

## Module Summary

Stage management for JuPedSim simulations.

## Classes and Methods

### Class StageManager

- Location: [evacusim/jps/stage_manager.py#L24](../../../evacusim/jps/stage_manager.py#L24)
- Summary: Manages stages (exits, waypoints) and journeys in JuPedSim simulations.
- Method count: 10

#### __init__

- Signature: def __init__(self, simulation) -> None
- Location: [evacusim/jps/stage_manager.py#L27](../../../evacusim/jps/stage_manager.py#L27)
- Summary: Initialize stage manager.

#### create_exit_at_zone_centroid

- Signature: def create_exit_at_zone_centroid(self, zone_name, zone_polygon, width=30.0, height=30.0) -> int
- Location: [evacusim/jps/stage_manager.py#L39](../../../evacusim/jps/stage_manager.py#L39)
- Summary: Create an exit stage at the centroid of a zone.
- Key calls: self.simulation.add_exit_stage, logger.debug

#### create_exit_at_coordinates

- Signature: def create_exit_at_coordinates(self, exit_name, coords) -> int
- Location: [evacusim/jps/stage_manager.py#L76](../../../evacusim/jps/stage_manager.py#L76)
- Summary: Create an exit stage at specific coordinates.
- Key calls: self.simulation.add_exit_stage, logger.debug

#### create_waypoint

- Signature: def create_waypoint(self, waypoint_name, coords, distance=1.0) -> int
- Location: [evacusim/jps/stage_manager.py#L94](../../../evacusim/jps/stage_manager.py#L94)
- Summary: Create a waypoint stage.
- Key calls: self.simulation.add_waypoint_stage, logger.debug

#### create_waiting_stage

- Signature: def create_waiting_stage(self, name, position, distance=5.0) -> int
- Location: [evacusim/jps/stage_manager.py#L115](../../../evacusim/jps/stage_manager.py#L115)
- Summary: Create a waiting area (waypoint) at a specific position.
- Key calls: self.simulation.add_waypoint_stage

#### create_journey

- Signature: def create_journey(self, journey_name, stage_ids) -> int
- Location: [evacusim/jps/stage_manager.py#L136](../../../evacusim/jps/stage_manager.py#L136)
- Summary: Create a journey through a sequence of stages.
- Key calls: jps.JourneyDescription, self.simulation.add_journey

#### create_simple_exit_journey

- Signature: def create_simple_exit_journey(self, journey_name, exit_id) -> int
- Location: [evacusim/jps/stage_manager.py#L153](../../../evacusim/jps/stage_manager.py#L153)
- Summary: Create a simple journey that goes directly to an exit.
- Key calls: self.create_journey

#### get_exit_id

- Signature: def get_exit_id(self, exit_name) -> int | None
- Location: [evacusim/jps/stage_manager.py#L166](../../../evacusim/jps/stage_manager.py#L166)
- Summary: Get stage ID for a named exit.
- Key calls: self.exits.get

#### get_waypoint_id

- Signature: def get_waypoint_id(self, waypoint_name) -> int | None
- Location: [evacusim/jps/stage_manager.py#L178](../../../evacusim/jps/stage_manager.py#L178)
- Summary: Get stage ID for a named waypoint.
- Key calls: self.waypoints.get

#### get_journey_id

- Signature: def get_journey_id(self, journey_name) -> int | None
- Location: [evacusim/jps/stage_manager.py#L190](../../../evacusim/jps/stage_manager.py#L190)
- Summary: Get journey ID for a named journey.
- Key calls: self.journeys.get

## Coverage Notes

- Nested helper functions documented in this file: 0
