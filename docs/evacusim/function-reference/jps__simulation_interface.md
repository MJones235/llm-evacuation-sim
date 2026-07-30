# Function Reference: evacusim/jps/simulation_interface.py

- Source file: [evacusim/jps/simulation_interface.py](../../../evacusim/jps/simulation_interface.py)
- Top-level classes: 1
- Top-level functions: 0

## Module Summary

Pedestrian simulation interface for station_concordia.

## Classes and Methods

### Class PedestrianSimulation

- Location: [evacusim/jps/simulation_interface.py#L15](../../../evacusim/jps/simulation_interface.py#L15)
- Summary: Interface for pedestrian simulation backends.
- Method count: 17

#### step

- Signature: def step(self) -> bool
- Location: [evacusim/jps/simulation_interface.py#L38](../../../evacusim/jps/simulation_interface.py#L38)
- Summary: Advance simulation by one timestep.

#### get_simulation_time

- Signature: def get_simulation_time(self) -> float
- Location: [evacusim/jps/simulation_interface.py#L48](../../../evacusim/jps/simulation_interface.py#L48)
- Summary: Get current simulation time in seconds.

#### add_agent

- Signature: def add_agent(self, agent_id, position, walking_speed=1.34) -> None
- Location: [evacusim/jps/simulation_interface.py#L58](../../../evacusim/jps/simulation_interface.py#L58)
- Summary: Add an agent to the simulation.

#### get_agent_position

- Signature: def get_agent_position(self, agent_id) -> tuple[float, float] | None
- Location: [evacusim/jps/simulation_interface.py#L74](../../../evacusim/jps/simulation_interface.py#L74)
- Summary: Get agent's current position.

#### get_all_agent_positions

- Signature: def get_all_agent_positions(self) -> dict[str, tuple[float, float]]
- Location: [evacusim/jps/simulation_interface.py#L86](../../../evacusim/jps/simulation_interface.py#L86)
- Summary: Get positions of all active agents.

#### get_nearby_agents

- Signature: def get_nearby_agents(self, agent_id, radius) -> list[dict[str, Any]]
- Location: [evacusim/jps/simulation_interface.py#L95](../../../evacusim/jps/simulation_interface.py#L95)
- Summary: Get information about agents within radius of given agent.

#### set_agent_target

- Signature: def set_agent_target(self, agent_id, target) -> None
- Location: [evacusim/jps/simulation_interface.py#L112](../../../evacusim/jps/simulation_interface.py#L112)
- Summary: Set an agent's movement target by creating a waypoint.

#### set_agent_evacuation_exit

- Signature: def set_agent_evacuation_exit(self, agent_id, exit_name) -> None
- Location: [evacusim/jps/simulation_interface.py#L126](../../../evacusim/jps/simulation_interface.py#L126)
- Summary: Direct an agent to a specific evacuation exit.

#### set_agent_speed

- Signature: def set_agent_speed(self, agent_id, speed) -> None
- Location: [evacusim/jps/simulation_interface.py#L140](../../../evacusim/jps/simulation_interface.py#L140)
- Summary: Set an agent's walking speed mid-simulation.

#### walkable_areas

- Signature: def walkable_areas(self) -> dict[str, Any]
- Location: [evacusim/jps/simulation_interface.py#L156](../../../evacusim/jps/simulation_interface.py#L156)
- Summary: Get walkable area polygons.

#### walkable_areas_with_obstacles

- Signature: def walkable_areas_with_obstacles(self) -> dict[str, Any]
- Location: [evacusim/jps/simulation_interface.py#L166](../../../evacusim/jps/simulation_interface.py#L166)
- Summary: Get walkable area polygons with obstacles integrated.

#### entrance_areas

- Signature: def entrance_areas(self) -> dict[str, Any]
- Location: [evacusim/jps/simulation_interface.py#L176](../../../evacusim/jps/simulation_interface.py#L176)
- Summary: Get entrance/exit area polygons.

#### platform_areas

- Signature: def platform_areas(self) -> dict[str, Any]
- Location: [evacusim/jps/simulation_interface.py#L186](../../../evacusim/jps/simulation_interface.py#L186)
- Summary: Get platform area polygons.

#### obstacles

- Signature: def obstacles(self) -> list[Any]
- Location: [evacusim/jps/simulation_interface.py#L196](../../../evacusim/jps/simulation_interface.py#L196)
- Summary: Get obstacle geometries.

#### evacuation_exits

- Signature: def evacuation_exits(self) -> dict[str, int]
- Location: [evacusim/jps/simulation_interface.py#L206](../../../evacusim/jps/simulation_interface.py#L206)
- Summary: Get evacuation exit identifiers.

#### evacuation_journeys

- Signature: def evacuation_journeys(self) -> dict[str, int]
- Location: [evacusim/jps/simulation_interface.py#L216](../../../evacusim/jps/simulation_interface.py#L216)
- Summary: Get evacuation journey identifiers.

#### generate_spawn_positions

- Signature: def generate_spawn_positions(self, num_agents, seed=42) -> list[tuple[float, float]]
- Location: [evacusim/jps/simulation_interface.py#L226](../../../evacusim/jps/simulation_interface.py#L226)
- Summary: Generate spawn positions for agents within the walkable geometry.

## Coverage Notes

- Nested helper functions documented in this file: 0
