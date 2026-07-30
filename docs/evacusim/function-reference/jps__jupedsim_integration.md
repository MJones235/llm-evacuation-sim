# Function Reference: evacusim/jps/jupedsim_integration.py

- Source file: [evacusim/jps/jupedsim_integration.py](../../../evacusim/jps/jupedsim_integration.py)
- Top-level classes: 1
- Top-level functions: 0

## Module Summary

Real JuPedSim integration for Concordia station evacuation simulation.

## Classes and Methods

### Class ConcordiaJuPedSimulation

- Location: [evacusim/jps/jupedsim_integration.py#L29](../../../evacusim/jps/jupedsim_integration.py#L29)
- Summary: Real JuPedSim simulation wrapper for Concordia integration.
- Method count: 16

#### __init__

- Signature: def __init__(self, network_path=None, dt=0.05, exit_radius=10.0, level_id=0, initially_blocked_exits=None)
- Location: [evacusim/jps/jupedsim_integration.py#L40](../../../evacusim/jps/jupedsim_integration.py#L40)
- Summary: Initialize JuPedSim simulation with station geometry.
- Key calls: str, GeometryManager, StageManager, ExitManager, AgentTracker, logger.info, ValueError, len

#### add_geometry_obstacle_for_exit

- Signature: def add_geometry_obstacle_for_exit(self, exit_name) -> None
- Location: [evacusim/jps/jupedsim_integration.py#L92](../../../evacusim/jps/jupedsim_integration.py#L92)
- Summary: Add a physical barrier that prevents agent access to a blocked exit.
- Key calls: exit_name.split, self.geometry_manager.escalator_corridors.get, next, buffer, self.geometry_manager.add_obstacle_polygon, logger.info, len, logger.warning, shapes.append, _union

#### add_agent

- Signature: def add_agent(self, agent_id, position, walking_speed=1.34) -> None
- Location: [evacusim/jps/jupedsim_integration.py#L141](../../../evacusim/jps/jupedsim_integration.py#L141)
- Summary: Add an agent to the simulation.
- Key calls: self.exit_manager.get_default_exit, self.simulation.add_agent, self.agent_tracker.add_agent, logger.info, jps.CollisionFreeSpeedModelAgentParameters

#### step

- Signature: def step(self) -> bool
- Location: [evacusim/jps/jupedsim_integration.py#L180](../../../evacusim/jps/jupedsim_integration.py#L180)
- Summary: Advance simulation by one timestep.
- Key calls: self.simulation.iterate, self.simulation.agent_count, logger.info

#### get_agent_position

- Signature: def get_agent_position(self, agent_id) -> tuple[float, float] | None
- Location: [evacusim/jps/jupedsim_integration.py#L202](../../../evacusim/jps/jupedsim_integration.py#L202)
- Summary: Get agent's current position.
- Key calls: self.agent_tracker.get_position

#### set_agent_target

- Signature: def set_agent_target(self, agent_id, target) -> None
- Location: [evacusim/jps/jupedsim_integration.py#L214](../../../evacusim/jps/jupedsim_integration.py#L214)
- Summary: Set an agent's movement target by creating a waypoint.
- Key calls: self.agent_tracker.get_jps_id, self.agent_tracker.set_target, self.simulation.add_waypoint_stage, jps.JourneyDescription, self.simulation.add_journey, self.simulation.switch_agent_journey, logger.info, self.agent_tracker.is_agent_active, logger.debug

#### set_agent_evacuation_exit

- Signature: def set_agent_evacuation_exit(self, agent_id, exit_name) -> None
- Location: [evacusim/jps/jupedsim_integration.py#L249](../../../evacusim/jps/jupedsim_integration.py#L249)
- Summary: Direct an agent to a specific evacuation exit.
- Key calls: self.agent_tracker.get_jps_id, self.exit_manager.get_exit_ids, self.simulation.switch_agent_journey, logger.info, self.agent_tracker.is_agent_active, logger.debug

#### set_agent_speed

- Signature: def set_agent_speed(self, agent_id, speed) -> None
- Location: [evacusim/jps/jupedsim_integration.py#L278](../../../evacusim/jps/jupedsim_integration.py#L278)
- Summary: Set an agent's walking speed mid-simulation.
- Key calls: self.agent_tracker.get_jps_id, self.simulation.agent, logger.info, self.agent_tracker.is_agent_active, logger.debug

#### get_agent_speed

- Signature: def get_agent_speed(self, agent_id) -> float | None
- Location: [evacusim/jps/jupedsim_integration.py#L301](../../../evacusim/jps/jupedsim_integration.py#L301)
- Summary: Get an agent's current desired walking speed.
- Key calls: self.agent_tracker.get_jps_id, self.simulation.agent, float, self.agent_tracker.is_agent_active

#### get_nearby_agents

- Signature: def get_nearby_agents(self, agent_id, radius) -> list[dict[str, Any]]
- Location: [evacusim/jps/jupedsim_integration.py#L318](../../../evacusim/jps/jupedsim_integration.py#L318)
- Summary: Get information about agents within radius of given agent.
- Key calls: self.agent_tracker.get_nearby_agents

#### get_all_nearby_agents_bulk

- Signature: def get_all_nearby_agents_bulk(self, radius) -> dict[str, list[dict[str, Any]]]
- Location: [evacusim/jps/jupedsim_integration.py#L331](../../../evacusim/jps/jupedsim_integration.py#L331)
- Summary: Return nearby-agent lists for ALL agents in a single O(n) pass.
- Key calls: self.agent_tracker.get_all_nearby_agents_bulk

#### get_simulation_time

- Signature: def get_simulation_time(self) -> float
- Location: [evacusim/jps/jupedsim_integration.py#L346](../../../evacusim/jps/jupedsim_integration.py#L346)
- Summary: Get current simulation time in seconds.

#### get_all_agent_positions

- Signature: def get_all_agent_positions(self) -> dict[str, tuple[float, float]]
- Location: [evacusim/jps/jupedsim_integration.py#L350](../../../evacusim/jps/jupedsim_integration.py#L350)
- Summary: Get positions of all agents for visualization.
- Key calls: self.agent_tracker.get_all_positions

#### check_exits

- Signature: def check_exits(self) -> dict[str, str]
- Location: [evacusim/jps/jupedsim_integration.py#L359](../../../evacusim/jps/jupedsim_integration.py#L359)
- Summary: Check for agents that have exited this level and determine exit type.
- Key calls: self.agent_tracker.get_all_positions, self.last_known_positions.update, set, self.agent_tracker.agent_ids.keys, current_positions.keys, len, logger.info, self.agent_assigned_exits.get, self.agent_tracker.agent_targets.get, float

#### get_geometry

- Signature: def get_geometry(self) -> dict[str, Any]
- Location: [evacusim/jps/jupedsim_integration.py#L445](../../../evacusim/jps/jupedsim_integration.py#L445)
- Summary: Get geometry information for visualization.
- Key calls: self.geometry_manager.get_geometry_data, get, self.exit_manager.evacuation_exits.keys

#### generate_spawn_positions

- Signature: def generate_spawn_positions(self, num_agents, seed=42) -> list[tuple[float, float]]
- Location: [evacusim/jps/jupedsim_integration.py#L463](../../../evacusim/jps/jupedsim_integration.py#L463)
- Summary: Generate spawn positions for agents within the walkable geometry.
- Key calls: random.seed, list, sum, logger.info, enumerate, RuntimeError, walkable_areas.items, int, len, logger.warning

## Coverage Notes

- Nested helper functions documented in this file: 0
