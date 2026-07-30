# Function Reference: evacusim/jps/multi_level_simulation.py

- Source file: [evacusim/jps/multi_level_simulation.py](../../../evacusim/jps/multi_level_simulation.py)
- Top-level classes: 1
- Top-level functions: 0

## Module Summary

Multi-level JuPedSim simulation for Monument Station.

## Classes and Methods

### Class MultiLevelJuPedSimulation

- Location: [evacusim/jps/multi_level_simulation.py#L25](../../../evacusim/jps/multi_level_simulation.py#L25)
- Summary: Manages multiple JuPedSim simulations for multi-level stations.
- Method count: 24

#### __init__

- Signature: def __init__(self, network_path, dt=0.05, exit_radius=10.0, levels=None, escalator_belt_speed=0.5, level_arrival_waypoints=None, initially_blocked_exits=None)
- Location: [evacusim/jps/multi_level_simulation.py#L38](../../../evacusim/jps/multi_level_simulation.py#L38)
- Summary: Initialize multi-level simulation.
- Key calls: Path, set, LevelTransferManager, self._classify_escalator_zones, logger.info, str, ConcordiaJuPedSimulation, float, items, len

#### _classify_escalator_zones

- Signature: def _classify_escalator_zones(self) -> dict[str, bool]
- Location: [evacusim/jps/multi_level_simulation.py#L161](../../../evacusim/jps/multi_level_simulation.py#L161)
- Summary: Classify every escalator zone as a departure (True) or arrival (False) zone.
- Key calls: self._ESCALATOR_ZONE_RE.match, m.groups, self.simulations.get

#### _enforce_escalator_constraints

- Signature: def _enforce_escalator_constraints(self) -> None
- Location: [evacusim/jps/multi_level_simulation.py#L187](../../../evacusim/jps/multi_level_simulation.py#L187)
- Summary: Per-step escalator physics enforcement — called every simulation step.
- Key calls: self.simulations.items, getattr, list, sim.agent_tracker.agent_ids.keys, sim.get_agent_position, Point, self.transfer_manager.escalator_zones.items, escalator_corridors.items, self._ESCALATOR_ZONE_RE.match, sim.get_agent_speed

#### _find_nearest_valid_exit_for_level

- Signature: def _find_nearest_valid_exit_for_level(self, pos, level_id) -> str | None
- Location: [evacusim/jps/multi_level_simulation.py#L311](../../../evacusim/jps/multi_level_simulation.py#L311)
- Summary: Return the name of the nearest registered exit on *level_id* to *pos*.
- Key calls: self.simulations.get, min, exits.items, name.startswith, math.hypot

#### get_agent_speed

- Signature: def get_agent_speed(self, agent_id) -> float | None
- Location: [evacusim/jps/multi_level_simulation.py#L342](../../../evacusim/jps/multi_level_simulation.py#L342)
- Summary: Get an agent's current desired walking speed.
- Key calls: get_agent_speed

#### geometry_manager

- Signature: def geometry_manager(self)
- Location: [evacusim/jps/multi_level_simulation.py#L360](../../../evacusim/jps/multi_level_simulation.py#L360)
- Summary: Get geometry manager from level 0 (concourse level).

#### add_agent

- Signature: def add_agent(self, agent_id, position, walking_speed=1.34, level_id='0') -> None
- Location: [evacusim/jps/multi_level_simulation.py#L372](../../../evacusim/jps/multi_level_simulation.py#L372)
- Summary: Add an agent to a specific level.
- Key calls: add_agent, logger.info, ValueError

#### step

- Signature: def step(self) -> bool
- Location: [evacusim/jps/multi_level_simulation.py#L396](../../../evacusim/jps/multi_level_simulation.py#L396)
- Summary: Advance all level simulations and process agent transfers between levels.
- Key calls: self._process_escalator_exits, self.simulations.values, self._enforce_escalator_constraints, sum, sim.step, logger.info, sim.simulation.agent_count

#### _process_escalator_exits

- Signature: def _process_escalator_exits(self, blocked_exits=None)
- Location: [evacusim/jps/multi_level_simulation.py#L430](../../../evacusim/jps/multi_level_simulation.py#L430)
- Summary: Check each level for agents that have exited through escalators.
- Key calls: self._pending_spawn_positions.clear, self.simulations.items, set, self._deferred_transfers.clear, sim.check_exits, exited_agents.items, logger.info, self._transfer_agent_through_escalator, logger.debug, self._last_transfer_step.get

#### _transfer_agent_through_escalator

- Signature: def _transfer_agent_through_escalator(self, agent_id, current_level, exit_name)
- Location: [evacusim/jps/multi_level_simulation.py#L523](../../../evacusim/jps/multi_level_simulation.py#L523)
- Summary: Transfer an agent from one level to another through an escalator.
- Key calls: exit_name.split, next, target_zone_poly.buffer, range, self._pending_spawn_positions.append, self._corridor_routed_exit.pop, logger.warning, len, logger.error, list

#### add_geometry_obstacle_for_exit

- Signature: def add_geometry_obstacle_for_exit(self, exit_name) -> None
- Location: [evacusim/jps/multi_level_simulation.py#L670](../../../evacusim/jps/multi_level_simulation.py#L670)
- Summary: Punch a geometry obstacle at the entrance of *exit_name* on every level.
- Key calls: self.simulations.items, sim.add_geometry_obstacle_for_exit, debug, _gl

#### consume_recently_transferred_agents

- Signature: def consume_recently_transferred_agents(self) -> set[str]
- Location: [evacusim/jps/multi_level_simulation.py#L686](../../../evacusim/jps/multi_level_simulation.py#L686)
- Summary: Return and clear agents transferred since last consume call.
- Key calls: set, self.recently_transferred_agents.clear

#### get_agent_position

- Signature: def get_agent_position(self, agent_id) -> tuple[float, float] | None
- Location: [evacusim/jps/multi_level_simulation.py#L692](../../../evacusim/jps/multi_level_simulation.py#L692)
- Summary: Get agent's current position.
- Key calls: get_agent_position

#### get_agent_level

- Signature: def get_agent_level(self, agent_id) -> str | None
- Location: [evacusim/jps/multi_level_simulation.py#L708](../../../evacusim/jps/multi_level_simulation.py#L708)
- Summary: Get the level an agent is currently on.
- Key calls: self.agent_levels.get

#### set_agent_target

- Signature: def set_agent_target(self, agent_id, target) -> None
- Location: [evacusim/jps/multi_level_simulation.py#L712](../../../evacusim/jps/multi_level_simulation.py#L712)
- Summary: Set an agent's movement target on their current level.
- Key calls: set_agent_target

#### set_agent_evacuation_exit

- Signature: def set_agent_evacuation_exit(self, agent_id, exit_name) -> None
- Location: [evacusim/jps/multi_level_simulation.py#L720](../../../evacusim/jps/multi_level_simulation.py#L720)
- Summary: Direct an agent to a specific evacuation exit on their current level.
- Key calls: re.match, level_sim.set_agent_evacuation_exit, esc_m.group, KeyError, logger.error, list, level_sim.exit_manager.evacuation_exits.keys

#### set_agent_speed

- Signature: def set_agent_speed(self, agent_id, speed) -> None
- Location: [evacusim/jps/multi_level_simulation.py#L780](../../../evacusim/jps/multi_level_simulation.py#L780)
- Summary: Set an agent's walking speed.
- Key calls: set_agent_speed

#### get_nearby_agents

- Signature: def get_nearby_agents(self, agent_id, radius) -> list[dict[str, Any]]
- Location: [evacusim/jps/multi_level_simulation.py#L788](../../../evacusim/jps/multi_level_simulation.py#L788)
- Summary: Get information about agents within radius on the same level.
- Key calls: get_nearby_agents

#### get_all_nearby_agents_bulk

- Signature: def get_all_nearby_agents_bulk(self, radius) -> dict[str, list[dict[str, Any]]]
- Location: [evacusim/jps/multi_level_simulation.py#L796](../../../evacusim/jps/multi_level_simulation.py#L796)
- Summary: Return nearby-agent lists for ALL agents in a single pass per level.
- Key calls: self.simulations.values, result.update, sim.get_all_nearby_agents_bulk

#### get_simulation_time

- Signature: def get_simulation_time(self) -> float
- Location: [evacusim/jps/multi_level_simulation.py#L814](../../../evacusim/jps/multi_level_simulation.py#L814)
- Summary: Get current simulation time in seconds.

#### get_all_agent_positions

- Signature: def get_all_agent_positions(self) -> dict[str, tuple[float, float]]
- Location: [evacusim/jps/multi_level_simulation.py#L818](../../../evacusim/jps/multi_level_simulation.py#L818)
- Summary: Get positions of all agents across all levels.
- Key calls: self.simulations.values, sim.get_all_agent_positions, all_positions.update

#### get_geometry

- Signature: def get_geometry(self, level_id=None) -> dict[str, Any]
- Location: [evacusim/jps/multi_level_simulation.py#L831](../../../evacusim/jps/multi_level_simulation.py#L831)
- Summary: Get geometry information for visualization.
- Key calls: self.simulations.items, get_geometry, sim.get_geometry

#### board_agents_on_platform

- Signature: def board_agents_on_platform(self, exit_name, agent_destinations=None) -> list[str]
- Location: [evacusim/jps/multi_level_simulation.py#L850](../../../evacusim/jps/multi_level_simulation.py#L850)
- Summary: Board agents that have committed to boarding this train.
- Key calls: self.simulations.get, level_sim.geometry_manager.walkable_areas.get, level_sim.agent_tracker.get_all_positions, current_positions.items, logger.debug, level_sim.agent_tracker.agent_ids.get, level_sim.simulation.mark_agent_for_removal, boarded.append, logger.info, exit_name.rsplit

#### generate_spawn_positions

- Signature: def generate_spawn_positions(self, num_agents, seed=42) -> list[tuple[float, float, str]]
- Location: [evacusim/jps/multi_level_simulation.py#L920](../../../evacusim/jps/multi_level_simulation.py#L920)
- Summary: Generate spawn positions distributed across all levels.
- Key calls: random.seed, self.simulations.items, sum, enumerate, level_areas.values, sorted, level_areas.items, int, generate_spawn_positions, len

## Coverage Notes

- Nested helper functions documented in this file: 0
