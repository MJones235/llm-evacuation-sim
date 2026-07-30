# Function Reference: evacusim/decision/action_executor.py

- Source file: [evacusim/decision/action_executor.py](../../../evacusim/decision/action_executor.py)
- Top-level classes: 1
- Top-level functions: 0

## Module Summary

Action Executor Executes translated agent actions in the JuPedSim simulation, including: - Speed adjustments - Movement to exits, waypoints, or toward other agents - Waiting behavior (standing still or seeking information)

## Classes and Methods

### Class ActionExecutor

- Location: [evacusim/decision/action_executor.py#L21](../../../evacusim/decision/action_executor.py#L21)
- Summary: Applies translated actions to the JuPedSim simulation.
- Method count: 6

#### __init__

- Signature: def __init__(self, jps_sim, state_queries, event_manager, station_layout, agent_injured, agent_action, agent_last_decision, agent_destinations, wait_events, agent_configs)
- Location: [evacusim/decision/action_executor.py#L24](../../../evacusim/decision/action_executor.py#L24)
- Summary: Initialize action executor.

#### execute_action

- Signature: def execute_action(self, agent_id, translated_action, current_sim_time)
- Location: [evacusim/decision/action_executor.py#L63](../../../evacusim/decision/action_executor.py#L63)
- Summary: Apply a translated action to the JuPedSim simulation.
- Key calls: logger.info, translated_action.get, convert_speed_to_ms, self.state_queries.get_agent_position, self._handle_move_action, logger.error, logger.debug, self.jps_sim.set_agent_speed, self._handle_wait_action, self._safe_follow_target

#### _get_walkable_areas_for_agent

- Signature: def _get_walkable_areas_for_agent(self, agent_id) -> dict
- Location: [evacusim/decision/action_executor.py#L169](../../../evacusim/decision/action_executor.py#L169)
- Summary: Return the walkable-area polygons for the level the agent is on.
- Key calls: hasattr, self.jps_sim.agent_levels.get, self.jps_sim.simulations.get

#### _safe_follow_target

- Signature: def _safe_follow_target(self, agent_id, agent_pos, target_pos) -> tuple[float, float]
- Location: [evacusim/decision/action_executor.py#L180](../../../evacusim/decision/action_executor.py#L180)
- Summary: Return a waypoint toward *target_pos* that lies strictly inside a walkable polygon.
- Key calls: self._get_walkable_areas_for_agent, Point, walkable_areas.values, float, logger.debug, poly.contains, p.distance, best_poly.contains, shapely_nearest_points

#### _handle_move_action

- Signature: def _handle_move_action(self, agent_id, translated_action, target)
- Location: [evacusim/decision/action_executor.py#L243](../../../evacusim/decision/action_executor.py#L243)
- Summary: Handle move action: agent moving to exit, waypoint, or toward another agent.
- Key calls: translated_action.get, self.state_queries.get_agent_position, logger.debug, extract_exit_name, self._safe_follow_target, self.jps_sim.set_agent_target, self.jps_sim.set_agent_evacuation_exit, logger.info, logger.warning

#### _handle_wait_action

- Signature: def _handle_wait_action(self, agent_id, translated_action, current_sim_time)
- Location: [evacusim/decision/action_executor.py#L310](../../../evacusim/decision/action_executor.py#L310)
- Summary: Handle wait action: agent staying in place or seeking information.
- Key calls: translated_action.get, self.state_queries.get_agent_position, next, self.wait_events.append, random.uniform, Point, hasattr, walkable_areas.values, logger.debug, self.jps_sim.set_agent_target

## Coverage Notes

- Nested helper functions documented in this file: 0
