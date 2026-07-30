# Function Reference: evacusim/systems/director_system.py

- Source file: [evacusim/systems/director_system.py](../../../evacusim/systems/director_system.py)
- Top-level classes: 1
- Top-level functions: 1

## Module Summary

Director System A configurable rule-based system for non-LLM director agents.

## Top-Level Functions

### _normalize_phases

- Signature: def _normalize_phases(system_config) -> list[dict]
- Location: [evacusim/systems/director_system.py#L104](../../../evacusim/systems/director_system.py#L104)
- Summary: Convert either a flat config or a ``phases`` list into a normalised list of phase dicts.
- Key calls: raw.get, normalized.append, system_config.get, float, str

## Classes and Methods

### Class DirectorSystem

- Location: [evacusim/systems/director_system.py#L164](../../../evacusim/systems/director_system.py#L164)
- Summary: Rule-based director agent system driven entirely by configuration.
- Method count: 12

#### __init__

- Signature: def __init__(self, system_name, system_config) -> None
- Location: [evacusim/systems/director_system.py#L174](../../../evacusim/systems/director_system.py#L174)
- Summary: No docstring summary available.
- Key calls: system_config.get, float, _normalize_phases, system_name.replace

#### setup

- Signature: def setup(self, jps_sim, station_layout, agent_roles) -> list[str]
- Location: [evacusim/systems/director_system.py#L237](../../../evacusim/systems/director_system.py#L237)
- Summary: Spawn director agents in JuPedSim and register their roles.
- Key calls: station_layout.get, enumerate, logger.info, hasattr, self._resolve_patrol_waypoints, str, self._resolve_position, logger.warning, entry.get, self.agent_ids.append

#### notify_event_fired

- Signature: def notify_event_fired(self) -> None
- Location: [evacusim/systems/director_system.py#L344](../../../evacusim/systems/director_system.py#L344)
- Summary: Signal that a simulation event has fired.
- Key calls: logger.info

#### step

- Signature: def step(self, current_sim_time, jps_sim, message_system, state_queries, exited_agents, zone_id_for_agent_fn=None) -> None
- Location: [evacusim/systems/director_system.py#L353](../../../evacusim/systems/director_system.py#L353)
- Summary: Per-step update for all director agents.
- Key calls: self._agent_phase.get, state_queries.get_agent_position, self._last_directive_time.get, message_system.deliver_directive, self._is_trigger_met, hasattr, self._phase_transition_grace_until.get, logger.debug, self._step_patrol, len

#### _is_trigger_met

- Signature: def _is_trigger_met(self, agent_id, phase, current_sim_time, jps_sim, zone_id_for_agent_fn) -> bool
- Location: [evacusim/systems/director_system.py#L438](../../../evacusim/systems/director_system.py#L438)
- Summary: No docstring summary available.
- Key calls: zone_id_for_agent_fn, self._agent_phase_activated_at.get, self._phase_transition_grace_until.get, hasattr, phase.get, jps_sim.agent_levels.get

#### _activate_phase

- Signature: def _activate_phase(self, agent_id, phase_idx, current_sim_time, jps_sim, state_queries) -> None
- Location: [evacusim/systems/director_system.py#L479](../../../evacusim/systems/director_system.py#L479)
- Summary: Transition an agent to the given phase index.
- Key calls: state_queries.get_agent_position, self._init_movement, logger.info, hasattr, jps_sim.agent_levels.get

#### _init_movement

- Signature: def _init_movement(self, agent_id, phase_idx, jps_sim, position, current_level) -> None
- Location: [evacusim/systems/director_system.py#L515](../../../evacusim/systems/director_system.py#L515)
- Summary: Set an initial JuPedSim target for the agent entering a phase.
- Key calls: jps_sim.set_agent_target

#### _step_hold_agent

- Signature: def _step_hold_agent(self, agent_id, phase, position, current_sim_time, jps_sim) -> None
- Location: [evacusim/systems/director_system.py#L541](../../../evacusim/systems/director_system.py#L541)
- Summary: Maintain a stationary hold target for one agent.
- Key calls: self._hold_targets.get, phase.get, self._zones_polygons_ref.get, jps_sim.set_agent_target, self._last_hold_refresh.get

#### _step_patrol

- Signature: def _step_patrol(self, agent_id, phase, current_sim_time, jps_sim, position) -> None
- Location: [evacusim/systems/director_system.py#L578](../../../evacusim/systems/director_system.py#L578)
- Summary: Advance patrol waypoint logic for a single director agent (phase-aware).
- Key calls: self._patrol_index.get, self._patrol_arrived_at.get, hasattr, jps_sim.agent_levels.get, self._cross_level_routing.get, logger.debug, self._route_to_escalator, jps_sim.set_agent_target, len

#### _route_to_escalator

- Signature: def _route_to_escalator(self, agent_id, position, current_level, target_level, jps_sim) -> None
- Location: [evacusim/systems/director_system.py#L668](../../../evacusim/systems/director_system.py#L668)
- Summary: Set the agent's target to the nearest *departure* escalator zone on ``current_level`` that leads toward ``target_level``.
- Key calls: float, esc_zones.items, hasattr, logger.debug, logger.warning, best_zone.split, jps_sim.transfer_manager.escalator_zones.items, name.startswith, name.endswith, len

#### _resolve_patrol_waypoints

- Signature: def _resolve_patrol_waypoints(self, entries, zones_polygons) -> list[dict]
- Location: [evacusim/systems/director_system.py#L746](../../../evacusim/systems/director_system.py#L746)
- Summary: Resolve patrol zone entries to dicts of {pos: (x, y), level_id: str}.
- Key calls: self._resolve_position, str, waypoints.append, entry.get

#### _resolve_position

- Signature: def _resolve_position(entry, zones_polygons, label) -> tuple[float, float] | None
- Location: [evacusim/systems/director_system.py#L761](../../../evacusim/systems/director_system.py#L761)
- Summary: Resolve a spawn / patrol entry to (x, y).
- Key calls: entry.get, logger.warning, zones_polygons.get, float, list, zones_polygons.keys

## Coverage Notes

- Nested helper functions documented in this file: 0
