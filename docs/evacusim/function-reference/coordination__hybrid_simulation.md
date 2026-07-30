# Function Reference: evacusim/coordination/hybrid_simulation.py

- Source file: [evacusim/coordination/hybrid_simulation.py](../../../evacusim/coordination/hybrid_simulation.py)
- Top-level classes: 1
- Top-level functions: 0

## Module Summary

Hybrid simulation runner that integrates Concordia with JuPedSim.

## Classes and Methods

### Class HybridSimulationRunner

- Location: [evacusim/coordination/hybrid_simulation.py#L52](../../../evacusim/coordination/hybrid_simulation.py#L52)
- Summary: Manages the hybrid Concordia + JuPedSim simulation.
- Method count: 10

#### build_systems_for_pre_spawn

- Signature: def build_systems_for_pre_spawn(systems_config, jps_sim, station_layout) -> tuple[list, dict]
- Location: [evacusim/coordination/hybrid_simulation.py#L65](../../../evacusim/coordination/hybrid_simulation.py#L65)
- Summary: Create and set up director systems independently of runner init so that director agents occupy JuPedSim positions *before* random passengers are spawned.
- Key calls: systems_config.items, DirectorSystem, system.setup, systems.append, logger.info, cfg.get, len

#### __init__

- Signature: def __init__(self, jupedsim_simulation, agents_config, station_layout, language_model, embedder, decision_interval=5.0, max_steps=3600, output_file=None, enable_video=False, monitoring_config=None, performance_config=None, systems_config=None, pace_to_realtime=False, pre_built_systems=None, pre_built_agent_roles=None)
- Location: [evacusim/coordination/hybrid_simulation.py#L91](../../../evacusim/coordination/hybrid_simulation.py#L91)
- Summary: Initialize the hybrid simulation runner.
- Key calls: SimulationStateQueries, ActionTranslator, ObservationGenerator, set, AgentBuilder, asyncio.run, EventManager, ExitTracker, MessageSystem, PerformanceTimer

#### _init_systems

- Signature: def _init_systems(self, systems_config, jps_sim, station_layout) -> None
- Location: [evacusim/coordination/hybrid_simulation.py#L357](../../../evacusim/coordination/hybrid_simulation.py#L357)
- Summary: Initialise and setup all enabled rule-based systems from config.
- Key calls: systems_config.items, DirectorSystem, system.setup, self._staff_systems.append, logger.info, cfg.get, len

#### _step_systems

- Signature: def _step_systems(self, current_sim_time) -> None
- Location: [evacusim/coordination/hybrid_simulation.py#L372](../../../evacusim/coordination/hybrid_simulation.py#L372)
- Summary: Call step() on all active rule-based systems.
- Key calls: system.step

#### _get_zone_id_for_agent

- Signature: def _get_zone_id_for_agent(self, agent_id) -> str | None
- Location: [evacusim/coordination/hybrid_simulation.py#L384](../../../evacusim/coordination/hybrid_simulation.py#L384)
- Summary: Return the zone_id the given agent is currently in, or None.
- Key calls: self.state_queries.get_agent_position, getattr, _Point, zones_polygons.items, poly.covers, poly.contains

#### _bootstrap_initial_decisions

- Signature: def _bootstrap_initial_decisions(self) -> None
- Location: [evacusim/coordination/hybrid_simulation.py#L402](../../../evacusim/coordination/hybrid_simulation.py#L402)
- Summary: Run one decision cycle at t=0 before the first JuPedSim step.
- Key calls: logger.info, self.observation_coordinator.generate_all_observations, self.decision_processor.process_all_agents, logger.error

#### run

- Signature: def run(self) -> dict[str, Any]
- Location: [evacusim/coordination/hybrid_simulation.py#L420](../../../evacusim/coordination/hybrid_simulation.py#L420)
- Summary: Run the hybrid simulation.
- Key calls: logger.info, time.time, sum, len, print, self.population_monitor.record_snapshot, self.population_monitor.display_summary, self.population_monitor.to_dict, self._io_executor.shutdown, self.perf_timer.report

#### cleanup

- Signature: def cleanup(self)
- Location: [evacusim/coordination/hybrid_simulation.py#L776](../../../evacusim/coordination/hybrid_simulation.py#L776)
- Summary: Save partial results when simulation is interrupted.
- Key calls: logger.warning, self.position_tracker.save_to_file, logger.info, hasattr, ResultsWriter.save_final_results, self.jps_sim.get_all_agent_positions, len, self.perf_timer.report

#### _step_jupedsim

- Signature: def _step_jupedsim(self) -> bool
- Location: [evacusim/coordination/hybrid_simulation.py#L812](../../../evacusim/coordination/hybrid_simulation.py#L812)
- Summary: Advance JuPedSim simulation by one timestep.
- Key calls: hasattr, self.jps_sim.step, set, logger.error

#### _should_make_decisions

- Signature: def _should_make_decisions(self) -> bool
- Location: [evacusim/coordination/hybrid_simulation.py#L830](../../../evacusim/coordination/hybrid_simulation.py#L830)
- Summary: Check if it's time for agents to make decisions.

## Coverage Notes

- Nested helper functions documented in this file: 0
