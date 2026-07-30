# Function Reference: evacusim/setup/simulation_runner_factory.py

- Source file: [evacusim/setup/simulation_runner_factory.py](../../../evacusim/setup/simulation_runner_factory.py)
- Top-level classes: 1
- Top-level functions: 0

## Module Summary

Simulation runner factory for Station Concordia simulations.

## Classes and Methods

### Class SimulationRunnerFactory

- Location: [evacusim/setup/simulation_runner_factory.py#L19](../../../evacusim/setup/simulation_runner_factory.py#L19)
- Summary: Handles creation and configuration of simulation runners.
- Method count: 2

#### create_runner

- Signature: def create_runner(jps_sim, agents_config, station_layout, model, embedder, decisions_file, config, pace_to_realtime=False, pre_built_systems=None, pre_built_agent_roles=None) -> HybridSimulationRunner
- Location: [evacusim/setup/simulation_runner_factory.py#L23](../../../evacusim/setup/simulation_runner_factory.py#L23)
- Summary: Create and configure a HybridSimulationRunner.
- Key calls: config.get, sim_config.get, video_config.get, logger.info, SimulationRunnerFactory._load_events, HybridSimulationRunner, logger.error, traceback.print_exc

#### _load_events

- Signature: def _load_events(runner, config) -> None
- Location: [evacusim/setup/simulation_runner_factory.py#L104](../../../evacusim/setup/simulation_runner_factory.py#L104)
- Summary: Load events from configuration into the runner.
- Key calls: config.get, record.setdefault, runner.event_manager.scheduled_events.append, logger.info, logger.warning, event.items, len

## Coverage Notes

- Nested helper functions documented in this file: 0
