# Runtime Lifecycle

## Startup Sequence

1. Config loading
- [ConfigLoader.load_and_validate](function-reference/config__config_loader.md)
- Includes inheritance via extends and deep merge.

2. Simulation construction
- [SimulationRunnerFactory.create_runner](function-reference/setup__simulation_runner_factory.md)
- [JuPedSimSetup.create_simulation](function-reference/setup__jupedsim_setup.md)

3. Geometry and exits
- [GeometryManager.__init__](function-reference/jps__geometry_manager.md)
- [ExitManager._setup_evacuation_exits](function-reference/jps__exit_manager.md)

4. Director pre-spawn systems
- [HybridSimulationRunner.build_systems_for_pre_spawn](function-reference/coordination__hybrid_simulation.md)
- [DirectorSystem.setup](function-reference/systems__director_system.md)

5. Station layout and agents
- [StationLayoutBuilder.build_layout](function-reference/setup__station_layout_builder.md)
- [AgentManager.create_and_populate_agents](function-reference/setup__agent_manager.md)

6. Concordia and decision runtime wiring
- [AgentBuilder.build_agents](function-reference/concordia__agent_builder.md)
- [DecisionProcessor.__init__](function-reference/decision__decision_processor.md)

7. Initial decision pass
- [HybridSimulationRunner._bootstrap_initial_decisions](function-reference/coordination__hybrid_simulation.md)

## Main Loop Sequence

Main control loop is in:
- [HybridSimulationRunner.run](function-reference/coordination__hybrid_simulation.md)

Each iteration generally executes:

1. Physics step
- [HybridSimulationRunner._step_jupedsim](function-reference/coordination__hybrid_simulation.md)

2. Exit checks and transfer handling
- [ExitTracker.check_exited_agents](function-reference/jps__exit_tracker.md)
- Multi-level transfer consumption and immediate redecision queue handling.

3. Event/system progression
- [EventManager.check_and_trigger_events](function-reference/systems__event_manager.md)
- [HybridSimulationRunner._step_systems](function-reference/coordination__hybrid_simulation.md)

4. Decision trigger check
- [HybridSimulationRunner._should_make_decisions](function-reference/coordination__hybrid_simulation.md)

5. Observation and decisions
- [ObservationCoordinator.generate_all_observations](function-reference/coordination__observation_coordinator.md)
- [DecisionProcessor.process_all_agents](function-reference/decision__decision_processor.md)

6. Output and analytics sampling
- Incremental save and positions sidecar updates.
- [PopulationMonitor.record_snapshot](function-reference/metrics__population_monitor.md)

## End-of-Run

- [ResultsWriter.save_final_results](function-reference/metrics__results_writer.md)
- [AnalyticsGenerator.save_all_analytics](function-reference/metrics__analytics_generator.md)
- [FinancialReporter.generate_report](function-reference/metrics__llm_cost_reporter.md)
