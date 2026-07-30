# Architecture Overview

## Core Concept

evacusim is a hybrid simulation engine that combines:

- JuPedSim for continuous pedestrian motion and collision avoidance.
- Concordia LLM agents for decision-making and communication.

The system runs physics at high frequency and decisions at lower frequency, with translation logic bridging geometric state and natural language observations/actions.

## Main Layers

1. Configuration and bootstrap
- [evacusim/config/config_loader.py](../../evacusim/config/config_loader.py)
- [evacusim/setup/simulation_runner_factory.py](../../evacusim/setup/simulation_runner_factory.py)
- [evacusim/setup/jupedsim_setup.py](../../evacusim/setup/jupedsim_setup.py)

2. Physics and geometry
- [evacusim/jps/geometry_manager.py](../../evacusim/jps/geometry_manager.py)
- [evacusim/jps/jupedsim_integration.py](../../evacusim/jps/jupedsim_integration.py)
- [evacusim/jps/multi_level_simulation.py](../../evacusim/jps/multi_level_simulation.py)

3. Coordination and loop orchestration
- [evacusim/coordination/hybrid_simulation.py](../../evacusim/coordination/hybrid_simulation.py)
- [evacusim/coordination/observation_coordinator.py](../../evacusim/coordination/observation_coordinator.py)

4. Translation and decision pipeline
- [evacusim/translation/observation_generator.py](../../evacusim/translation/observation_generator.py)
- [evacusim/decision/decision_processor.py](../../evacusim/decision/decision_processor.py)
- [evacusim/translation/action_translator.py](../../evacusim/translation/action_translator.py)
- [evacusim/decision/action_executor.py](../../evacusim/decision/action_executor.py)

5. Dynamic systems and events
- [evacusim/systems/event_manager.py](../../evacusim/systems/event_manager.py)
- [evacusim/systems/director_system.py](../../evacusim/systems/director_system.py)
- [evacusim/systems/messaging/message_system.py](../../evacusim/systems/messaging/message_system.py)

6. Outputs and analytics
- [evacusim/metrics/results_writer.py](../../evacusim/metrics/results_writer.py)
- [evacusim/metrics/population_monitor.py](../../evacusim/metrics/population_monitor.py)
- [evacusim/metrics/analytics_generator.py](../../evacusim/metrics/analytics_generator.py)

## Architectural Data Flow

- Config file -> validated config dict -> simulation objects and agents.
- JuPedSim state -> observation pipeline -> agent prompt context.
- Agent action JSON -> translator -> JuPedSim target/speed updates.
- Events, exits, and transfers -> immediate decision overrides when needed.
- Runtime data -> incremental and final output artifacts.

## Key Cross-Cutting Mechanisms

- Prompt caching for LLM cost and latency reduction.
- Staggered decision groups to smooth API load.
- Level-transfer manager for escalator-based movement across separate level simulations.
- Spatial analyzer for zone identification, line-of-sight, and visible exits.
- Event-driven overrides to force immediate reactions to critical changes.

## Where To Drill Down

- Runtime sequencing: [runtime-lifecycle.md](runtime-lifecycle.md)
- Function-level details: [function-reference/README.md](function-reference/README.md)
- Design rationale: [design-decisions.md](design-decisions.md)
- Risks and improvements: [risk-register.md](risk-register.md)
