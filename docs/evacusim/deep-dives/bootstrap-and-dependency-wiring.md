# Deep Dive: Bootstrap and Dependency Wiring

## Why This Matters

Bootstrap logic determines whether all subsystems receive consistent configuration and shared state. Most downstream behavior quality depends on this stage being deterministic.

## Primary Construction Path

Key entry points:

- [ConfigLoader.load_and_validate](../function-reference/config__config_loader.md)
- [SimulationRunnerFactory.create_runner](../function-reference/setup__simulation_runner_factory.md)
- [JuPedSimSetup.create_simulation](../function-reference/setup__jupedsim_setup.md)
- [HybridSimulationRunner.__init__](../function-reference/coordination__hybrid_simulation.md)

Typical assembly order:

1. YAML config is loaded, merged, and validated.
2. JuPedSim simulation object is created (single-level or multi-level).
3. Station layout and exits are built from geometry and config.
4. Agent configs are created and populated into simulation.
5. Concordia agents are built.
6. Decision, translation, event, system, and metrics components are wired.

## Dependency Boundaries

Core boundaries to watch:

- Setup package owns object creation.
- Coordination package owns lifecycle and sequencing.
- Decision and translation packages depend on shared runtime dictionaries and layout structures.
- JPS package owns geometry and movement primitives.

Important shared mutable state appears in runner and decision paths, such as agent decisions, destinations, observations, and exited sets.

## Multi-Level Branching

Multi-level setup introduces additional wiring:

- [MultiLevelJuPedSimulation.__init__](../function-reference/jps__multi_level_simulation.md)
- [LevelTransferManager.__init__](../function-reference/coordination__level_transfer_manager.md)
- level-specific geometry/exit managers and agent-level mappings.

This branch affects many later assumptions in translation, visibility, and action routing.

## Typical Failure Surfaces

- Inconsistent layout keys expected by translation and decision modules.
- Missing geometry naming conventions for escalator and platform logic.
- Divergence between single-level and multi-level behavior paths.
- Silent config schema drift in optional sections.

## Review Guidance

When reviewing bootstrap logic:

1. Verify all required runtime maps are populated before first decision cycle.
2. Confirm per-level structures are normalized to a shared interface.
3. Validate that event/system setup happens before loop start.
4. Confirm t=0 bootstrap decision behavior is intentionally configured.
