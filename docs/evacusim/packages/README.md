# Package Guide

This guide explains what each package owns and how data moves across package boundaries.

## config

- Files: [evacusim/config](../../evacusim/config)
- Owns: loading, inheritance merge, CLI overrides, validation.
- Primary output: validated config dictionary.

## setup

- Files: [evacusim/setup](../../evacusim/setup)
- Owns: object assembly, station layout build, agent config/spawn.
- Primary outputs: initialized simulation, station_layout, agent config.

## jps

- Files: [evacusim/jps](../../evacusim/jps)
- Owns: JuPedSim interface/wrappers, geometry and exits, level transfers.
- Primary outputs: positions, nearby agent sets, active exits, transfer state.

## coordination

- Files: [evacusim/coordination](../../evacusim/coordination)
- Owns: simulation loop orchestration, observation scheduling, state query helpers.

## translation

- Files: [evacusim/translation](../../evacusim/translation)
- Owns: state-to-text observations, text/json action to simulation coordinates and speed.

## decision

- Files: [evacusim/decision](../../evacusim/decision)
- Owns: prompt assembly, cache checks, concurrent per-agent decision execution.

## systems

- Files: [evacusim/systems](../../evacusim/systems)
- Owns: event scheduling/firing, director patrol/broadcast behavior, messaging.

## concordia

- Files: [evacusim/concordia](../../evacusim/concordia)
- Owns: Concordia entity construction and Azure OpenAI provider integration.

## metrics

- Files: [evacusim/metrics](../../evacusim/metrics)
- Owns: persistence and analytics artifacts.

## utils

- Files: [evacusim/utils](../../evacusim/utils)
- Owns: shared sampling/logging/performance utility functions.

## visualization

- Files: [evacusim/visualization](../../evacusim/visualization)
- Owns: live viewers, position history, and offline video generation.

## Full Function-Level Documentation

Use [../function-reference/README.md](../function-reference/README.md) for complete per-file class/function/method details.
