# Deep Dive: Event Scheduling and Director Systems

## Event Engine Role

The event subsystem is the primary mechanism for non-agent external changes during runtime.

Key references:

- [EventManager.check_and_trigger_events](../function-reference/systems__event_manager.md)
- [EventManager.block_exit](../function-reference/systems__event_manager.md)

Event classes include:

- infrastructure changes (for example blocked exits),
- communication events (PA announcements),
- train lifecycle events,
- generic scheduled notifications.

## Scheduling Model

Events are loaded into runner setup and evaluated each simulation step. Repeating and one-shot patterns are both supported through event-manager checks.

Coupling point in runtime loop:

- [HybridSimulationRunner.run](../function-reference/coordination__hybrid_simulation.md)

A fired event can force immediate decision refresh for agents, bypassing normal stagger cadence.

## Director System Role

Director agents provide non-LLM, rule-driven behavior for guidance and patrol logic.

Key references:

- [DirectorSystem.setup](../function-reference/systems__director_system.md)
- [DirectorSystem.step](../function-reference/systems__director_system.md)
- [DirectorSystem._normalize_phases](../function-reference/systems__director_system.md)

Core capabilities:

- phase-based behavior progression,
- trigger-based activation (time/event/reach-zone),
- movement modes (hold vs patrol),
- directive broadcast to nearby passengers.

## Interaction Between Events and Directors

Common interactions:

1. Event fires (for example exit blocked).
2. Director behavior may shift phase or patrol routing.
3. Director broadcasts updated guidance.
4. Passenger decisions are refreshed on next immediate cycle.

This layering lets the simulation represent both environmental change and social guidance.

## Design Tradeoffs

- Deterministic director logic improves reproducibility.
- It also limits adaptability compared with LLM-driven staff behavior.
- Event semantics are powerful but can become hard to validate without strong schema checks.

## Review Guidance

1. Confirm event ordering is deterministic for simultaneous events.
2. Validate idempotency for repeated event triggers.
3. Verify director phase transitions under edge trigger timing.
4. Check that director agents cannot deadlock patrol in invalid zones.
