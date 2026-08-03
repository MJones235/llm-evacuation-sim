# Escalator Rewrite Specification

## Status

- Date: 2026-07-31
- Owner: evacusim core simulation
- Scope: Multi-level movement between concourse and platform levels
- Purpose: Replace brittle escalator behavior with a deterministic, explicit, testable transfer system

Current implementation progress:

1. Phase A completed: explicit registry, edge schema, startup validation, state primitives.
2. Phase B in progress: transfer routing now resolves through registry edges only; legacy prefix/first-match target-zone selection removed.
3. Phase C started: escalator geometry now actively enforces direction and speed; escalator occupancy no longer excludes agents from decision cycles.

## Problem Statement

Current escalator behavior is unreliable because semantics are split across multiple mechanisms:

1. JuPedSim exit journeys
2. transfer-zone naming conventions
3. escalator corridor polygons
4. post-transfer decision scheduling
5. escalator decision suppression
6. speed overrides used as movement proxies

This has produced two classes of failures in recent runs:

1. Non-intent bounce loops (L0 -> L-1 -> L0) with no explicit up-intent decision.
2. Transfer deadlocks where agents become stationary in/near escalator areas.

## Design Goals

1. No hidden routing decisions.
2. All level transitions are explicit transfer events.
3. No reliance on default exits to move transferred agents.
4. Deterministic and auditable state transitions.
5. Geometry semantics are explicit and validated at startup.
6. Fail-fast on invalid escalator definitions.

## Non-Goals

1. This rewrite does not change LLM prompting semantics for non-escalator movement.
2. This rewrite does not introduce optimization work unrelated to transfer reliability.

## Core Principles

1. Escalators are transfer edges, not evacuation exits.
2. Transfer lifecycle is modeled by a state machine.
3. Decision suppression applies only to explicit in_transit state, not broad corridor occupancy.
4. Geometry parsing and transfer logic are separated from behavior execution.

## Target Architecture

### 1. Escalator Schema

Each escalator is represented explicitly as directed edges.

Required fields:

1. escalator_id
2. direction
3. from_level
4. to_level
5. from_zone_name
6. to_zone_name
7. from_corridor_name
8. to_corridor_name
9. optional travel_time_seconds
10. optional capacity

### 2. Escalator Agent State Machine

States:

1. normal_navigation
2. approaching_escalator
3. boarding
4. in_transit
5. disembarking

Allowed transitions:

1. normal_navigation -> approaching_escalator
2. approaching_escalator -> boarding
3. boarding -> in_transit
4. in_transit -> disembarking
5. disembarking -> normal_navigation

Invariants:

1. Agent in in_transit is not driven by default-exit journey assignment.
2. Agent entering disembarking is queued for immediate decision eligibility.
3. No automatic reroute to any other exit during boarding/in_transit/disembarking.

### 3. Transfer Trigger and Resolution

1. Trigger transfer only from explicit source zones that map to one edge.
2. Resolve destination using edge mapping, never by prefix/first-match lookup.
3. If mapping is missing/ambiguous, fail loudly and keep agent in safe pre-transfer state.

## Geometry Contract

Preferred long-term geometry model:

1. Explicit entry portal polygons per level and escalator endpoint.
2. Explicit exit portal polygons per level and escalator endpoint.
3. Optional corridor polygons for observation/analytics, not transfer semantics.

Interim compatibility model:

1. Parse existing L{level}_esc_{letter}_{direction} zones.
2. Parse existing L{level}_esc_corridor_{letter} corridors.
3. Build explicit edge registry and validate consistency at startup.

## Observability Requirements

Mandatory structured logs for:

1. state transitions
2. transfer edge selection
3. transfer start/complete
4. failed transfer validations
5. decision eligibility after disembark

Required per-agent transfer trace fields:

1. agent_id
2. time
3. previous_state
4. next_state
5. escalator_id
6. from_level
7. to_level
8. source_zone
9. target_zone

## Safety and Failure Handling

1. Do not auto-fallback-route transferred agents.
2. Do not silently substitute destination exits.
3. If transfer cannot resolve deterministically, raise a hard warning and hold agent in safe state while retrying with explicit diagnostics.

## Rollout Plan

### Phase A (start now): Registry and state scaffolding

Deliverables:

1. EscalatorController module with explicit schema dataclasses.
2. Registry builder from current geometry/transfer mappings.
3. Startup validation with warnings for inconsistencies.
4. Agent state primitives (no runtime behavior switch yet).

Acceptance:

1. Existing scenarios still run.
2. Startup logs include registry summary and validation outcomes.

### Phase B: Deterministic transfer edge cutover

Deliverables:

1. Replace prefix-based target zone selection with edge lookup.
2. Route transfer only through registry edges.

Acceptance:

1. No ambiguity in source->target transfer mapping.

### Phase C: State-machine driven runtime

Deliverables:

1. Introduce explicit state transitions in transfer pipeline.
2. Restrict decision suppression to in_transit only.
3. Force immediate decision eligibility at disembark.

Acceptance:

1. No transfer deadlocks attributable to escalator state handling.

### Phase D: Remove legacy escalator coupling

Deliverables:

1. Remove escalator-as-exit coupling for transfer behavior.
2. Remove redundant corridor/zone auto-routing logic.

Acceptance:

1. Transfer behavior depends only on controller and explicit decisions.

### Phase E: Hardening and tests

Deliverables:

1. Unit tests for registry build and validation.
2. Unit tests for state machine transitions.
3. Scenario tests for bounce-loop and stationary-regression prevention.

Acceptance gates:

1. Zero non-intent bounce loops.
2. Zero escalator deadlocks.
3. Deterministic replay for fixed seeds and configs.

## Immediate Implementation Notes

1. Phase A should not alter transfer behavior yet; it only builds the model and diagnostics.
2. Phase B is the first behavior-changing milestone and should be introduced with a feature flag to reduce migration risk.
