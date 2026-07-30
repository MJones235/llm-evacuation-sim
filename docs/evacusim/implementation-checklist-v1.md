# Implementation Checklist v1

This checklist is intentionally execution-focused: concrete tasks, verification steps, and definition-of-done gates.

## Scope for v1

- Workstream A: Escalator arrival de-clustering via immediate re-decision model.
- Workstream B: Simplified escalator speed semantics.
- Workstream E: Significant-change decision gating and route commitment.
- Minimal Workstream F: centralize key realism constants in one config block.

## Phase 1: Escalator Exit -> Immediate Next Action (Workstream A)

### Tasks

1. Remove semantic dependence on transfer temporary waypoint as a destination.
2. On level transfer completion, mark agent for immediate decision eligibility in the same loop cycle.
3. Add anti-block guard for escalator context:
- If agent is in escalator corridor/arrival zone, ignore in-place wait outcomes and keep moving to nearest valid local egress point.
4. Keep a short fallback movement nudge only when no valid post-transfer target exists.

### Code Touchpoints

- [evacusim/jps/multi_level_simulation.py](../../evacusim/jps/multi_level_simulation.py)
- [evacusim/coordination/hybrid_simulation.py](../../evacusim/coordination/hybrid_simulation.py)
- [evacusim/decision/decision_processor.py](../../evacusim/decision/decision_processor.py)
- [evacusim/decision/action_executor.py](../../evacusim/decision/action_executor.py)

### Verification

1. Run a scenario with high escalator throughput.
2. Inspect trajectories near transfer exits:
- no persistent clustering at one synthetic point,
- agents clear escalator context before stopping/waiting.
3. Confirm immediate decision trigger occurs after transfer.

### Definition of Done

- No single-point transfer attractor in replay.
- No standstill blocking in escalator arrival/corridor context.
- Transfer-to-decision behavior is explainable in one sentence.

## Phase 2: Simplified Escalator Speed Rule (Workstream B)

### Tasks

1. Implement one escalator speed rule:
- effective speed >= belt speed,
- optional small uplift cap above belt speed.
2. Clamp/escalate speed commands while in escalator context to this rule.
3. Add clear inline comments describing this as intentional simplification.

### Code Touchpoints

- [evacusim/jps/multi_level_simulation.py](../../evacusim/jps/multi_level_simulation.py)
- [evacusim/decision/action_executor.py](../../evacusim/decision/action_executor.py)
- [evacusim/utils/speed_utils.py](../../evacusim/utils/speed_utils.py)

### Verification

1. Record sampled speeds for agents in escalator context.
2. Confirm no escalator agents are below belt speed.
3. Confirm upper bound is respected for escalator-context speed.

### Definition of Done

- Escalator speed behavior is deterministic and explainable.
- No stalled agents on escalators.

## Phase 3: Significant-Change Re-Decision Gating (Workstream E)

### Tasks

1. Define significant-change triggers in code (v1 set):
- new event fired,
- selected exit blocked/inactive,
- transfer completion,
- major local congestion shift.
2. Route commitment default:
- if no significant change, keep current journey and skip new LLM decision.
3. Integrate trigger logging with reason codes for each forced re-decision.

### Code Touchpoints

- [evacusim/coordination/hybrid_simulation.py](../../evacusim/coordination/hybrid_simulation.py)
- [evacusim/decision/decision_processor.py](../../evacusim/decision/decision_processor.py)
- [evacusim/decision/prompt_cache.py](../../evacusim/decision/prompt_cache.py)
- [evacusim/decision/action_executor.py](../../evacusim/decision/action_executor.py)

### Verification

1. Compare before/after LLM calls per simulated minute.
2. Confirm unchanged environments do not trigger repeated re-decisions.
3. Confirm trigger events always force re-decision.

### Definition of Done

- Lower LLM call rate without obvious stale behavior.
- Re-decision causes are auditable via reason codes.

## Phase 4: Minimal Calibration Block (Workstream F, Minimal)

### Tasks

1. Add `behavior_calibration` section to config handling.
2. Move v1 constants into that block:
- escalator belt speed,
- escalator speed uplift cap,
- transfer cooldown,
- blocked-exit discovery radius,
- significant-change thresholds.
3. Add concise docs for each parameter and expected effect.

### Code Touchpoints

- [evacusim/config/config_loader.py](../../evacusim/config/config_loader.py)
- modules currently owning these constants
- [docs/evacusim/architecture.md](architecture.md)
- [docs/evacusim/risk-register.md](risk-register.md)

### Verification

1. Scenario runs with defaults unchanged.
2. Scenario runs with custom calibration values applied.
3. Logs show loaded calibration values at startup.

### Definition of Done

- One clear configuration location for realism-critical thresholds.

## Cross-Phase Logging Requirements

1. Add structured reason codes for:
- transfer deferred,
- transfer blocked,
- escalator anti-block wait override,
- re-decision trigger type,
- exit-validation anomaly.
2. Include aggregate counts in final outputs.

## PR Slicing (Recommended)

1. PR-1: Phase 1 only.
2. PR-2: Phase 2 only.
3. PR-3: Phase 3 only.
4. PR-4: Phase 4 plus docs updates.

Each PR should include:

- targeted code changes,
- one scenario validation note,
- before/after metrics snippet,
- explicit regression risks.

## v1 Exit Criteria

1. Escalator top clustering artifact removed.
2. Escalator non-stall behavior guaranteed.
3. Significant-change gating reduces unnecessary LLM calls.
4. Core realism constants are centralized and documented.
5. Behavior is simple to explain to non-technical reviewers.
