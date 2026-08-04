# EvacuSim Follow-up Implementation Plan (2026-08-04)

## Goals

1. Stabilize the new `pace` decision schema end-to-end.
2. Protect simulation fidelity after recent decision/transfer refactors.
3. Add observability to detect regressions quickly.

## Priority Tiers

### Tier 0 - Simulation quality critical

1. Pace multiplier wiring and calibration guardrails
- Scope:
  - Wire `hurrying` and `running` multipliers from scenario config.
  - Fail early if configured values are invalid.
  - Keep runtime fail-fast for unset multipliers when those pace values are used.
- Quality impact: High. Directly affects movement realism and can currently trigger runtime failures.
- Acceptance:
  - `normal_pace` leaves speed unchanged.
  - `hurrying`/`running` apply configured factors.
  - Missing or invalid factors produce clear errors.

2. Decision schema contract tests (prompt/validator/translator/executor)
- Scope:
  - Validate `pace=null` iff `action=wait`.
  - Validate non-wait actions require `pace in {normal_pace,hurrying,running}`.
  - Validate wait-only offered set omits pace schema text.
  - Validate malformed pace uses retry/repair path and does not coerce.
- Quality impact: High. Prevents interface drift across LLM and runtime layers.

3. Seeded KPI regression harness
- Scope:
  - Compare baseline vs current on evacuation time distribution, queueing near exits/escalators, wait-loop rate, and action mix.
- Quality impact: High. Ensures improvements do not trade off realism.

### Tier 1 - Important reliability and diagnostics

4. Transfer/discharge stress scenarios
- Scope:
  - Target high-density connector transitions and escalator-mouth congestion.
  - Validate no idle deadlocks and no oscillatory retargeting.
- Quality impact: Medium-high for multi-level scenarios.

5. Decision telemetry expansion
- Scope:
  - Emit per-run metrics for schema repair rate, fallback rate, pace distribution, cue-triggered redecision rate.
- Quality impact: Medium. Improves tuning and fault detection speed.

### Tier 2 - Nice to have

6. Prompt prose refinement
- Scope:
  - Clarify behavior wording without changing contract semantics.
- Quality impact: Low-medium.

7. Documentation refresh
- Scope:
  - Update docs and checklists to reflect pace schema and latest transfer behavior.
- Quality impact: Low direct runtime impact.

## Execution Order

1. Tier 0.1: pace multiplier wiring (start now)
2. Tier 0.2: schema contract tests
3. Tier 0.3: seeded KPI regression harness
4. Tier 1.4: transfer/discharge stress runs
5. Tier 1.5: telemetry expansion
6. Tier 2 docs/prompt cleanup

## Implementation Notes for Current Task

For pace multipliers, use scenario config key:
- `performance.pace_multipliers.hurrying`
- `performance.pace_multipliers.running`

Behavior:
- If missing: keep runtime fail-fast when that pace is selected.
- If present: must be positive floats.
- Recommend (not enforce) monotonicity: `running >= hurrying >= 1.0`.
