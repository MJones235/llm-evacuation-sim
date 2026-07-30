# Preliminary Fix Plan (For Review Before Implementation)

This plan is intentionally preliminary and scoped for discussion. It prioritizes realism, explainability, and code clarity while preserving current simulation functionality where possible.

## Objectives

1. Eliminate clearly non-realistic artifacts (for example escalator-top clustering).
2. Replace hardcoded heuristics with explicit, explainable behavior models.
3. Reduce hidden fallback behavior and ambiguous control paths.
4. Improve auditability of outcomes (especially anomalies and forced fallbacks).

## Review Alignment (2026-07-30)

This revision incorporates stakeholder guidance for simplicity and explainability:

1. Escalator model should stay minimal: exit escalator and choose next action.
2. Avoid overcomplicated escalator behavior (no lanes, no stand/walk sub-models for now).
3. Prevent escalator blocking from wait decisions.
4. Minimize unnecessary LLM calls via significant-change decision gating.

## Scope (First Pass)

- Escalator transfer arrival behavior and movement continuity.
- Escalator speed semantics (standing vs walking while on moving escalators).
- Transfer pipeline cleanup and level-routing generalization.
- Exit validation correctness and anomaly accounting.
- Target update churn reduction in movement execution.
- Configuration centralization for realism-critical constants.

## Proposed Workstreams

### Workstream A: Escalator Arrival De-Clustering

Problem:
- Shared temporary waypoint assignment after transfer creates artificial attractors and implies false semantic significance.

Proposal:
1. Remove the concept of a meaningful temporary post-transfer waypoint.
2. On transfer completion, force immediate next-action eligibility for the agent (event-style immediate decision path).
3. Keep movement continuity only as a safety fallback: if an agent has no valid next target yet, assign a short local egress nudge that is never treated as a destination goal.
4. Add explicit anti-block rule for escalator arrival/corridor contexts: waiting in-place is disallowed until the agent has cleared the escalator context.

Likely touchpoints:
- jps/multi_level_simulation.py
- coordination/hybrid_simulation.py
- decision/decision_processor.py
- decision/action_executor.py

Acceptance criteria:
- Agents do not accumulate around one artificial transfer point.
- Agents transfer, clear escalator context, and receive immediate next-action selection.
- No stand-still blocking inside escalator arrival/corridor geometry.

### Workstream B: Escalator Behavior Model

Problem:
- Current model is simple but needs clearer semantics for explainability.

Proposal:
1. Keep a single simplified escalator behavior model (no stand/walk states, no lanes).
2. Speed rule while in escalator context:
   - effective speed is at least belt speed,
   - agents may move slightly faster than belt speed within a small configurable cap.
3. Ignore or clamp implausible speed commands while in escalator context.
4. Document this as an intentional simplifying assumption.

Likely touchpoints:
- jps/multi_level_simulation.py
- decision/action_executor.py
- utils/speed_utils.py

Acceptance criteria:
- Agents never stall on escalators.
- Escalator speed behavior is simple to explain and deterministic under same inputs.
- Effective speeds stay within configured realism bounds.

### Workstream C: Transfer Pipeline Cleanup

Problem:
- Two-level hardcoded flipping and redundant/unused transfer logic reduce trust and extensibility.

Proposal:
1. Remove binary level flip assumptions and resolve destination via transfer mappings only.
2. Consolidate transfer checks into a single authoritative path.
3. Deprecate or remove unused transfer logic after parity checks.
4. Add transfer reason codes in logs/metrics (normal, deferred, blocked, cooldown-denied).

Likely touchpoints:
- jps/multi_level_simulation.py
- coordination/level_transfer_manager.py
- metrics/results_writer.py

Acceptance criteria:
- No hardcoded level flip remains in active transfer path.
- Exactly one transfer decision path is used at runtime.

### Workstream D: Exit Validation and Anomaly Handling

Problem:
- Failed exit validation can still produce successful-exit accounting.

Proposal:
1. Introduce anomaly outcome class separate from exited_agents.
2. Record anomaly details (agent, expected exit, last position, distance, level).
3. Expose anomaly counts and rates in final outputs.
4. Optionally fail scenario quality checks when anomaly rate exceeds threshold.
5. Add structured reason codes for forced reroutes and blocked-exit bounces.

Likely touchpoints:
- jps/exit_tracker.py
- metrics/analytics_generator.py
- metrics/results_writer.py

Acceptance criteria:
- Validation failures are measurable and never silently counted as clean exits.

### Workstream E: Movement Target Stability

Problem:
- Frequent journey/waypoint reconstruction can create non-physical motion churn and unnecessary LLM decision churn.

Proposal:
1. Keep journey commitment as default: agents continue current route unless significant new information appears.
2. Define project-specific significant-change thresholds that trigger re-decision, for example:
   - new event fired,
   - selected exit becomes blocked/inactive,
   - major local congestion shift,
   - transfer completion.
3. Reuse current journey when change is below threshold.
4. Reduce LLM call volume by gating decisions on significant change rather than frequent micro-updates.

Likely touchpoints:
- coordination/hybrid_simulation.py
- decision/prompt_cache.py
- decision/decision_processor.py
- decision/action_executor.py

Acceptance criteria:
- LLM call rate decreases without introducing obvious stale behavior.
- Agents maintain coherent route commitment in stable conditions.
- Re-decision behavior is explainable through explicit change triggers.

### Workstream F: Heuristic Constant Governance

Problem:
- Hardcoded thresholds are scattered and weakly justified.

Proposal:
1. Introduce one concise `behavior_calibration` config block for realism-sensitive thresholds.
2. Start with a minimal parameter set only (no broad preset system yet), for example:
   - escalator belt speed,
   - escalator max speed uplift,
   - transfer cooldown,
   - blocked-exit discovery radius,
   - significant-change thresholds for re-decision.
3. Document each parameter with plain-language purpose and expected effect.
4. Add optional profile presets later only if needed.

Likely touchpoints:
- config/config_loader.py
- all modules with behavioral constants
- docs update in risk register and architecture docs

Acceptance criteria:
- Constants used in realism logic are centrally discoverable and documented.
- A reviewer can understand and tune behavior from one config section without reading code internals.

## Recommended Execution Order

1. Workstream A (arrival de-clustering)
2. Workstream B (escalator behavior model)
3. Workstream E (significant-change decision gating and route commitment)
4. Workstream C (transfer pipeline cleanup)
5. Workstream D (exit anomaly accounting and reason-code logging)
6. Workstream F (constant governance)

Rationale:
- A and B directly affect your highest-priority realism concerns.
- E reduces unnecessary LLM calls while preserving explainability.
- C and D improve correctness/auditability under scrutiny.
- F hardens long-term maintainability.

## Validation Plan (Preliminary)

1. Visual validation
- Before/after replay comparisons for escalator-top behavior.
- Density heatmaps around transfer zones.

2. Behavioral metrics
- Transfer spread index (spatial variance at arrival zones).
- Escalator throughput and average escalator-context speed.
- Target churn rate per agent-minute.
- Exit anomaly rate.
- LLM calls per simulated minute before/after significant-change gating.

3. Explainability checks
- Each fallback/override emits a structured reason code.
- Final report includes counts by reason code.

## Out-of-Scope (This Pass)

- Full social force recalibration beyond escalator contexts.
- Large prompt/agent cognition redesign unrelated to identified risks.
- Venue-specific bespoke behavior scripts beyond calibration hooks.

## Open Decisions For Your Review

1. Exact definition of significant new information for re-decision gating.
2. Whether anomaly outcomes should hard-fail runs or only warn.
3. Whether grouped decision optimization remains enabled by default once significant-change gating is in place.

## Deliverable Strategy

If approved, implementation should be split into small, reviewable PR slices aligned to the workstreams above, each with:
- code changes,
- tests/validation artifacts,
- documentation updates,
- before/after behavior notes.

## Execution Artifact

Use [implementation-checklist-v1.md](implementation-checklist-v1.md) as the concrete implementation checklist for the first pass.
