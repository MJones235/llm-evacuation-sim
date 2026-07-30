# Risk Register and Improvement Targets

This register highlights potential issues identified from implementation review.

## High Priority

1. Runtime obstacle blocks appear one-way
- Risk: geometry mutation path does not clearly expose unblock/reversal lifecycle.
- References:
  - [GeometryManager.add_obstacle_polygon](function-reference/jps__geometry_manager.md)
  - [EventManager.block_exit](function-reference/systems__event_manager.md)
- Improvement: preserve baseline geometry and support reversible block operations.

2. Exit validation tolerance can mask edge failures
- Risk: permissive success handling in exit validation may hide route/config defects.
- References:
  - [ExitTracker.check_exited_agents](function-reference/jps__exit_tracker.md)
  - [ExitTracker._validate_exit_reached](function-reference/jps__exit_tracker.md)
- Improvement: tighten policy and emit stronger diagnostics on validation failure.

3. Shared mutable state touched by concurrent paths
- Risk: dictionaries and sets updated during parallel decision processing can race depending on execution path.
- References:
  - [DecisionProcessor.process_all_agents](function-reference/decision__decision_processor.md)
  - [ActionExecutor.execute_action](function-reference/decision__action_executor.md)
- Improvement: formalize lock boundaries around shared writes.

## Medium Priority

4. Platform-side visibility mappings are topology-specific
- Risk: hardcoded side mappings reduce portability to new station layouts.
- References:
  - [SpatialAnalyzer._belongs_to_platform_side](function-reference/translation__spatial_analyzer.md)
- Improvement: move mappings into config schema.

5. Escalator and LOS assumptions depend on naming conventions
- Risk: geometry naming drift can silently degrade behavior.
- References:
  - [LevelTransferManager._load_escalator_zones](function-reference/coordination__level_transfer_manager.md)
  - [SpatialAnalyzer.get_visible_exits](function-reference/translation__spatial_analyzer.md)
- Improvement: add startup validation for expected naming patterns.

6. Group approximations may over-generalize behavior
- Risk: grouped decision optimization can suppress per-agent nuance in edge cases.
- References:
  - [DecisionProcessor.process_all_agents](function-reference/decision__decision_processor.md)
- Improvement: apply stricter eligibility criteria for group-follow behavior.

## Low Priority

7. Hardcoded operational constants
- Risk: fixed write intervals and heuristic thresholds may not fit all scenarios.
- References:
  - [HybridSimulationRunner.run](function-reference/coordination__hybrid_simulation.md)
- Improvement: centralize into tunable config entries.

8. Incomplete injury randomization hook
- Risk: injury assignment helper can be underutilized if configuration expectations differ.
- References:
  - [AgentFactory._determine_injured_agents](function-reference/setup__agent_factory.md)
- Improvement: explicitly expose and validate injury-rate config path.

## Additional Findings (Realism-Focused Review, 2026-07-30)

### High Priority

9. Transfer arrival waypoint funneling causes artificial clustering
- Risk: transferred agents are assigned a shared temporary waypoint per level, producing non-human convergence at escalator tops.
- References:
  - [MultiLevelJuPedSimulation._transfer_agent_through_escalator](function-reference/jps__multi_level_simulation.md)
  - [ConcordiaJuPedSimulation.set_agent_target](function-reference/jps__jupedsim_integration.md)
- Improvement: replace single-point arrival steering with sampled egress bands and short-horizon local targets.

10. Escalator speed semantics are underspecified for standing vs walking behavior
- Risk: current logic enforces only a minimum speed floor; no explicit model separates standing-on-belt from walking-on-belt behavior.
- References:
  - [MultiLevelJuPedSimulation._enforce_escalator_constraints](function-reference/jps__multi_level_simulation.md)
  - [ActionExecutor.execute_action](function-reference/decision__action_executor.md)
- Improvement: implement explicit escalator behavior states and compose effective speed from belt speed plus voluntary walking contribution.

11. Waypoint/journey churn may inject non-physical stop-start artifacts
- Risk: repeated waypoint stage and journey creation on frequent target updates can produce behavior jitter and higher simulation overhead.
- References:
  - [ConcordiaJuPedSimulation.set_agent_target](function-reference/jps__jupedsim_integration.md)
  - [ActionExecutor._handle_wait_action](function-reference/decision__action_executor.md)
- Improvement: introduce target-change hysteresis and reuse persistent movement goals where possible.

12. Exit disappearance validation still fails open on suspicious removals
- Risk: agents that fail exit proximity validation are still marked exited, masking geometry/routing defects.
- References:
  - [ExitTracker.check_exited_agents](function-reference/jps__exit_tracker.md)
  - [ExitTracker._validate_exit_reached](function-reference/jps__exit_tracker.md)
- Improvement: route failed validations to anomaly state and metrics instead of immediate successful evacuation.

### Medium Priority

13. Multi-level transfer routing is hardcoded to two-level flipping
- Risk: target level is derived by binary flip logic, reducing explainability and extensibility for stations with more than two levels.
- References:
  - [MultiLevelJuPedSimulation._transfer_agent_through_escalator](function-reference/jps__multi_level_simulation.md)
  - [LevelTransferManager._build_transfer_mappings](function-reference/coordination__level_transfer_manager.md)
- Improvement: route target level strictly via transfer mappings and remove hardcoded level assumptions.

14. Dead/parallel transfer logic paths increase maintenance ambiguity
- Risk: unused transfer-check pathways indicate implementation drift and unclear source of truth for transfers.
- References:
  - [LevelTransferManager.check_transfer](function-reference/coordination__level_transfer_manager.md)
  - [MultiLevelJuPedSimulation._process_escalator_exits](function-reference/jps__multi_level_simulation.md)
- Improvement: consolidate to one authoritative transfer pipeline and deprecate or remove unused path.

15. Translation safety-net redirects indicate brittle action semantics
- Risk: platform-zone and cross-level redirect heuristics suggest ontology/validation gaps and increase hidden behavior overrides.
- References:
  - [ActionTranslator.translate](function-reference/translation__action_translator.md)
  - [ActionTranslator._redirect_to_next_level_exit](function-reference/translation__action_translator.md)
- Improvement: tighten action schema and level-aware validation before execution, reducing fallback redirect hacks.

16. Hardcoded realism thresholds remain distributed and weakly calibrated
- Risk: discovery radii, transfer radii, cooldowns, and spacing constants are spread across modules and difficult to justify empirically.
- References:
  - [SpatialAnalyzer.get_visible_exits](function-reference/translation__spatial_analyzer.md)
  - [MultiLevelJuPedSimulation._transfer_agent_through_escalator](function-reference/jps__multi_level_simulation.md)
  - [LevelTransferManager.check_transfer](function-reference/coordination__level_transfer_manager.md)
- Improvement: centralize constants under calibrated config with scenario-specific profiles and rationale metadata.
