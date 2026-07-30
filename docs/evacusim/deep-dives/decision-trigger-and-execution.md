# Deep Dive: Decision Trigger and Execution Pipeline

## Triggering Cadence

Core trigger checks:
- [HybridSimulationRunner._should_make_decisions](../function-reference/coordination__hybrid_simulation.md)
- [HybridSimulationRunner.run](../function-reference/coordination__hybrid_simulation.md)

Mechanisms:
- Interval-based decisions (decision_interval).
- Event-driven overrides to force immediate updates.
- Staggered decision groups to reduce burst LLM traffic.
- Immediate-decision queue for transferred or rerouted agents.

## Observation and Prompt Construction

- [ObservationCoordinator.generate_all_observations](../function-reference/coordination__observation_coordinator.md)
- [DecisionProcessor._build_zones_section](../function-reference/decision__decision_processor.md)
- [DecisionProcessor._get_valid_exits_section](../function-reference/decision__decision_processor.md)
- [DecisionProcessor._get_following_constraint_text](../function-reference/decision__decision_processor.md)

## Cache Gate and LLM Invocation

- [PromptCache.should_call_llm](../function-reference/decision__prompt_cache.md)
- [DecisionProcessor.process_all_agents](../function-reference/decision__decision_processor.md)

Pipeline:

1. Build observation and action spec.
2. Check semantic cache hash.
3. If miss, execute agent action with controlled parallelism and timeout.
4. Parse action JSON and feed message/action pipelines.

## Action Translation and Execution

- [ActionTranslator.translate](../function-reference/translation__action_translator.md)
- [ActionExecutor.execute_action](../function-reference/decision__action_executor.md)

Execution responsibilities:
- Convert abstract exits/zones to coordinates.
- Apply movement targets and speed controls.
- Resolve follow-target behavior robustly.

## Messaging Coupling

- [MessageSystem.extract_and_deliver_message](../function-reference/systems__messaging__message_system.md)

Messages extracted from action JSON are rate-limited and delivered before final motion execution records are committed.

## Design Implications

- The pipeline balances realism and cost using caching, grouping, and selective immediate overrides.
- Correctness depends on robust JSON action parsing and deterministic downstream translation.
