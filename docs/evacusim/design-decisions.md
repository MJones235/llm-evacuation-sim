# Design Decisions and Tradeoffs

This document records major implementation choices inferred from code and their practical tradeoffs.

## Hybrid Timing Model

Decision:
- High-frequency physics updates, lower-frequency cognitive decisions.

Benefits:
- Realistic movement continuity.
- Lower LLM cost and API pressure.

Tradeoffs:
- Reactions are delayed without event-driven overrides.

Key references:
- [HybridSimulationRunner.run](function-reference/coordination__hybrid_simulation.md)
- [HybridSimulationRunner._should_make_decisions](function-reference/coordination__hybrid_simulation.md)

## Prompt Cache as First-Class Control

Decision:
- Semantic hashing gate before LLM call.

Benefits:
- Large call reduction in stable conditions.

Tradeoffs:
- Risk of stale behavior if state changes are not reflected in significant content extraction.

Key references:
- [PromptCache.should_call_llm](function-reference/decision__prompt_cache.md)
- [PromptCache._filter_observation](function-reference/decision__prompt_cache.md)

## Multi-Level Simulation by Transfer, Not Continuous Vertical Physics

Decision:
- Separate simulation instance per level and transfer agents across escalator zones.

Benefits:
- Practical decomposition of geometry and routing concerns.

Tradeoffs:
- Transfer appears discrete and can reduce vertical-motion realism.

Key references:
- [MultiLevelJuPedSimulation._do_level_transfer](function-reference/jps__multi_level_simulation.md)
- [LevelTransferManager._build_transfer_mappings](function-reference/coordination__level_transfer_manager.md)

## Geometry Blocking as Primary Exit Restriction

Decision:
- Blocked exits are enforced by geometry obstacles.

Benefits:
- Strong physical consistency with route planning.

Tradeoffs:
- Runtime geometry mutation complexity and potential reversibility constraints.

Key references:
- [ConcordiaJuPedSimulation.add_geometry_obstacle_for_exit](function-reference/jps__jupedsim_integration.md)
- [GeometryManager.add_obstacle_polygon](function-reference/jps__geometry_manager.md)

## Grouped Decision Processing

Decision:
- Staggered groups and controlled parallelism.

Benefits:
- Better throughput and reduced burst load.

Tradeoffs:
- Some agents may have stale decisions between group windows.

Key references:
- [DecisionProcessor.process_all_agents](function-reference/decision__decision_processor.md)
- [HybridSimulationRunner.run](function-reference/coordination__hybrid_simulation.md)

## Minimal Concordia Memory Stack

Decision:
- Excludes higher-cost memory-retrieval components in favor of low-overhead components.

Benefits:
- Lower decision latency and fewer model calls.

Tradeoffs:
- Reduced deep contextual recall.

Key references:
- [EvacuationAgent.build](function-reference/concordia__evacuation_agent.md)
