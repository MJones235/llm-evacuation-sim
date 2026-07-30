# Deep Dive: Outputs, Analytics, and Observability

## Output Layer Role

The output layer turns runtime state into artifacts for replay, analysis, and cost/accountability review.

Primary components:

- [ResultsWriter.save_incremental](../function-reference/metrics__results_writer.md)
- [ResultsWriter.save_positions_only](../function-reference/metrics__results_writer.md)
- [ResultsWriter.save_final_results](../function-reference/metrics__results_writer.md)
- [AnalyticsGenerator.save_all_analytics](../function-reference/metrics__analytics_generator.md)
- [PopulationMonitor.save](../function-reference/metrics__population_monitor.md)

## Artifact Types

Typical generated artifacts include:

- full decision/state JSON,
- lightweight positions sidecar for live viewers,
- position history JSONL for video generation,
- route/wait/messaging analytics text outputs,
- population time-series outputs,
- LLM token/cost summaries.

## Observability Value

These artifacts support:

- post-run behavioral audits,
- policy comparison across scenario variants,
- debugging decision/action anomalies,
- performance and cost tuning.

## Timing and I/O Strategy

The runner uses periodic and end-of-run writes to avoid heavy per-step output overhead.

Coordination references:

- [HybridSimulationRunner.run](../function-reference/coordination__hybrid_simulation.md)
- [PositionHistoryTracker](../function-reference/visualization__position_history.md)

## Design Tradeoffs

- Frequent incremental writes improve recoverability and live introspection.
- They can increase I/O overhead and contention on slower disks.
- Rich outputs improve analysis depth but increase storage footprint.

## Review Guidance

1. Confirm write frequencies are configurable for workload size.
2. Validate atomic-write behavior under interruption.
3. Check consistency between final full output and sidecar history.
4. Ensure analytics assumptions match current decision/action schema.
