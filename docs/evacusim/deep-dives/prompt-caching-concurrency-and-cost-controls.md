# Deep Dive: Prompt Caching, Concurrency, and Cost Controls

## Why This Is Core

evacusim's scalability and economic viability depend on how often and how many concurrent LLM calls are made.

Key components:

- [PromptCache.should_call_llm](../function-reference/decision__prompt_cache.md)
- [DecisionProcessor.process_all_agents](../function-reference/decision__decision_processor.md)
- [AzureLLMConcordia.sample_text](../function-reference/concordia__azure_llm_concordia.md)
- [FinancialReporter.generate_report](../function-reference/metrics__llm_cost_reporter.md)

## Prompt Cache Strategy

The cache avoids repeat calls when semantically meaningful context has not changed.

Core mechanics include:

- observation filtering and normalization,
- significant-content extraction,
- stable hash comparisons,
- cached decision reuse on hits.

Because the full rendered action-spec text is part of the cache key input,
editing the external decision prompt template invalidates prior cache hits for
affected agents and forces fresh LLM calls until a new steady state is reached.

The Azure system prompt is externalized too, but it is not part of the
PromptCache hash gate. Changing the system prompt influences subsequent LLM
calls, while cache-hit decisions continue to reuse their previously cached
actions until another cache-miss trigger occurs.

This strategy is one of the most important design choices for cost control.

## Concurrency Model

Decision processing uses bounded parallelism with asynchronous orchestration and thread-based model calls.

Control points include:

- semaphore limits for in-flight LLM calls,
- per-agent timeouts,
- staggered decision groups,
- immediate-decision overrides for event/transfer edge cases.

## Cost Tracking and Reporting

Provider usage is tracked and summarized to make runs auditable.

Relevant references:

- [AzureLLMConcordia.get_usage_stats](../function-reference/concordia__azure_llm_concordia.md)
- [FinancialReporter.generate_report](../function-reference/metrics__llm_cost_reporter.md)

## Design Tradeoffs

- Aggressive caching reduces cost but risks stale behavior.
- Higher parallelism improves throughput but can increase API pressure and race complexity.
- Grouped decisions reduce load but may reduce per-agent nuance.

## Review Guidance

1. Validate that cache-significant content includes all behavior-changing signals.
2. Tune semaphore and timeout values against provider limits.
3. Verify fairness for agents repeatedly queued for immediate decisions.
4. Compare behavior quality with cache and grouping toggled.
