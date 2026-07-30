# Function Reference: evacusim/metrics/llm_cost_reporter.py

- Source file: [evacusim/metrics/llm_cost_reporter.py](../../../evacusim/metrics/llm_cost_reporter.py)
- Top-level classes: 1
- Top-level functions: 0

## Module Summary

Financial reporting for LLM usage and costs.

## Classes and Methods

### Class FinancialReporter

- Location: [evacusim/metrics/llm_cost_reporter.py#L18](../../../evacusim/metrics/llm_cost_reporter.py#L18)
- Summary: Generates financial reports from LLM usage statistics.
- Method count: 1

#### generate_report

- Signature: def generate_report(llm_provider, num_agents) -> str
- Location: [evacusim/metrics/llm_cost_reporter.py#L27](../../../evacusim/metrics/llm_cost_reporter.py#L27)
- Summary: Generate a financial report from LLM usage statistics.
- Key calls: llm_provider.get_usage_stats, lines.append, join, hasattr, logger.warning

## Coverage Notes

- Nested helper functions documented in this file: 0
