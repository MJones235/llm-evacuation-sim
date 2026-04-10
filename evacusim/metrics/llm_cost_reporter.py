"""
Financial reporting for LLM usage and costs.

Generates reports showing:
- Token usage (prompt, completion, total)
- Cost breakdown by input/output
- Per-agent averages
- Total costs in GBP
"""

from typing import Any

from evacusim.utils.logger import get_logger

logger = get_logger(__name__)


class FinancialReporter:
    """
    Generates financial reports from LLM usage statistics.

    This class is responsible for formatting and presenting LLM usage
    costs in a human-readable format, including per-agent breakdowns.
    """

    @staticmethod
    def generate_report(llm_provider: Any, num_agents: int) -> str:
        """
        Generate a financial report from LLM usage statistics.

        Args:
            llm_provider: LLM provider instance with get_usage_stats() method
            num_agents: Total number of agents in the simulation

        Returns:
            Formatted financial report string
        """
        if not llm_provider or not hasattr(llm_provider, "get_usage_stats"):
            return "\n=== FINANCIAL REPORT ===\nLLM provider usage stats not available\n"

        try:
            stats = llm_provider.get_usage_stats()

            lines = []
            lines.append("\n=== FINANCIAL REPORT ===")
            lines.append("\nLLM Token Usage:")
            lines.append(f"  Prompt tokens:      {stats['prompt_tokens']:,}")
            lines.append(f"  Completion tokens:  {stats['completion_tokens']:,}")
            lines.append(f"  Total tokens:       {stats['total_tokens']:,}")
            lines.append(f"  Total requests:     {stats['total_requests']:,}")
            lines.append("\nCost Breakdown (£):")
            lines.append(f"  Input cost:         £{stats['input_cost_gbp']:.4f}")
            lines.append(f"  Output cost:        £{stats['output_cost_gbp']:.4f}")
            lines.append(f"  TOTAL COST:         £{stats['estimated_cost_gbp']:.4f}")
            lines.append("\nPer-Agent Averages:")
            if num_agents > 0:
                lines.append(f"  Tokens per agent:   {stats['total_tokens'] / num_agents:.0f}")
                lines.append(
                    f"  Cost per agent:     £{stats['estimated_cost_gbp'] / num_agents:.4f}"
                )
                lines.append(f"  Requests per agent: {stats['total_requests'] / num_agents:.1f}")
            lines.append("\n" + "=" * 40)

            return "\n".join(lines)
        except Exception as e:
            logger.warning(f"Failed to generate financial report: {e}")
            return f"\n=== FINANCIAL REPORT ===\nError generating report: {e}\n"
