"""
Results writing for Station Concordia simulations.

Handles saving simulation results to JSON files, both incrementally
during simulation (for live viewing) and final results at completion.
"""

import json
from pathlib import Path
from typing import Any

from evacusim.utils.logger import get_logger
from evacusim.metrics.analytics_generator import AnalyticsGenerator
from evacusim.metrics.llm_cost_reporter import FinancialReporter

logger = get_logger(__name__)


class ResultsWriter:
    """
    Handles writing simulation results to files.

    Manages both incremental saves (for live viewing) and final results
    with all reports and analytics.
    """

    @staticmethod
    def save_incremental(
        output_file: Path,
        agent_decisions: dict[str, Any],
        agent_positions: dict[str, tuple[float, float]],
        current_sim_time: float,
        event_history: list[dict[str, Any]],
        blocked_exits: set[str],
        message_history: list[dict[str, Any]],
        decision_interval: float,
        max_steps: int,
        num_agents: int,
        agent_levels: dict[str, str] | None = None,
    ) -> None:
        """
        Save current results incrementally for live viewing.

        Args:
            output_file: Path to output JSON file
            agent_decisions: Agent decision history
            agent_positions: Current agent positions
            current_sim_time: Current simulation time
            event_history: All events that occurred
            blocked_exits: Set of currently blocked exits
            message_history: All messages sent
            decision_interval: Time between decisions
            max_steps: Maximum simulation steps
            num_agents: Number of agents
            agent_levels: Current level for each agent (multi-level simulations)
        """
        if not output_file:
            return

        results = {
            "agent_decisions": agent_decisions,
            "agent_positions": agent_positions,
            "current_time": current_sim_time,
            "events": event_history,
            "blocked_exits": list(blocked_exits),
            "messages": message_history,
            "config": {
                "decision_interval": decision_interval,
                "max_steps": max_steps,
                "num_agents": num_agents,
            },
        }

        # Add agent levels for multi-level visualization
        if agent_levels:
            results["agent_levels"] = agent_levels

        try:
            output_file.parent.mkdir(parents=True, exist_ok=True)
            tmp_file = output_file.with_suffix(output_file.suffix + ".tmp")
            with open(tmp_file, "w") as f:
                json.dump(results, f, indent=2)
            tmp_file.replace(output_file)
        except Exception as e:
            logger.warning(f"Failed to save incremental results: {e}")

    @staticmethod
    def save_final_results(
        output_path: Path,
        agent_decisions: dict[str, Any],
        agent_positions: dict[str, tuple[float, float]],
        final_sim_time: float,
        event_history: list[dict[str, Any]],
        blocked_exits: set[str],
        message_history: list[dict[str, Any]],
        wait_events: list[dict[str, Any]],
        decision_interval: float,
        max_steps: int,
        num_agents: int,
        performance_report: str,
        llm_provider: Any,
        agent_levels: dict[str, str] | None = None,
    ) -> None:
        """
        Save final simulation results with all reports.

        Args:
            output_path: Path to main results JSON file
            agent_decisions: Complete agent decision history
            agent_positions: Final agent positions
            final_sim_time: Final simulation time
            event_history: All events that occurred
            blocked_exits: Set of blocked exits
            message_history: All messages sent
            wait_events: All waiting behavior events
            decision_interval: Time between decisions
            max_steps: Maximum simulation steps
            num_agents: Number of agents
            performance_report: Performance timing report
            llm_provider: LLM provider instance for cost tracking
            agent_levels: Final level for each agent (multi-level simulations)
        """
        # Extract route changes for analytics
        route_changes = []
        for agent_id, data in agent_decisions.items():
            for decision in data.get("decisions", []):
                if "route_change" in decision:
                    route_changes.append(
                        {
                            "agent": agent_id,
                            "time": decision["time"],
                            "from_exit": decision["route_change"]["from_exit"],
                            "to_exit": decision["route_change"]["to_exit"],
                            "reason": decision["route_change"]["reason"],
                        }
                    )

        # Main results JSON
        results = {
            "agent_decisions": agent_decisions,
            "agent_positions": agent_positions,
            "final_time": final_sim_time,
            "events": event_history,
            "blocked_exits": list(blocked_exits),
            "route_changes": route_changes,
            "messages": message_history,
            "config": {
                "decision_interval": decision_interval,
                "max_steps": max_steps,
                "num_agents": num_agents,
            },
        }

        # Add agent levels for multi-level visualization
        if agent_levels:
            results["agent_levels"] = agent_levels

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {output_path}")

        # Save performance report
        perf_report_path = output_path.parent / "performance_report.txt"
        with open(perf_report_path, "w") as f:
            f.write(performance_report)
        logger.info(f"Performance report saved to {perf_report_path}")

        # Save financial report
        financial_report_path = output_path.parent / "financial_report.txt"
        financial_report = FinancialReporter.generate_report(llm_provider, num_agents)
        with open(financial_report_path, "w") as f:
            f.write(financial_report)
        logger.info(f"Financial report saved to {financial_report_path}")

        # Save all analytics
        AnalyticsGenerator.save_all_analytics(
            output_path,
            route_changes,
            wait_events,
            message_history,
        )
