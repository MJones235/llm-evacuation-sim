"""
Results writing for Station Concordia simulations.

Handles saving simulation results to JSON files, both incrementally
during simulation (for live viewing) and final results at completion.
"""

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from evacusim.metrics.analytics_generator import AnalyticsGenerator
from evacusim.metrics.decision_telemetry import build_decision_telemetry
from evacusim.metrics.llm_cost_reporter import FinancialReporter
from evacusim.utils.logger import get_logger

logger = get_logger(__name__)


DECISIONS_COLUMNS = [
    "time_s",
    "agent_id",
    "stage",
    "action",
    "wait_reason",
    "exit_id",
    "pace",
    "reassess_when",
    "zone",
    "cues",
    "repair_status",
    "llm_called",
    "prompt_hash",
    "route_changed_to",
    "rationale",
]
"""Columns of decisions.csv (documented in docs/outputs.md)."""


@dataclass
class RunRecord:
    """Everything a finished (or interrupted) run writes out.

    Built by :meth:`HybridSimulationRunner.run_record`.
    """

    agent_decisions: dict[str, Any]
    agent_positions: dict[str, tuple[float, float]]
    final_sim_time: float
    event_history: list[dict[str, Any]]
    blocked_exits: set[str]
    message_history: list[dict[str, Any]]
    wait_events: list[dict[str, Any]]
    decision_interval: float
    max_steps: int
    num_agents: int
    performance_report: str
    llm_provider: Any
    agent_levels: dict[str, str] | None = None
    agent_roles: dict[str, str] | None = None
    exit_log: list[dict[str, Any]] | None = None
    spawn_log: list[dict[str, Any]] | None = None
    escalator_system: Any = None


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
            "blocked_exits": sorted(blocked_exits),
            "messages": message_history,
            "decision_telemetry": build_decision_telemetry(agent_decisions),
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
    def save_positions_only(
        output_file: Path,
        agent_positions: dict[str, tuple[float, float]],
        current_sim_time: float,
        agent_levels: dict[str, str] | None = None,
        blocked_exits: set[str] | None = None,
        agent_roles: dict[str, str] | None = None,
        active_train_exits: set[str] | None = None,
    ) -> None:
        """
        Write a lightweight positions sidecar file for the live viewer.

        Only serialises positions, levels, time and blocked exits — far smaller
        than the full incremental write — so it can be called every 0.5 s without
        meaningful I/O cost.  The file is named ``<stem>_positions.json`` alongside
        the main output file.

        Args:
            output_file: Path to the main output JSON file (used to derive the
                sidecar path).
            agent_positions: Current agent positions.
            current_sim_time: Current simulation time.
            agent_levels: Per-agent level (multi-level simulations).
            blocked_exits: Currently blocked exits.
            agent_roles: Optional dict of agent_id → role label for director agents.
        """
        if not output_file:
            return

        sidecar = output_file.with_name(output_file.stem + "_positions.json")
        payload: dict[str, Any] = {
            "agent_positions": agent_positions,
            "current_time": current_sim_time,
        }
        if agent_levels:
            payload["agent_levels"] = agent_levels
        if blocked_exits is not None:
            payload["blocked_exits"] = sorted(blocked_exits)
        if agent_roles:
            payload["agent_roles"] = agent_roles
        if active_train_exits is not None:
            payload["active_train_exits"] = sorted(active_train_exits)

        try:
            output_file.parent.mkdir(parents=True, exist_ok=True)
            tmp_file = sidecar.with_suffix(sidecar.suffix + ".tmp")
            with open(tmp_file, "w") as f:
                json.dump(payload, f)
            tmp_file.replace(sidecar)
        except Exception as e:
            logger.warning(f"Failed to save position sidecar: {e}")

    @staticmethod
    def _save_exit_log(
        path: Path,
        exit_log: list[dict[str, Any]],
        spawn_log: list[dict[str, Any]] | None,
    ) -> None:
        """Write exit_log.csv, enriched with each agent's spawn provenance."""
        spawn_by_id = {entry["id"]: entry for entry in (spawn_log or [])}
        fields = [
            "agent_id",
            "exit_name",
            "intended_exit",
            "exit_distance_m",
            "time_s",
            "level",
            "x",
            "y",
            "validated",
            "spawn_source",
            "spawn_location",
            "spawn_time_s",
        ]
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for record in exit_log:
                spawn = spawn_by_id.get(record["agent_id"], {})
                row = dict(record)
                row["spawn_source"] = spawn.get("source", "")
                row["spawn_location"] = spawn.get("location", "")
                row["spawn_time_s"] = spawn.get("time_s", "")
                writer.writerow({k: row.get(k, "") for k in fields})
        logger.info(f"Exit log ({len(exit_log)} records) saved to {path}")

    @staticmethod
    def _save_decisions_csv(path: Path, agent_decisions: dict[str, Any]) -> None:
        """Write decisions.csv: one row per decision, in time order."""
        rows = []
        for agent_id, data in agent_decisions.items():
            for d in data.get("decisions", []):
                payload = d.get("decision_payload") or {}
                assessment = payload.get("assessment") or {}
                route_change = d.get("route_change") or {}
                rows.append(
                    {
                        "time_s": d.get("time"),
                        "agent_id": agent_id,
                        "stage": d.get("stage") or "",
                        "action": payload.get("action", ""),
                        "wait_reason": payload.get("wait_reason") or "",
                        "exit_id": payload.get("exit_id") or "",
                        "pace": payload.get("pace") or "",
                        "reassess_when": payload.get("reassess_when") or "",
                        "zone": d.get("zone") or "",
                        "cues": "|".join(d.get("cue_types") or []),
                        "repair_status": d.get("repair_status", ""),
                        "llm_called": d.get("prompt") not in (None, "cached"),
                        "prompt_hash": d.get("prompt_hash") or "",
                        "route_changed_to": route_change.get("to_exit", ""),
                        "rationale": assessment.get("option_chosen_because", "")
                        if isinstance(assessment, dict)
                        else "",
                    }
                )
        rows.sort(key=lambda r: (r["time_s"] or 0.0, r["agent_id"]))
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=DECISIONS_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
        logger.info(f"Decisions ({len(rows)}) saved to {path}")

    @staticmethod
    def _save_escalator_log(path: Path, ride_log: list[dict[str, Any]]) -> None:
        """Write escalator_log.csv: one row per completed ride."""
        fields = [
            "agent_id",
            "escalator",
            "direction",
            "lane",
            "chose_s",
            "queue_join_s",
            "board_s",
            "alight_s",
            "ride_s",
            "stall_wait_s",
            "discharge_attempts",
        ]
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for record in ride_log:
                writer.writerow({k: record.get(k, "") for k in fields})
        logger.info(f"Escalator log ({len(ride_log)} rides) saved to {path}")

    @staticmethod
    def save_final_results(output_path: Path, record: RunRecord) -> None:
        """Write a run's final results and reports next to ``output_path``.

        Writes ``agent_decisions.json`` (``output_path``), ``decisions.csv``,
        ``performance_report.txt``, ``financial_report.txt``, ``exit_log.csv``,
        ``escalator_log.csv`` / ``escalators.json`` and the analytics text files.
        """
        agent_decisions = record.agent_decisions
        wait_events = record.wait_events
        message_history = record.message_history
        num_agents = record.num_agents

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
            "agent_positions": record.agent_positions,
            "final_time": record.final_sim_time,
            "events": record.event_history,
            "blocked_exits": sorted(record.blocked_exits),
            "route_changes": route_changes,
            "messages": message_history,
            "decision_telemetry": build_decision_telemetry(agent_decisions, wait_events),
            "config": {
                "decision_interval": record.decision_interval,
                "max_steps": record.max_steps,
                "num_agents": num_agents,
            },
        }

        # Add agent levels and roles for multi-level visualization
        if record.agent_levels:
            results["agent_levels"] = record.agent_levels
        if record.agent_roles:
            results["agent_roles"] = record.agent_roles

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {output_path}")
        ResultsWriter._save_decisions_csv(output_path.parent / "decisions.csv", agent_decisions)

        # Save performance report
        perf_report_path = output_path.parent / "performance_report.txt"
        with open(perf_report_path, "w") as f:
            f.write(record.performance_report)
        logger.info(f"Performance report saved to {perf_report_path}")

        # Save financial report
        financial_report_path = output_path.parent / "financial_report.txt"
        financial_report = FinancialReporter.generate_report(record.llm_provider, num_agents)
        with open(financial_report_path, "w") as f:
            f.write(financial_report)
        logger.info(f"Financial report saved to {financial_report_path}")

        # Per-agent exit log, joined with spawn provenance so each row says
        # where the person came from as well as where they left.
        if record.exit_log is not None:
            ResultsWriter._save_exit_log(
                output_path.parent / "exit_log.csv", record.exit_log, record.spawn_log
            )

        # One row per completed escalator ride, plus static escalator
        # geometry so plots can draw the conveyors on their own axes.
        escalators = record.escalator_system
        if escalators is not None:
            ResultsWriter._save_escalator_log(
                output_path.parent / "escalator_log.csv", escalators.ride_log
            )
            with open(output_path.parent / "escalators.json", "w") as f:
                json.dump(escalators.geometry(), f, indent=2)

        # Save all analytics
        AnalyticsGenerator.save_all_analytics(
            output_path,
            route_changes,
            wait_events,
            message_history,
        )
