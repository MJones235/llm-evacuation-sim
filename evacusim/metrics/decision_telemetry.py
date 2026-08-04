"""Decision telemetry rollups for simulation output files."""

from __future__ import annotations

from typing import Any


def build_decision_telemetry(
    agent_decisions: dict[str, Any],
    wait_events: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Aggregate decision diagnostics for easier run-level analysis."""
    action_counts: dict[str, int] = {}
    pace_counts: dict[str, int] = {}
    reassess_mode_counts: dict[str, int] = {}
    repair_status_counts: dict[str, int] = {}
    cue_type_counts: dict[str, int] = {}

    decision_records_total = 0
    wait_decisions = 0
    llm_called_count = 0
    fallback_count = 0

    for agent_data in agent_decisions.values():
        if not isinstance(agent_data, dict):
            continue
        decisions = agent_data.get("decisions", [])
        if not isinstance(decisions, list):
            continue

        for record in decisions:
            if not isinstance(record, dict):
                continue
            decision_records_total += 1

            payload = record.get("decision_payload", {})
            if isinstance(payload, dict):
                action = payload.get("action")
                if isinstance(action, str):
                    action_counts[action] = action_counts.get(action, 0) + 1
                    if action == "wait":
                        wait_decisions += 1

                pace = payload.get("pace")
                if isinstance(pace, str):
                    pace_counts[pace] = pace_counts.get(pace, 0) + 1

                reassess_when = payload.get("reassess_when")
                if isinstance(reassess_when, str):
                    reassess_mode_counts[reassess_when] = (
                        reassess_mode_counts.get(reassess_when, 0) + 1
                    )

            repair_status = record.get("repair_status")
            if isinstance(repair_status, str):
                repair_status_counts[repair_status] = repair_status_counts.get(repair_status, 0) + 1
                if repair_status.startswith("fallback"):
                    fallback_count += 1

            prompt_marker = record.get("prompt")
            if isinstance(prompt_marker, str) and prompt_marker != "cached":
                llm_called_count += 1

            cue_types = record.get("cue_types")
            if isinstance(cue_types, list):
                for cue in cue_types:
                    if isinstance(cue, str):
                        cue_type_counts[cue] = cue_type_counts.get(cue, 0) + 1

    non_wait_decisions = max(0, decision_records_total - wait_decisions)

    return {
        "decision_records_total": decision_records_total,
        "wait_decisions": wait_decisions,
        "wait_decision_rate": (wait_decisions / decision_records_total)
        if decision_records_total
        else 0.0,
        "non_wait_decisions": non_wait_decisions,
        "llm_called_count": llm_called_count,
        "llm_call_rate": (llm_called_count / decision_records_total) if decision_records_total else 0.0,
        "fallback_count": fallback_count,
        "fallback_rate": (fallback_count / decision_records_total) if decision_records_total else 0.0,
        "action_counts": action_counts,
        "pace_counts": pace_counts,
        "reassess_mode_counts": reassess_mode_counts,
        "repair_status_counts": repair_status_counts,
        "cue_type_counts": cue_type_counts,
        "wait_events_recorded": len(wait_events) if wait_events is not None else None,
    }
