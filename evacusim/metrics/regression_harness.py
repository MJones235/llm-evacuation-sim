"""Seeded run regression harness for evacuation-quality KPIs.

This module compares two simulation outputs (baseline vs candidate) and reports
behavioral deltas that are sensitive to decision-layer changes.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Any


@dataclass
class KpiSummary:
    """Comparable KPI snapshot derived from a single run output."""

    final_time_s: float | None
    decisions_total: int
    wait_decisions: int
    wait_decision_rate: float
    wait_loop_rate: float
    fallback_rate: float
    new_cue_only_rate: float
    route_change_count: int
    action_mix: dict[str, int]
    pace_mix: dict[str, int]
    repair_status_counts: dict[str, int]
    max_zone_occupancy: dict[str, int]


def _safe_ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator) / float(denominator)


def _load_json(path: Path) -> dict[str, Any]:
    with open(path) as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected object JSON in {path}")
    return data


def _load_population_timeseries(results_path: Path, results: dict[str, Any]) -> dict[str, Any] | None:
    embedded = results.get("population_timeseries")
    if isinstance(embedded, dict):
        return embedded

    sidecar = results_path.parent / "population_timeseries.json"
    if sidecar.exists():
        data = _load_json(sidecar)
        if isinstance(data, dict):
            return data
    return None


def _iter_decision_records(results: dict[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    decisions = results.get("agent_decisions", {})
    if not isinstance(decisions, dict):
        return records

    for agent_data in decisions.values():
        if not isinstance(agent_data, dict):
            continue
        for record in agent_data.get("decisions", []):
            if isinstance(record, dict):
                records.append(record)
    return records


def _compute_wait_loop_rate(results: dict[str, Any]) -> float:
    decisions = results.get("agent_decisions", {})
    if not isinstance(decisions, dict):
        return 0.0

    loop_flags: list[int] = []
    for agent_data in decisions.values():
        if not isinstance(agent_data, dict):
            continue
        agent_decisions = agent_data.get("decisions", [])
        if not isinstance(agent_decisions, list):
            continue

        previous_wait = False
        for record in agent_decisions:
            if not isinstance(record, dict):
                continue
            payload = record.get("decision_payload", {})
            action = payload.get("action") if isinstance(payload, dict) else None
            current_wait = action == "wait"
            if current_wait and previous_wait:
                loop_flags.append(1)
            else:
                loop_flags.append(0)
            previous_wait = current_wait

    if not loop_flags:
        return 0.0
    return mean(loop_flags)


def _compute_zone_maximums(population: dict[str, Any] | None) -> dict[str, int]:
    if not isinstance(population, dict):
        return {}

    snapshots = population.get("snapshots", [])
    if not isinstance(snapshots, list) or not snapshots:
        return {}

    ignored_keys = {"sim_time_s", "sim_time_min"}
    zone_max: dict[str, int] = {}
    for snap in snapshots:
        if not isinstance(snap, dict):
            continue
        for key, val in snap.items():
            if key in ignored_keys:
                continue
            if isinstance(val, int):
                zone_max[key] = max(zone_max.get(key, 0), val)

    return zone_max


def summarize_run(results_path: Path) -> KpiSummary:
    """Create a KPI summary from a simulation results JSON file."""
    results = _load_json(results_path)
    population = _load_population_timeseries(results_path, results)
    records = _iter_decision_records(results)
    decision_telemetry = results.get("decision_telemetry", {})

    action_mix: dict[str, int] = {}
    pace_mix: dict[str, int] = {}
    wait_decisions = 0
    new_cue_only = 0

    for record in records:
        payload = record.get("decision_payload", {})
        if not isinstance(payload, dict):
            continue

        action = payload.get("action")
        if isinstance(action, str):
            action_mix[action] = action_mix.get(action, 0) + 1
            if action == "wait":
                wait_decisions += 1

        pace = payload.get("pace")
        if isinstance(pace, str):
            pace_mix[pace] = pace_mix.get(pace, 0) + 1

        reassess_when = payload.get("reassess_when")
        if reassess_when == "new_cue_only":
            new_cue_only += 1

    final_time = results.get("final_time")
    if final_time is None:
        final_time = results.get("sim_time")

    decisions_total = len(records)
    route_changes = results.get("route_changes", [])
    route_change_count = len(route_changes) if isinstance(route_changes, list) else 0
    fallback_rate = 0.0
    repair_status_counts: dict[str, int] = {}

    if isinstance(decision_telemetry, dict):
        maybe_fallback_rate = decision_telemetry.get("fallback_rate")
        if isinstance(maybe_fallback_rate, (float, int)):
            fallback_rate = float(maybe_fallback_rate)
        maybe_repair = decision_telemetry.get("repair_status_counts")
        if isinstance(maybe_repair, dict):
            repair_status_counts = {
                str(k): int(v)
                for k, v in maybe_repair.items()
                if isinstance(v, int)
            }

    return KpiSummary(
        final_time_s=float(final_time) if isinstance(final_time, (int, float)) else None,
        decisions_total=decisions_total,
        wait_decisions=wait_decisions,
        wait_decision_rate=_safe_ratio(wait_decisions, decisions_total),
        wait_loop_rate=_compute_wait_loop_rate(results),
        fallback_rate=fallback_rate,
        new_cue_only_rate=_safe_ratio(new_cue_only, decisions_total),
        route_change_count=route_change_count,
        action_mix=action_mix,
        pace_mix=pace_mix,
        repair_status_counts=repair_status_counts,
        max_zone_occupancy=_compute_zone_maximums(population),
    )


def _diff_counter(base: dict[str, int], candidate: dict[str, int]) -> dict[str, int]:
    keys = set(base) | set(candidate)
    return {key: candidate.get(key, 0) - base.get(key, 0) for key in sorted(keys)}


def compare_runs(baseline_results: Path, candidate_results: Path) -> dict[str, Any]:
    """Compare two run outputs and return a structured delta report."""
    baseline = summarize_run(baseline_results)
    candidate = summarize_run(candidate_results)

    final_time_delta = None
    if baseline.final_time_s is not None and candidate.final_time_s is not None:
        final_time_delta = candidate.final_time_s - baseline.final_time_s

    report = {
        "baseline": asdict(baseline),
        "candidate": asdict(candidate),
        "delta": {
            "final_time_s": final_time_delta,
            "decisions_total": candidate.decisions_total - baseline.decisions_total,
            "wait_decisions": candidate.wait_decisions - baseline.wait_decisions,
            "wait_decision_rate": candidate.wait_decision_rate - baseline.wait_decision_rate,
            "wait_loop_rate": candidate.wait_loop_rate - baseline.wait_loop_rate,
            "fallback_rate": candidate.fallback_rate - baseline.fallback_rate,
            "new_cue_only_rate": candidate.new_cue_only_rate - baseline.new_cue_only_rate,
            "route_change_count": candidate.route_change_count - baseline.route_change_count,
            "action_mix": _diff_counter(baseline.action_mix, candidate.action_mix),
            "pace_mix": _diff_counter(baseline.pace_mix, candidate.pace_mix),
            "repair_status_counts": _diff_counter(
                baseline.repair_status_counts,
                candidate.repair_status_counts,
            ),
            "max_zone_occupancy": _diff_counter(
                baseline.max_zone_occupancy,
                candidate.max_zone_occupancy,
            ),
        },
        "notes": [
            "Zone occupancy deltas are based on per-zone snapshot maxima.",
            "For queue-density hotspot analysis, ensure monitoring.zones defines explicit queue zones.",
        ],
    }
    return report


def _format_report(report: dict[str, Any]) -> str:
    delta = report.get("delta", {})
    lines = [
        "=== Regression KPI Comparison ===",
        f"final_time_s delta: {delta.get('final_time_s')}",
        f"decisions_total delta: {delta.get('decisions_total')}",
        f"wait_decision_rate delta: {delta.get('wait_decision_rate'):.6f}",
        f"wait_loop_rate delta: {delta.get('wait_loop_rate'):.6f}",
        f"fallback_rate delta: {delta.get('fallback_rate'):.6f}",
        f"new_cue_only_rate delta: {delta.get('new_cue_only_rate'):.6f}",
        f"route_change_count delta: {delta.get('route_change_count')}",
        "action_mix delta:",
    ]

    for action, change in delta.get("action_mix", {}).items():
        lines.append(f"  {action}: {change:+d}")

    lines.append("pace_mix delta:")
    for pace, change in delta.get("pace_mix", {}).items():
        lines.append(f"  {pace}: {change:+d}")

    lines.append("repair_status_counts delta:")
    for status, change in delta.get("repair_status_counts", {}).items():
        lines.append(f"  {status}: {change:+d}")

    lines.append("max_zone_occupancy delta:")
    for zone, change in delta.get("max_zone_occupancy", {}).items():
        lines.append(f"  {zone}: {change:+d}")

    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare two EvacuSim run outputs.")
    parser.add_argument("baseline", type=Path, help="Path to baseline results JSON")
    parser.add_argument("candidate", type=Path, help="Path to candidate results JSON")
    parser.add_argument(
        "--out-json",
        type=Path,
        default=None,
        help="Optional output path for machine-readable comparison JSON",
    )

    args = parser.parse_args(argv)
    report = compare_runs(args.baseline, args.candidate)
    print(_format_report(report))

    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_json, "w") as f:
            json.dump(report, f, indent=2)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
