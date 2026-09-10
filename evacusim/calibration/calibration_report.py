"""Calibration report: realised arrivals/occupancy vs. expected (Feature A).

After a calibration run this compares what the simulation actually produced
against the input data, so the physical model can be tuned:

* **Arrivals** — realised entrance spawns are binned per (entrance, interval)
  and compared to the expected ``arrivals`` from the usage CSV.  Because
  arrivals are a Poisson process, exact counts differ run-to-run; the report
  surfaces per-cell error plus aggregate MAE/RMSE so systematic bias (wrong
  rate) is distinguishable from sampling noise.
* **Occupancy** — peak/mean per-zone occupancy is summarised from the existing
  :class:`~evacusim.metrics.population_monitor.PopulationMonitor` time series.

Writes ``calibration_report.json`` and ``calibration_arrivals.csv`` to the run's
output directory and logs a one-line summary.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

from evacusim.utils.logger import get_logger

logger = get_logger(__name__)


def _bin_realised_entrance_arrivals(spawn_log, intervals) -> dict[tuple[str, int], int]:
    """Count entrance spawns falling in each (entrance_id, interval index)."""
    counts: dict[tuple[str, int], int] = {}
    for entry in spawn_log:
        if entry.get("source") != "entrance":
            continue
        loc = entry.get("location")
        t = float(entry.get("time_s", 0.0))
        for idx, iv in enumerate(intervals):
            if iv.entrance_id == loc and iv.start_s <= t < iv.end_s:
                counts[(loc, idx)] = counts.get((loc, idx), 0) + 1
                break
    return counts


def _occupancy_summary(population_monitor) -> dict[str, dict[str, float]]:
    if population_monitor is None:
        return {}
    try:
        data = population_monitor.to_dict()
    except Exception:  # pragma: no cover - defensive
        return {}
    summary: dict[str, dict[str, float]] = {}
    for zone, series in getattr(population_monitor, "counts", {}).items():
        if not series:
            continue
        summary[zone] = {
            "peak": float(max(series)),
            "mean": float(sum(series) / len(series)),
            "final": float(series[-1]),
        }
    return summary


def build_calibration_report(expected_intervals, spawn_log, population_monitor) -> dict[str, Any]:
    """Assemble the calibration report dict (pure; no file I/O)."""
    realised = _bin_realised_entrance_arrivals(spawn_log, expected_intervals)

    cells = []
    abs_errors: list[float] = []
    sq_errors: list[float] = []
    total_expected = 0
    total_realised = 0
    for idx, iv in enumerate(expected_intervals):
        got = realised.get((iv.entrance_id, idx), 0)
        err = got - iv.arrivals
        abs_errors.append(abs(err))
        sq_errors.append(err * err)
        total_expected += iv.arrivals
        total_realised += got
        cells.append(
            {
                "entrance_id": iv.entrance_id,
                "interval_start_s": iv.start_s,
                "interval_end_s": iv.end_s,
                "expected_arrivals": iv.arrivals,
                "realised_arrivals": got,
                "error": err,
            }
        )

    n = len(cells) or 1
    mae = sum(abs_errors) / n
    rmse = math.sqrt(sum(sq_errors) / n)

    train_spawns = sum(1 for e in spawn_log if e.get("source") == "train")

    return {
        "arrivals": {
            "cells": cells,
            "total_expected_entrance_arrivals": total_expected,
            "total_realised_entrance_arrivals": total_realised,
            "total_train_alighting_spawns": train_spawns,
            "mae": mae,
            "rmse": rmse,
        },
        "occupancy": _occupancy_summary(population_monitor),
    }


def write_calibration_report(
    expected_intervals,
    spawn_log,
    population_monitor,
    out_dir,
) -> dict[str, Any]:
    """Build the report and write JSON + CSV into ``out_dir``. Returns the report."""
    report = build_calibration_report(expected_intervals, spawn_log, population_monitor)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "calibration_report.json"
    with open(json_path, "w") as fh:
        json.dump(report, fh, indent=2)

    csv_path = out_dir / "calibration_arrivals.csv"
    with open(csv_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            ["entrance_id", "interval_start_s", "interval_end_s",
             "expected_arrivals", "realised_arrivals", "error"]
        )
        for c in report["arrivals"]["cells"]:
            writer.writerow(
                [c["entrance_id"], c["interval_start_s"], c["interval_end_s"],
                 c["expected_arrivals"], c["realised_arrivals"], c["error"]]
            )

    a = report["arrivals"]
    logger.info(
        "Calibration report: entrance arrivals realised=%d expected=%d "
        "(MAE=%.2f, RMSE=%.2f); train alighting spawns=%d. Written to %s",
        a["total_realised_entrance_arrivals"],
        a["total_expected_entrance_arrivals"],
        a["mae"],
        a["rmse"],
        a["total_train_alighting_spawns"],
        json_path,
    )
    return report
