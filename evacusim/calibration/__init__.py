"""Calibration under non-evacuation conditions (Feature A).

This package drives a *normal-operation* station model — passengers arrive over
time as a Poisson process from usage data and train timetables, walk to their
destination, and depart — so simulated occupancy/flow can be compared against
real data to calibrate the physical model.  It is built on the LLM-free
rule-based decision engine (Feature B): calibration runs make no model calls.

Modules:
    usage_data          CSV loaders for entrance usage + train timetables.
    poisson_scheduler   Deterministic Poisson arrival schedule builder.
    spawn_controller     Turns a schedule into runtime agent registrations.
    calibration_report  Compares realised arrivals/occupancy against expected.
"""

from evacusim.calibration.usage_data import (
    CalibrationDataError,
    TrainArrival,
    UsageInterval,
    load_entrance_usage,
    load_timetable,
)
from evacusim.calibration.poisson_scheduler import SpawnEvent, build_arrival_schedule
from evacusim.calibration.spawn_controller import RuntimeSpawnController

__all__ = [
    "CalibrationDataError",
    "TrainArrival",
    "UsageInterval",
    "load_entrance_usage",
    "load_timetable",
    "SpawnEvent",
    "build_arrival_schedule",
    "RuntimeSpawnController",
]
