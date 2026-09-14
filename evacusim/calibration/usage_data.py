"""CSV loaders for calibration input data (Feature A).

Two data sources drive non-evacuation calibration:

* **Entrance usage** — per-interval passenger arrival counts at each street
  entrance, converted to a Poisson rate (arrivals per second).
* **Train timetable** — train arrivals per platform, each alighting a burst of
  passengers and dwelling for a fixed period.

Both loaders validate every row and raise :class:`CalibrationDataError` with a
1-based row number on malformed input, so bad data fails loudly at load time
rather than mid-run.  The same loaders work for mock and real CSVs — only the
file contents change.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


class CalibrationDataError(ValueError):
    """Raised when a calibration CSV is missing columns or has an invalid row."""


@dataclass(frozen=True)
class UsageInterval:
    """One entrance's expected arrivals over a time interval."""

    start_s: float
    end_s: float
    entrance_id: str
    arrivals: int

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s

    @property
    def rate_per_s(self) -> float:
        """Mean Poisson arrival rate over the interval (arrivals per second)."""
        d = self.duration_s
        return self.arrivals / d if d > 0 else 0.0


@dataclass(frozen=True)
class TrainArrival:
    """One train arriving at a platform, alighting a burst of passengers."""

    arrival_s: float
    platform: str
    alighting: int
    dwell_s: float


_USAGE_COLUMNS = ("interval_start_s", "interval_end_s", "entrance_id", "arrivals")
_TIMETABLE_COLUMNS = ("arrival_s", "platform", "alighting", "dwell_s")


def _require_columns(header, required, path) -> None:
    have = set(header or [])
    missing = [c for c in required if c not in have]
    if missing:
        raise CalibrationDataError(
            f"{path}: missing required column(s): {', '.join(missing)}"
        )


def load_entrance_usage(path) -> list[UsageInterval]:
    """Load entrance usage intervals from a CSV.

    Columns: ``interval_start_s, interval_end_s, entrance_id, arrivals``.
    """
    path = Path(path)
    intervals: list[UsageInterval] = []
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        _require_columns(reader.fieldnames, _USAGE_COLUMNS, path)
        for i, row in enumerate(reader, start=2):  # row 1 is the header
            try:
                start_s = float(row["interval_start_s"])
                end_s = float(row["interval_end_s"])
                entrance_id = (row["entrance_id"] or "").strip()
                arrivals = int(row["arrivals"])
            except (TypeError, ValueError) as e:
                raise CalibrationDataError(f"{path}:row {i}: invalid value ({e})") from e
            if not entrance_id:
                raise CalibrationDataError(f"{path}:row {i}: entrance_id is empty")
            if end_s <= start_s:
                raise CalibrationDataError(
                    f"{path}:row {i}: interval_end_s ({end_s}) must exceed "
                    f"interval_start_s ({start_s})"
                )
            if arrivals < 0:
                raise CalibrationDataError(f"{path}:row {i}: arrivals must be non-negative")
            intervals.append(UsageInterval(start_s, end_s, entrance_id, arrivals))
    if not intervals:
        raise CalibrationDataError(f"{path}: no usage intervals found")
    return intervals


def load_timetable(path) -> list[TrainArrival]:
    """Load train arrivals from a CSV.

    Columns: ``arrival_s, platform, alighting, dwell_s``.  An empty file (header
    only) is allowed — a run may have entrance arrivals but no trains.
    """
    path = Path(path)
    trains: list[TrainArrival] = []
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        _require_columns(reader.fieldnames, _TIMETABLE_COLUMNS, path)
        for i, row in enumerate(reader, start=2):
            try:
                arrival_s = float(row["arrival_s"])
                platform = (row["platform"] or "").strip()
                alighting = int(row["alighting"])
                dwell_s = float(row["dwell_s"])
            except (TypeError, ValueError) as e:
                raise CalibrationDataError(f"{path}:row {i}: invalid value ({e})") from e
            if not platform:
                raise CalibrationDataError(f"{path}:row {i}: platform is empty")
            if arrival_s < 0:
                raise CalibrationDataError(f"{path}:row {i}: arrival_s must be non-negative")
            if alighting < 0:
                raise CalibrationDataError(f"{path}:row {i}: alighting must be non-negative")
            if dwell_s <= 0:
                raise CalibrationDataError(f"{path}:row {i}: dwell_s must be positive")
            trains.append(TrainArrival(arrival_s, platform, alighting, dwell_s))
    return trains
