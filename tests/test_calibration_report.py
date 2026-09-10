"""Feature A: calibration report — arrival binning, MAE/RMSE, file output."""

import json
import tempfile
import unittest
from pathlib import Path

from evacusim.calibration.calibration_report import (
    build_calibration_report,
    write_calibration_report,
)
from evacusim.calibration.usage_data import UsageInterval


class _FakeMonitor:
    counts = {"concourse": [0, 3, 5], "platform": [2, 1, 0]}

    def to_dict(self):
        return {"snapshots": []}


class CalibrationReportTests(unittest.TestCase):
    def _intervals(self):
        return [
            UsageInterval(0, 60, "entrance_a", 3),
            UsageInterval(60, 120, "entrance_a", 5),
        ]

    def _spawn_log(self):
        # 2 realised in first interval (expected 3), 5 in second (expected 5),
        # plus a train spawn that must be ignored for entrance binning.
        return [
            {"source": "entrance", "location": "entrance_a", "time_s": 10.0},
            {"source": "entrance", "location": "entrance_a", "time_s": 50.0},
            {"source": "entrance", "location": "entrance_a", "time_s": 61.0},
            {"source": "entrance", "location": "entrance_a", "time_s": 70.0},
            {"source": "entrance", "location": "entrance_a", "time_s": 80.0},
            {"source": "entrance", "location": "entrance_a", "time_s": 90.0},
            {"source": "entrance", "location": "entrance_a", "time_s": 119.0},
            {"source": "train", "location": "1", "time_s": 30.0},
        ]

    def test_build_report_bins_and_errors(self):
        report = build_calibration_report(self._intervals(), self._spawn_log(), _FakeMonitor())
        a = report["arrivals"]
        self.assertEqual(a["total_expected_entrance_arrivals"], 8)
        self.assertEqual(a["total_realised_entrance_arrivals"], 7)  # 2 + 5
        self.assertEqual(a["total_train_alighting_spawns"], 1)
        # errors: interval0 = 2-3 = -1, interval1 = 5-5 = 0 -> MAE 0.5
        self.assertAlmostEqual(a["mae"], 0.5)
        self.assertAlmostEqual(a["rmse"], (0.5) ** 0.5)
        self.assertEqual(report["occupancy"]["concourse"]["peak"], 5.0)

    def test_write_report_creates_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = write_calibration_report(
                self._intervals(), self._spawn_log(), _FakeMonitor(), tmp
            )
            self.assertTrue((Path(tmp) / "calibration_report.json").exists())
            self.assertTrue((Path(tmp) / "calibration_arrivals.csv").exists())
            loaded = json.loads((Path(tmp) / "calibration_report.json").read_text())
            self.assertEqual(loaded["arrivals"]["mae"], report["arrivals"]["mae"])


if __name__ == "__main__":
    unittest.main()
