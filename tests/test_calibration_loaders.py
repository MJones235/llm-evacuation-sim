"""Feature A: CSV loaders parse valid data and reject malformed rows."""

import tempfile
import unittest
from pathlib import Path

from evacusim.calibration.usage_data import (
    CalibrationDataError,
    TrainArrival,
    UsageInterval,
    load_entrance_usage,
    load_timetable,
)


def _write(tmp, name, text):
    p = Path(tmp) / name
    p.write_text(text)
    return p


class UsageLoaderTests(unittest.TestCase):
    def test_valid_usage_parses_and_computes_rate(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = _write(
                tmp,
                "usage.csv",
                "interval_start_s,interval_end_s,entrance_id,arrivals\n"
                "0,60,entrance_a,30\n"
                "60,120,entrance_b,0\n",
            )
            rows = load_entrance_usage(p)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0], UsageInterval(0.0, 60.0, "entrance_a", 30))
        self.assertAlmostEqual(rows[0].rate_per_s, 0.5)
        self.assertEqual(rows[1].rate_per_s, 0.0)  # zero arrivals -> zero rate

    def test_missing_column_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = _write(tmp, "u.csv", "interval_start_s,entrance_id,arrivals\n0,e,1\n")
            with self.assertRaises(CalibrationDataError):
                load_entrance_usage(p)

    def test_bad_interval_and_values_raise(self):
        with tempfile.TemporaryDirectory() as tmp:
            bad_order = _write(
                tmp, "b1.csv",
                "interval_start_s,interval_end_s,entrance_id,arrivals\n60,60,e,1\n",
            )
            with self.assertRaises(CalibrationDataError):
                load_entrance_usage(bad_order)

            neg = _write(
                tmp, "b2.csv",
                "interval_start_s,interval_end_s,entrance_id,arrivals\n0,60,e,-3\n",
            )
            with self.assertRaises(CalibrationDataError):
                load_entrance_usage(neg)

            empty_ent = _write(
                tmp, "b3.csv",
                "interval_start_s,interval_end_s,entrance_id,arrivals\n0,60,,3\n",
            )
            with self.assertRaises(CalibrationDataError):
                load_entrance_usage(empty_ent)

    def test_empty_usage_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = _write(tmp, "e.csv", "interval_start_s,interval_end_s,entrance_id,arrivals\n")
            with self.assertRaises(CalibrationDataError):
                load_entrance_usage(p)


class TimetableLoaderTests(unittest.TestCase):
    def test_valid_timetable_parses(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = _write(
                tmp, "tt.csv",
                "arrival_s,platform,alighting,dwell_s\n"
                "30,1,20,25\n"
                "90,2,0,30\n",
            )
            rows = load_timetable(p)
        self.assertEqual(rows[0], TrainArrival(30.0, "1", 20, 25.0))
        self.assertEqual(len(rows), 2)

    def test_header_only_timetable_is_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = _write(tmp, "tt.csv", "arrival_s,platform,alighting,dwell_s\n")
            self.assertEqual(load_timetable(p), [])

    def test_bad_dwell_and_negative_raise(self):
        with tempfile.TemporaryDirectory() as tmp:
            bad = _write(tmp, "b.csv", "arrival_s,platform,alighting,dwell_s\n30,1,5,0\n")
            with self.assertRaises(CalibrationDataError):
                load_timetable(bad)
            neg = _write(tmp, "n.csv", "arrival_s,platform,alighting,dwell_s\n-1,1,5,10\n")
            with self.assertRaises(CalibrationDataError):
                load_timetable(neg)


if __name__ == "__main__":
    unittest.main()
