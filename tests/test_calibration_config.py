"""Calibration factory wiring.

Schema validation of the section is tested in test_config_schema.py.
"""

import tempfile
import unittest
from pathlib import Path

from evacusim.config.schema import CalibrationConfig
from evacusim.setup.simulation_runner_factory import SimulationRunnerFactory


def _base_calibration(**over):
    cfg = {
        "decision": {"engine": "rule_based"},
        "calibration": {
            "enabled": True,
            "seed": 7,
            "entrance_usage_csv": "u.csv",
            "timetable_csv": "t.csv",
            "entrance_dest_exits": ["train_platform_1"],
            "platform_exit": "street_exit_a",
            "spawn_points": {
                "entrance_a": {"level": "0", "xy": [10.0, 5.0]},
                "1": {"level": "-1", "xy": [20.0, -3.0]},
            },
        },
    }
    cfg["calibration"].update(over)
    return cfg


class FactoryWiringTests(unittest.TestCase):
    def test_build_calibration_disabled_returns_none(self):
        controller, timetable = SimulationRunnerFactory._build_calibration(None)
        self.assertIsNone(controller)
        self.assertEqual(timetable, [])

    def test_build_calibration_builds_controller_and_timetable(self):
        with tempfile.TemporaryDirectory() as tmp:
            u = Path(tmp) / "u.csv"
            u.write_text(
                "interval_start_s,interval_end_s,entrance_id,arrivals\n0,600,entrance_a,60\n"
            )
            t = Path(tmp) / "t.csv"
            t.write_text("arrival_s,platform,alighting,dwell_s\n120,1,10,30\n")
            cfg = _base_calibration()
            cfg["calibration"]["entrance_usage_csv"] = str(u)
            cfg["calibration"]["timetable_csv"] = str(t)

            controller, timetable = SimulationRunnerFactory._build_calibration(
                CalibrationConfig.model_validate(cfg["calibration"])
            )
        self.assertIsNotNone(controller)
        self.assertEqual(len(timetable), 1)
        self.assertGreater(controller.total, 10)  # entrance arrivals + 10 alighters
        self.assertTrue(hasattr(controller, "expected_intervals"))

    def test_load_calibration_train_events_maps_arrivals(self):
        from evacusim.calibration.usage_data import TrainArrival

        class _FakeEM:
            def __init__(self):
                self.scheduled_events = []

        class _FakeRunner:
            def __init__(self):
                self.event_manager = _FakeEM()

        runner = _FakeRunner()
        SimulationRunnerFactory._load_calibration_train_events(
            runner, [TrainArrival(120.0, "1", 10, 30.0)]
        )
        evs = runner.event_manager.scheduled_events
        self.assertEqual(len(evs), 1)
        self.assertEqual(evs[0]["type"], "train_arrival")
        self.assertEqual(evs[0]["platforms"], ["1"])
        self.assertEqual(evs[0]["dwell_seconds"], 30.0)


if __name__ == "__main__":
    unittest.main()
