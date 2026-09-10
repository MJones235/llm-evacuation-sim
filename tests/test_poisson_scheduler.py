"""Feature A: Poisson arrival scheduler — determinism, rate accuracy, ordering."""

import unittest

from evacusim.calibration.poisson_scheduler import SpawnEvent, build_arrival_schedule
from evacusim.calibration.usage_data import TrainArrival, UsageInterval

_SPAWN_CFG = {
    "entrance_level": "0",
    "entrance_dest_exits": ["train_platform_1"],
    "platform_level": "-1",
    "platform_exit": "street_exit_a",
}


class PoissonSchedulerTests(unittest.TestCase):
    def test_same_seed_is_deterministic(self):
        intervals = [UsageInterval(0, 600, "entrance_a", 300)]
        a = build_arrival_schedule(intervals, [], _SPAWN_CFG, seed=7)
        b = build_arrival_schedule(intervals, [], _SPAWN_CFG, seed=7)
        self.assertEqual(a, b)

    def test_different_seed_differs(self):
        intervals = [UsageInterval(0, 600, "entrance_a", 300)]
        a = build_arrival_schedule(intervals, [], _SPAWN_CFG, seed=1)
        b = build_arrival_schedule(intervals, [], _SPAWN_CFG, seed=2)
        self.assertNotEqual([e.time_s for e in a], [e.time_s for e in b])

    def test_rate_accuracy_within_tolerance(self):
        # 3600 s at 0.5/s -> expect ~1800 entrance arrivals.
        intervals = [UsageInterval(0, 3600, "entrance_a", 1800)]
        sched = build_arrival_schedule(intervals, [], _SPAWN_CFG, seed=123)
        entrance = [e for e in sched if e.source == "entrance"]
        # Poisson(1800) has sd ~42; allow generous +/-8%.
        self.assertTrue(1650 <= len(entrance) <= 1950, f"got {len(entrance)}")
        # all within the interval bounds
        self.assertTrue(all(0 <= e.time_s < 3600 for e in entrance))

    def test_zero_rate_interval_produces_no_entrance_events(self):
        intervals = [UsageInterval(0, 600, "entrance_a", 0)]
        sched = build_arrival_schedule(intervals, [], _SPAWN_CFG, seed=5)
        self.assertEqual([e for e in sched if e.source == "entrance"], [])

    def test_timetable_expands_to_alighting_spawns(self):
        timetable = [TrainArrival(120.0, "1", 15, 30.0), TrainArrival(240.0, "2", 5, 30.0)]
        sched = build_arrival_schedule([], timetable, _SPAWN_CFG, seed=0)
        train = [e for e in sched if e.source == "train"]
        self.assertEqual(len(train), 20)
        at_120 = [e for e in train if e.time_s == 120.0]
        self.assertEqual(len(at_120), 15)
        self.assertTrue(all(e.dest_exit == "street_exit_a" for e in train))
        self.assertTrue(all(e.level == "-1" for e in train))

    def test_output_is_time_sorted(self):
        intervals = [UsageInterval(0, 600, "entrance_a", 120)]
        timetable = [TrainArrival(300.0, "1", 10, 30.0)]
        sched = build_arrival_schedule(intervals, timetable, _SPAWN_CFG, seed=9)
        times = [e.time_s for e in sched]
        self.assertEqual(times, sorted(times))

    def test_multiple_dest_exits_are_used(self):
        cfg = dict(_SPAWN_CFG, entrance_dest_exits=["train_platform_1", "train_platform_2"])
        intervals = [UsageInterval(0, 3600, "entrance_a", 2000)]
        sched = build_arrival_schedule(intervals, [], cfg, seed=3)
        dests = {e.dest_exit for e in sched if e.source == "entrance"}
        self.assertEqual(dests, {"train_platform_1", "train_platform_2"})


if __name__ == "__main__":
    unittest.main()
