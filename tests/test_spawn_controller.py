"""Feature A: RuntimeSpawnController — due-popping, cfg building, positions."""

import unittest

from evacusim.calibration.poisson_scheduler import SpawnEvent
from evacusim.calibration.spawn_controller import RuntimeSpawnController

_SPAWN_POINTS = {
    "entrance_a": {"level": "0", "xy": [10.0, 5.0]},
    "1": {"level": "-1", "xy": [20.0, -3.0]},
}


def _schedule():
    return [
        SpawnEvent(1.0, "entrance", "entrance_a", "0", "train_platform_1"),
        SpawnEvent(1.0, "train", "1", "-1", "street_exit_a"),
        SpawnEvent(5.0, "entrance", "entrance_a", "0", "train_platform_1"),
    ]


class PopDueTests(unittest.TestCase):
    def test_pop_due_is_monotonic_and_no_double_spawn(self):
        c = RuntimeSpawnController(_schedule(), _SPAWN_POINTS, seed=1)
        self.assertEqual(c.pop_due(0.5), [])
        first = c.pop_due(1.0)
        self.assertEqual(len(first), 2)  # both t=1.0 events
        self.assertEqual(c.pop_due(1.0), [])  # already consumed
        rest = c.pop_due(100.0)
        self.assertEqual(len(rest), 1)
        self.assertEqual(c.remaining, 0)
        self.assertEqual(c.spawned, 3)

    def test_total_and_len(self):
        c = RuntimeSpawnController(_schedule(), _SPAWN_POINTS)
        self.assertEqual(len(c), 3)
        self.assertEqual(c.total, 3)


class BuildCfgTests(unittest.TestCase):
    def test_entrance_cfg_fields(self):
        c = RuntimeSpawnController(_schedule(), _SPAWN_POINTS, seed=2, jitter_m=0.0)
        ev = SpawnEvent(1.0, "entrance", "entrance_a", "0", "train_platform_1")
        cfg, pos, level = c.build_agent_cfg(ev)
        self.assertEqual(level, "0")
        self.assertEqual(pos, (10.0, 5.0))  # zero jitter -> exact spawn point
        self.assertEqual(cfg["level_id"], "0")
        self.assertEqual(cfg["initial_zone"], "concourse")
        self.assertEqual(cfg["target"], "train_platform_1")
        self.assertIn("train", cfg["goal_state"].lower())  # boarding goal
        self.assertEqual(cfg["spawn_source"], "entrance")
        self.assertFalse(cfg["is_injured"])

    def test_train_cfg_fields(self):
        c = RuntimeSpawnController(_schedule(), _SPAWN_POINTS, seed=2, jitter_m=0.0)
        ev = SpawnEvent(1.0, "train", "1", "-1", "street_exit_a")
        cfg, pos, level = c.build_agent_cfg(ev)
        self.assertEqual(level, "-1")
        self.assertEqual(pos, (20.0, -3.0))
        self.assertEqual(cfg["initial_zone"], "platform")
        self.assertEqual(cfg["target"], "street_exit_a")
        self.assertIn("leave", cfg["goal_state"].lower())  # exit goal, no train keyword
        self.assertNotIn("train", cfg["goal_state"].lower())

    def test_ids_are_unique(self):
        c = RuntimeSpawnController(_schedule(), _SPAWN_POINTS, seed=2)
        ev = SpawnEvent(1.0, "entrance", "entrance_a", "0", "train_platform_1")
        ids = {c.build_agent_cfg(ev)[0]["id"] for _ in range(5)}
        self.assertEqual(len(ids), 5)

    def test_jitter_stays_within_radius(self):
        c = RuntimeSpawnController(_schedule(), _SPAWN_POINTS, seed=4, jitter_m=0.5)
        ev = SpawnEvent(1.0, "entrance", "entrance_a", "0", "train_platform_1")
        for _ in range(50):
            _, (x, y), _ = c.build_agent_cfg(ev)
            self.assertLessEqual(((x - 10.0) ** 2 + (y - 5.0) ** 2) ** 0.5, 0.5 + 1e-9)

    def test_unknown_spawn_point_raises(self):
        c = RuntimeSpawnController(_schedule(), _SPAWN_POINTS)
        ev = SpawnEvent(1.0, "entrance", "nowhere", "0", "x")
        with self.assertRaises(KeyError):
            c.build_agent_cfg(ev)


if __name__ == "__main__":
    unittest.main()
