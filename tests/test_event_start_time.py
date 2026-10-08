import unittest
from types import SimpleNamespace

from evacusim.coordination.hybrid_simulation import HybridSimulationRunner
from evacusim.systems.event_manager import EventManager


class EventStartTimeTests(unittest.TestCase):
    def _manager(self):
        return EventManager({"exits": {}}, object())

    def test_skips_old_messages_and_keeps_future_events(self):
        manager = self._manager()
        old = {"time": 100.0, "message": "old"}
        future = {"time": 300.0, "message": "future"}
        manager.scheduled_events = [old, future]

        manager.prepare_for_start_time(200.0)

        self.assertTrue(old["_fired"])
        self.assertNotIn("_fired", future)

    def test_applies_persistent_blockage_before_start(self):
        manager = self._manager()
        event = {"time": 100.0, "type": "block_exit", "exits": ["closed_exit"]}
        manager.scheduled_events = [event]

        manager.prepare_for_start_time(200.0)

        self.assertEqual(manager.blocked_exits, {"closed_exit"})
        self.assertTrue(event["_fired"])

    def test_retains_train_whose_dwell_overlaps_start(self):
        manager = self._manager()
        event = {
            "time": 190.0,
            "type": "train_arrival",
            "platforms": [2],
            "dwell_seconds": 30.0,
        }
        manager.scheduled_events = [event]

        manager.prepare_for_start_time(200.0)

        self.assertEqual(manager.active_train_exits, {"train_platform_2"})
        self.assertEqual(manager._train_departure_times["train_platform_2"], 220.0)
        self.assertTrue(event["_fired"])

    def test_repeating_event_resumes_at_next_interval(self):
        manager = self._manager()
        event = {"time": 100.0, "message": "repeat", "repeat_interval": 60.0}
        manager.scheduled_events = [event]

        manager.prepare_for_start_time(205.0)

        self.assertEqual(event["_last_fired"], 160.0)
        self.assertNotIn("_fired", event)

    def test_bootstrap_uses_absolute_start_clock(self):
        runner = HybridSimulationRunner.__new__(HybridSimulationRunner)
        runner.start_time_s = 27000.0
        runner.observation_coordinator = SimpleNamespace(
            generate_all_observations=lambda time: {"time": time}
        )
        seen = []
        runner.decision_processor = SimpleNamespace(
            process_all_agents=lambda observations, time: seen.append((observations, time)) or time
        )

        runner._bootstrap_initial_decisions()

        self.assertEqual(seen, [({"time": 27000.0}, 27000.0)])
        self.assertEqual(runner.last_decision_time, 27000.0)


if __name__ == "__main__":
    unittest.main()
