import unittest

import evacusim.decision.action_executor as action_executor_module
from evacusim.decision.action_executor import ActionExecutor


class _DummySim:
    def __init__(self):
        self._speeds = {"agent_1": 1.0}

    def get_agent_speed(self, agent_id):
        return self._speeds.get(agent_id)

    def set_agent_speed(self, agent_id, speed):
        self._speeds[agent_id] = speed


class ActionExecutorPaceTests(unittest.TestCase):
    def setUp(self):
        self._original_hurrying = action_executor_module.HURRYING_MULTIPLIER
        self._original_running = action_executor_module.RUNNING_MULTIPLIER

    def tearDown(self):
        action_executor_module.HURRYING_MULTIPLIER = self._original_hurrying
        action_executor_module.RUNNING_MULTIPLIER = self._original_running

    def _build_executor(self, pace_multipliers=None):
        return ActionExecutor(
            jps_sim=_DummySim(),
            state_queries=None,
            event_manager=None,
            station_layout={},
            agent_injured=set(),
            agent_action={},
            agent_last_decision={},
            agent_destinations={},
            wait_events=[],
            agent_configs=[],
            agent_roles={},
            pace_multipliers=pace_multipliers or {},
        )

    def test_apply_hurrying_multiplier_updates_speed(self):
        executor = self._build_executor({"hurrying": 1.25})
        executor._apply_pace_speed("agent_1", "hurrying")
        self.assertAlmostEqual(executor.jps_sim.get_agent_speed("agent_1"), 1.25)

    def test_apply_running_multiplier_updates_speed(self):
        executor = self._build_executor({"running": 1.8})
        executor._apply_pace_speed("agent_1", "running")
        self.assertAlmostEqual(executor.jps_sim.get_agent_speed("agent_1"), 1.8)

    def test_invalid_multiplier_type_raises(self):
        with self.assertRaises(ValueError):
            self._build_executor({"hurrying": "fast"})

    def test_non_positive_multiplier_raises(self):
        with self.assertRaises(ValueError):
            self._build_executor({"running": 0})

    def test_unset_multiplier_raises_when_used(self):
        executor = self._build_executor({})
        with self.assertRaises(ValueError):
            executor._apply_pace_speed("agent_1", "hurrying")

    def test_normal_pace_keeps_speed(self):
        executor = self._build_executor({"hurrying": 1.2, "running": 1.7})
        before = executor.jps_sim.get_agent_speed("agent_1")
        executor._apply_pace_speed("agent_1", "normal_pace")
        after = executor.jps_sim.get_agent_speed("agent_1")
        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
