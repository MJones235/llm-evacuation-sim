import unittest

import evacusim.decision.action_executor as action_executor_module
from evacusim.decision.action_executor import ActionExecutor


class _DummySim:
    def __init__(self):
        self._speeds = {"agent_1": 1.0}
        self.targets = []

    def get_agent_speed(self, agent_id):
        return self._speeds.get(agent_id)

    def set_agent_speed(self, agent_id, speed):
        self._speeds[agent_id] = speed

    def set_agent_target(self, agent_id, target):
        self.targets.append((agent_id, target))


class ActionExecutorPaceTests(unittest.TestCase):
    def setUp(self):
        self._original_hurrying = action_executor_module.HURRYING_MULTIPLIER
        self._original_running = action_executor_module.RUNNING_MULTIPLIER

    def tearDown(self):
        action_executor_module.HURRYING_MULTIPLIER = self._original_hurrying
        action_executor_module.RUNNING_MULTIPLIER = self._original_running

    def _build_executor(self, pace_multipliers=None, agent_destinations=None, agent_configs=None):
        return ActionExecutor(
            jps_sim=_DummySim(),
            state_queries=type(
                "StateQueries",
                (),
                {"get_agent_position": lambda self, agent_id: (5.0, 6.0)},
            )(),
            event_manager=None,
            station_layout={},
            agent_injured=set(),
            agent_action={},
            agent_last_decision={},
            agent_destinations=agent_destinations or {},
            wait_events=[],
            agent_configs=agent_configs or [],
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

    def test_train_bound_wait_preserves_down_escalator_route(self):
        destinations = {"agent_1": "escalator_d_down"}
        executor = self._build_executor(
            agent_destinations=destinations,
            agent_configs=[{"id": "agent_1", "goal_state": "Board a train at Platform 2."}],
        )

        executor._handle_wait_action("agent_1", {"wait_reason": "awaiting_information"}, 10.0)

        self.assertEqual(destinations["agent_1"], "escalator_d_down")
        self.assertEqual(executor.agent_action["agent_1"], "moving")
        self.assertEqual(executor.jps_sim.targets, [])

    def test_stationary_wait_clears_replaced_route(self):
        destinations = {"agent_1": "escalator_d_down"}
        executor = self._build_executor(
            agent_destinations=destinations,
            agent_configs=[{"id": "agent_1", "goal_state": "Board a train at Platform 2."}],
        )

        executor._handle_wait_action("agent_1", {"wait_reason": "route_blocked"}, 10.0)

        self.assertNotIn("agent_1", destinations)
        self.assertEqual(executor.jps_sim.targets, [("agent_1", (5.0, 6.0))])


if __name__ == "__main__":
    unittest.main()
