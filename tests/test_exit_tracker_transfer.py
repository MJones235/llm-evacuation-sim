import unittest

from evacusim.jps.exit_tracker import ExitTracker


class _MultiLevelSimulation:
    def __init__(self, tracked_ids=()):
        self.agent_levels = {agent_id: "0" for agent_id in tracked_ids}

    def get_all_agent_positions(self):
        return {}


class ExitTrackerTransferTests(unittest.TestCase):
    def test_temporarily_absent_tracked_agent_is_not_marked_exited(self):
        exited = set()
        tracker = ExitTracker(
            agents={"transferring": object()},
            exited_agents=exited,
            agent_destinations={},
            jps_sim=_MultiLevelSimulation({"transferring"}),
        )

        tracker.check_exited_agents(current_sim_time=100.0, current_step=1)

        self.assertEqual(exited, set())

    def test_untracked_missing_agent_is_still_classified_as_exited(self):
        exited = set()
        tracker = ExitTracker(
            agents={"gone": object()},
            exited_agents=exited,
            agent_destinations={},
            jps_sim=_MultiLevelSimulation(),
        )

        tracker.check_exited_agents(current_sim_time=100.0, current_step=1)

        self.assertEqual(exited, {"gone"})


if __name__ == "__main__":
    unittest.main()
