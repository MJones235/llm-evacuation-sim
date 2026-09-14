"""Feature A: runtime spawning registers LLM-free agents into the pipeline.

Exercises the real ``HybridSimulationRunner.register_runtime_agent`` /
``_spawn_arrivals`` logic with fakes at the physics/decision boundary (as in
``test_zero_llm_run``), without constructing the whole runner.  Asserts new
agents are added to JuPedSim, registered as ``NoOpAgent``s (no LLM), handed to
the decision processor, and logged — and that due-popping is monotonic.
"""

import unittest

from evacusim.coordination.hybrid_simulation import HybridSimulationRunner
from evacusim.coordination.noop_agent import NoOpAgent
from evacusim.calibration.poisson_scheduler import SpawnEvent
from evacusim.calibration.spawn_controller import RuntimeSpawnController


class _FakeMultiSim:
    """Multi-level sim: presence of ``simulations`` selects the level-aware path."""

    simulations = {"0": object(), "-1": object()}

    def __init__(self, fail_ids=()):
        self.added = []
        self._fail_ids = set(fail_ids)

    def add_agent(self, agent_id, position, walking_speed=1.34, level_id="0",
                  assign_default_destination=True):
        if agent_id in self._fail_ids:
            raise RuntimeError("occupied spawn point")
        self.added.append((agent_id, position, level_id))


class _FakeSingleSim:
    """Single-level sim: no ``simulations`` attribute -> no level_id kwarg."""

    def __init__(self):
        self.added = []

    def add_agent(self, agent_id, position, walking_speed=1.34,
                  assign_default_destination=True):
        self.added.append((agent_id, position))


class _FakeProcessor:
    def __init__(self):
        self.registered = []

    def register_agent(self, cfg):
        self.registered.append(cfg["id"])


_SPAWN_POINTS = {
    "entrance_a": {"level": "0", "xy": [1.0, 2.0]},
    "1": {"level": "-1", "xy": [3.0, 4.0]},
}


def _schedule():
    return [
        SpawnEvent(1.0, "entrance", "entrance_a", "0", "train_platform_1"),
        SpawnEvent(1.0, "train", "1", "-1", "street_exit_a"),
        SpawnEvent(5.0, "entrance", "entrance_a", "0", "train_platform_1"),
    ]


def _make_runner(jps_sim, controller):
    r = HybridSimulationRunner.__new__(HybridSimulationRunner)
    r.jps_sim = jps_sim
    r.concordia_agents = {}
    r.agent_configs = []
    r.decision_processor = _FakeProcessor()
    r.spawn_log = []
    r.spawn_controller = controller
    return r


class RuntimeSpawnTests(unittest.TestCase):
    def test_due_spawns_register_llm_free_agents(self):
        sim = _FakeMultiSim()
        controller = RuntimeSpawnController(_schedule(), _SPAWN_POINTS, seed=1)
        r = _make_runner(sim, controller)

        HybridSimulationRunner._spawn_arrivals(r, 0.5)
        self.assertEqual(r.concordia_agents, {})  # nothing due yet

        HybridSimulationRunner._spawn_arrivals(r, 1.0)
        self.assertEqual(len(r.concordia_agents), 2)
        self.assertEqual(len(sim.added), 2)
        self.assertEqual(len(r.decision_processor.registered), 2)
        self.assertEqual(len(r.spawn_log), 2)
        # All spawned agents are LLM-free NoOpAgents.
        self.assertTrue(all(isinstance(a, NoOpAgent) for a in r.concordia_agents.values()))
        # Level routing preserved: one on "0", one on "-1".
        self.assertEqual({lvl for _, _, lvl in sim.added}, {"0", "-1"})

        HybridSimulationRunner._spawn_arrivals(r, 100.0)
        self.assertEqual(len(r.concordia_agents), 3)  # remaining event consumed once
        self.assertEqual(controller.remaining, 0)

    def test_single_level_path(self):
        sim = _FakeSingleSim()
        controller = RuntimeSpawnController(
            [SpawnEvent(1.0, "entrance", "entrance_a", "0", "train_platform_1")],
            _SPAWN_POINTS, seed=1,
        )
        r = _make_runner(sim, controller)
        HybridSimulationRunner._spawn_arrivals(r, 2.0)
        self.assertEqual(len(sim.added), 1)
        self.assertEqual(len(r.concordia_agents), 1)

    def test_failed_insertion_does_not_register(self):
        # Force the first-built agent id to fail physical insertion.
        controller = RuntimeSpawnController(
            [SpawnEvent(1.0, "entrance", "entrance_a", "0", "train_platform_1")],
            _SPAWN_POINTS, seed=1,
        )
        # Peek the id the controller will assign (counter starts at 0).
        sim = _FakeMultiSim(fail_ids={"calib_entrance_0"})
        r = _make_runner(sim, controller)
        HybridSimulationRunner._spawn_arrivals(r, 2.0)
        self.assertEqual(r.concordia_agents, {})
        self.assertEqual(r.decision_processor.registered, [])
        self.assertEqual(r.spawn_log, [])

    def test_no_controller_is_noop(self):
        sim = _FakeMultiSim()
        r = _make_runner(sim, None)
        HybridSimulationRunner._spawn_arrivals(r, 10.0)
        self.assertEqual(sim.added, [])


if __name__ == "__main__":
    unittest.main()
