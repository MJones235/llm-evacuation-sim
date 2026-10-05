"""Regression tests for recovering from JuPedSim's "outside of accessible
area" step failure instead of aborting the whole simulation.

This is a pre-existing JuPedSim numerical fragility (confirmed to occur even
with escalator admission throttling in place, and in the original unfixed
code too) where the collision-avoidance solver's per-step position update
can overshoot the walkable navmesh boundary under extreme local density.
Admission-side throttling reduces how often this happens but cannot fully
rule it out, so ConcordiaJuPedSimulation.step() now attempts one recovery:
remove the agent nearest the reported bad point and retry once. See
ConcordiaJuPedSimulation._recover_from_step_exception.
"""

import pytest

from evacusim.jps.jupedsim_integration import ConcordiaJuPedSimulation


class _FakeJpsSimulation:
    def __init__(self, iterate_effects):
        self._iterate_effects = list(iterate_effects)
        self.marked_for_removal: list[int] = []
        self.iterate_calls = 0

    def iterate(self):
        self.iterate_calls += 1
        effect = self._iterate_effects.pop(0)
        if effect is not None:
            raise effect

    def agent_count(self):
        return 1

    def mark_agent_for_removal(self, jps_id):
        self.marked_for_removal.append(jps_id)


class _FakeAgentTracker:
    def __init__(self, positions, jps_ids):
        self.agent_ids = dict(jps_ids)
        self.jps_to_concordia = {v: k for k, v in jps_ids.items()}
        self._positions = dict(positions)

    def get_all_positions(self):
        return dict(self._positions)

    def get_jps_id(self, agent_id):
        return self.agent_ids.get(agent_id)


def _make_sim(iterate_effects, positions, jps_ids):
    sim = ConcordiaJuPedSimulation.__new__(ConcordiaJuPedSimulation)
    sim.is_complete = False
    sim.current_step = 0
    sim.level_id = "-1"
    sim.simulation = _FakeJpsSimulation(iterate_effects)
    sim.agent_tracker = _FakeAgentTracker(positions, jps_ids)
    sim.agent_assigned_exits = {}
    return sim


def test_step_recovers_by_removing_nearest_agent_and_retrying():
    exc = RuntimeError("Point (1.0, 2.0) is outside of accessible area")
    sim = _make_sim(
        iterate_effects=[exc, None],
        positions={"agent1": (1.05, 2.02), "agent2": (50.0, 50.0)},
        jps_ids={"agent1": 11, "agent2": 22},
    )

    result = sim.step()

    assert result is True
    assert sim.simulation.marked_for_removal == [11]  # the nearer agent
    assert sim.simulation.iterate_calls == 2  # original attempt + one retry
    assert "agent1" not in sim.agent_tracker.agent_ids
    assert "agent2" in sim.agent_tracker.agent_ids  # untouched


def test_step_reraises_when_recovery_retry_also_fails():
    exc = RuntimeError("Point (1.0, 2.0) is outside of accessible area")
    sim = _make_sim(
        iterate_effects=[exc, exc],
        positions={"agent1": (1.0, 2.0)},
        jps_ids={"agent1": 11},
    )

    with pytest.raises(RuntimeError):
        sim.step()


def test_step_reraises_unrelated_exceptions_without_attempting_recovery():
    exc = ValueError("something else entirely")
    sim = _make_sim(iterate_effects=[exc], positions={"agent1": (1.0, 2.0)}, jps_ids={"agent1": 11})

    with pytest.raises(ValueError):
        sim.step()

    assert sim.simulation.marked_for_removal == []  # no recovery attempted


def test_recovery_picks_the_nearest_agent_among_several():
    sim = _make_sim(
        iterate_effects=[None],  # the retry inside recovery
        positions={
            "far": (100.0, 100.0),
            "near": (1.1, 2.1),
            "nearer": (1.01, 2.01),
        },
        jps_ids={"far": 1, "near": 2, "nearer": 3},
    )
    exc = RuntimeError("Point (1.0, 2.0) is outside of accessible area")

    recovered = sim._recover_from_step_exception(exc, sim.agent_tracker.get_all_positions())

    assert recovered is True
    assert sim.simulation.marked_for_removal == [3]  # "nearer"
