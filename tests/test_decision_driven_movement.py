"""Agents move, and leave by train, only because of a decision.

- Spawned passengers hold where they are until their first decision; they get
  no implicit journey to a default exit.
- A role with ``target_from_zone`` waits for the train at the platform it
  spawned on.
- A waiting passenger boards a dwelling train only at its own platform.
"""

from types import SimpleNamespace

from shapely.geometry import box

from evacusim.config.schema import AgentRole
from evacusim.coordination.hybrid_simulation import HybridSimulationRunner
from evacusim.setup.agent_manager import AgentManager

# --------------------------------------------------------------------------- #
# Spawn
# --------------------------------------------------------------------------- #


class _SpawnSim:
    """Single-level sim with two platform zones; records add_agent calls."""

    def __init__(self):
        self.geometry_manager = SimpleNamespace(
            walkable_areas_with_obstacles={
                "platform_1": box(0, 0, 10, 10),
                "platform_3": box(20, 0, 30, 10),
            }
        )
        self.added = {}

    def add_agent(self, agent_id, position, walking_speed=1.34, assign_default_destination=True):
        self.added[agent_id] = assign_default_destination


def _params(**role_fields):
    role = AgentRole(
        spawn_zones=["platform_1", "platform_3"],
        goal="Wait on Platform {target}.",
        target=[1, 2, 3, 4],
        **role_fields,
    )
    return SimpleNamespace(
        agents=SimpleNamespace(roles={"waiting_for_train": role}),
        station=SimpleNamespace(zone_boundaries={}),
    )


def _spawn(params):
    sim = _SpawnSim()
    agents = [{"id": f"a{i}"} for i in range(8)]
    positions = [(5.0, 5.0) if i % 2 == 0 else (25.0, 5.0) for i in range(8)]
    AgentManager._add_agents_to_jupedsim(sim, agents, positions, set(), params)
    return sim, agents


def test_spawned_agents_hold_with_no_default_exit():
    sim, _ = _spawn(_params())
    assert sim.added and not any(sim.added.values())


def test_snapshot_agents_hold_with_no_default_exit():
    sim = _SpawnSim()
    AgentManager._add_agents_to_jupedsim_from_snapshot(
        sim, [{"id": "a0", "start_position": (5.0, 5.0)}]
    )
    assert sim.added == {"a0": False}


def test_target_from_zone_waits_for_own_platform():
    _, agents = _spawn(_params(target_from_zone=True))
    for cfg in agents:
        platform = int(cfg["initial_zone"].removeprefix("platform_"))
        assert cfg["target"] == platform
        assert cfg["goal_state"] == f"Wait on Platform {platform}."


def test_target_is_sampled_without_target_from_zone():
    import random

    random.seed(0)
    _, agents = _spawn(_params())
    assert all(cfg["target"] in (1, 2, 3, 4) for cfg in agents)
    assert any(cfg["target"] != int(cfg["initial_zone"][-1]) for cfg in agents)


def test_target_from_zone_keeps_the_random_sequence():
    """Overriding the target draws the same numbers, so other draws don't shift."""
    import random

    random.seed(7)
    _spawn(_params())
    after_sampled = random.random()
    random.seed(7)
    _spawn(_params(target_from_zone=True))
    assert random.random() == after_sampled


# --------------------------------------------------------------------------- #
# Train boarding
# --------------------------------------------------------------------------- #


class _BoardingSim:
    """Records who was eligible to board each train; boards nobody."""

    def __init__(self):
        self.eligible = {}

    def board_agents_on_platform(self, exit_name, agent_destinations=None, eligible_ids=None):
        self.eligible[exit_name] = set(eligible_ids)
        return []


def _runner(goals, targets, destinations, trains):
    r = HybridSimulationRunner.__new__(HybridSimulationRunner)
    r.jps_sim = _BoardingSim()
    r.event_manager = SimpleNamespace(active_train_exits=set(trains))
    r.decision_processor = SimpleNamespace(
        agent_goals=dict(goals), agent_target=lambda aid: targets.get(aid)
    )
    r.agent_destinations = dict(destinations)
    r.exited_agents = set()
    r.exit_log = []
    r.current_sim_time = 20.0
    return r


WAIT = "Wait on Platform 1 and board your train when it arrives."


def test_waiting_passenger_boards_only_at_own_platform():
    r = _runner(
        goals={"own": WAIT, "other": WAIT, "untargeted": WAIT},
        targets={"own": 1, "other": 3, "untargeted": None},
        destinations={},
        trains=["train_platform_1", "train_platform_3"],
    )
    r._board_trains()
    assert r.jps_sim.eligible["train_platform_1"] == {"own", "untargeted"}
    assert r.jps_sim.eligible["train_platform_3"] == {"other", "untargeted"}


def test_calibration_target_names_the_train_exit():
    r = _runner(
        goals={"c": "Travel to the platform to board a train."},
        targets={"c": "train_platform_2"},
        destinations={},
        trains=["train_platform_1", "train_platform_2"],
    )
    r._board_trains()
    assert r.jps_sim.eligible == {"train_platform_1": set(), "train_platform_2": {"c"}}


def test_passenger_leaving_by_another_exit_is_not_boarded():
    r = _runner(
        goals={"leaving": WAIT, "waiting": WAIT},
        targets={"leaving": 1, "waiting": 1},
        # "waiting" chose an exit earlier, then chose to wait: wait clears it.
        destinations={"leaving": "escalator_e_up"},
        trains=["train_platform_1"],
    )
    r._board_trains()
    assert r.jps_sim.eligible["train_platform_1"] == {"waiting"}


def test_alighter_is_not_boarded():
    r = _runner(
        goals={"alighter": "Leave the station and continue with your day."},
        targets={"alighter": None},
        destinations={},
        trains=["train_platform_1"],
    )
    r._board_trains()
    assert r.jps_sim.eligible["train_platform_1"] == set()


# --------------------------------------------------------------------------- #
# Watchdog
# --------------------------------------------------------------------------- #


def test_stepping_aside_cancels_post_transfer_routing():
    """A wedged agent walking from an escalator to its platform re-decides."""
    level = SimpleNamespace(
        geometry_manager=SimpleNamespace(_combined_geometry=box(-20, -20, 20, 20))
    )
    jps = SimpleNamespace(
        simulations={"-1": level},
        get_agent_level=lambda aid: "-1",
        transfer_escape_waypoints={"a": (5.0, 5.0)},
        transfer_platform_waypoints={"a": (10.0, 10.0)},
        targets={},
    )
    jps.set_agent_target = lambda aid, t: jps.targets.__setitem__(aid, t)
    r = HybridSimulationRunner.__new__(HybridSimulationRunner)
    r.jps_sim = jps
    r.agent_destinations = {}
    r.decision_processor = SimpleNamespace(
        clear_goal_for_redecision=lambda aid: None, reset_agent_decision=lambda aid: None
    )
    r._sidestepping = {}
    r.current_sim_time = 100.0

    r._sidestep("a", {"a": (0.0, 0.0), "b": (0.4, 0.0)}, (10.0, 10.0))

    assert "a" in jps.targets and "a" in r._sidestepping
    assert jps.transfer_escape_waypoints == {}
    assert jps.transfer_platform_waypoints == {}
