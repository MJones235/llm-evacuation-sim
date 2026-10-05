"""Escalator conveyors on the real Monument geometry (floor simulation + conveyor).

Requires the Monument geometry; set EVACUSIM_MONUMENT_NETWORK or keep the
monument-evacuation checkout next to this repo. Skipped otherwise.
"""

import math
import os
import random
from pathlib import Path

import pytest
from shapely.geometry import Point, Polygon

NETWORK = Path(os.environ.get(
    "EVACUSIM_MONUMENT_NETWORK",
    Path(__file__).resolve().parents[2] / "monument-evacuation" / "geometry" / "monument" / "network",
))
pytestmark = pytest.mark.skipif(not (NETWORK / "level_-1.xml").exists(),
                                reason="Monument geometry not available")

DT = 0.1


def make_sim(**kwargs):
    from evacusim.jps.multi_level_simulation import MultiLevelJuPedSimulation
    return MultiLevelJuPedSimulation(network_path=NETWORK, dt=DT, levels=["0", "-1"], **kwargs)


def spawn_on_platforms(ml, n, seed=1):
    """n agents on the platform 1/2 area of level -1, each with its own speed."""
    floor = ml.simulations["-1"].geometry_manager._combined_geometry
    area = floor.intersection(Polygon([(-25, -28), (2, -28), (2, -5), (-25, -5)])).buffer(-0.3)
    rng = random.Random(seed)
    placed, speeds = [], {}
    x0, y0, x1, y1 = area.bounds
    while len(placed) < n:
        p = (rng.uniform(x0, x1), rng.uniform(y0, y1))
        if area.contains(Point(p)) and all(math.dist(p, q) >= 0.6 for q in placed):
            agent_id = f"a{len(placed)}"
            speeds[agent_id] = rng.uniform(0.6, 1.6)
            ml.add_agent(agent_id, p, walking_speed=speeds[agent_id], level_id="-1",
                         assign_default_destination=False)
            placed.append(p)
    return speeds


def run_until(ml, done, max_s):
    t = 0.0
    while t < max_s and not done():
        ml.step()
        t += DT
    return t


def test_sixty_agents_ride_escalator_f_without_deadlock():
    ml = make_sim()
    es = ml.escalator_system
    speeds = spawn_on_platforms(ml, 60)
    for agent_id in speeds:
        ml.set_agent_destination_exit(agent_id, "escalator_f_up")
    assert len(es.queue["escalator_f_up"]) == 60

    t = run_until(ml, lambda: all(ml.agent_levels.get(a) == "0" for a in speeds), 400)

    arrived = [a for a in speeds if ml.agent_levels.get(a) == "0"]
    assert len(arrived) == 60, f"only {len(arrived)}/60 reached the concourse in {t:.0f}s"
    log = [r for r in es.ride_log if r["escalator"] == "escalator_f_up"]
    assert len(log) == 60
    spec = es.escalators["escalator_f_up"].spec
    standers = [r["ride_s"] for r in log if r["lane"] == "stand"]
    assert standers and all(r >= spec.length_m / spec.belt_speed - 1.0 for r in standers)

    # Boarding throughput: at or below the conveyor ceiling, and not starved.
    boards = sorted(r["board_s"] for r in log)
    rate = (len(boards) - 1) / (boards[-1] - boards[0])
    ceiling = spec.belt_speed / spec.step_depth + (spec.belt_speed + 1.6 * spec.walk_speed_factor) / (2 * spec.step_depth)
    assert 0.4 <= rate <= ceiling, rate

    # Each rider keeps their own walking speed after stepping off.
    for agent_id in arrived:
        assert ml.agent_base_speed[agent_id] == pytest.approx(speeds[agent_id])
        assert ml.simulations["0"].get_agent_speed(agent_id) == pytest.approx(speeds[agent_id])


def test_riders_are_in_transit_not_gone():
    ml = make_sim()
    spawn_on_platforms(ml, 3)
    for agent_id in ("a0", "a1", "a2"):
        ml.set_agent_destination_exit(agent_id, "escalator_f_up")
    es = ml.escalator_system
    run_until(ml, lambda: bool(es.riding), 120)
    rider = next(iter(es.riding))
    assert rider not in ml.agent_levels
    assert ml.is_agent_in_transit(rider)
    assert ml.get_agent_position(rider) is None


def test_up_escalator_refused_from_concourse():
    ml = make_sim()
    floor = ml.simulations["0"].geometry_manager._combined_geometry.buffer(-0.5)
    p = floor.representative_point()
    ml.add_agent("c", (p.x, p.y), level_id="0", assign_default_destination=False)
    ml.set_agent_destination_exit("c", "escalator_f_up")
    assert ml.escalator_system.queued_exit("c") is None


def test_rerouting_leaves_the_queue():
    ml = make_sim()
    spawn_on_platforms(ml, 2)
    ml.set_agent_destination_exit("a0", "escalator_f_up")
    assert ml.escalator_system.queued_exit("a0") == "escalator_f_up"
    pos = ml.get_agent_position("a0")
    ml.set_agent_target("a0", pos)
    assert ml.escalator_system.queued_exit("a0") is None
    ml.set_agent_destination_exit("a1", "escalator_f_up")
    ml.set_agent_destination_exit("a1", "escalator_e_up")
    assert ml.escalator_system.queued_exit("a1") == "escalator_e_up"
    assert "a1" not in ml.escalator_system.queue["escalator_f_up"]


def test_closing_escalator_releases_queue_and_riders_finish():
    ml = make_sim()
    speeds = spawn_on_platforms(ml, 20)
    for agent_id in speeds:
        ml.set_agent_destination_exit(agent_id, "escalator_f_up")
    es = ml.escalator_system
    run_until(ml, lambda: len(es.riding) >= 3, 200)
    riders = set(es.riding)
    ml.block_escalator("escalator_f_up")
    assert not es.queue["escalator_f_up"]
    assert ml.agents_needing_redecision >= {a for a in speeds if a not in riders
                                            and ml.agent_levels.get(a) == "-1"}
    run_until(ml, lambda: not es.riding, 120)
    assert all(ml.agent_levels.get(a) == "0" for a in riders)

    # Choosing it again does not queue: the agent walks over, sees it is
    # closed, and is flagged to re-decide.
    late = next(a for a in speeds if ml.agent_levels.get(a) == "-1")
    ml.agents_needing_redecision.clear()
    ml.set_agent_destination_exit(late, "escalator_f_up")
    assert not es.queue["escalator_f_up"]
    assert es.discovering == {late: "escalator_f_up"}
    run_until(ml, lambda: late in ml.agents_needing_redecision, 120)
    assert late in ml.agents_needing_redecision and late not in es.discovering


def test_pre_blocked_escalator_is_not_offered():
    ml = make_sim(initially_blocked_exits={"escalator_f_up"})
    assert "escalator_f_up" not in ml.simulations["-1"].exit_manager.evacuation_exits
    assert "escalator_f_up" in ml.simulations["-1"].geometry_manager.blocked_exit_positions
    assert ml.escalator_system.escalators["escalator_f_up"].conveyor.closed


def test_last_agent_in_station_still_completes_the_ride():
    """The run must not end while the only agent is stepping onto an escalator."""
    ml = make_sim()
    ml.add_agent("solo", (-35.5, 37.5), walking_speed=1.0, level_id="-1",
                 assign_default_destination=False)
    ml.set_agent_destination_exit("solo", "escalator_c_up")
    run_until(ml, lambda: ml.agent_levels.get("solo") == "0", 120)
    assert ml.agent_levels.get("solo") == "0"


def test_displaced_early_joiner_does_not_keep_the_head_of_the_line():
    """Line order is physical: someone pushed behind the line loses the head slot."""
    ml = make_sim()
    es = ml.escalator_system
    entry = es.escalators["escalator_c_up"]
    es.lane_for = lambda agent_id, exit_name: "stand"
    head, second = entry.line_slots["stand"][:2]
    behind = (entry.line_slots["stand"][-1][0] - 1.5, head[1])
    ml.add_agent("early", behind, level_id="-1", assign_default_destination=False)
    ml.add_agent("front", second, level_id="-1", assign_default_destination=False)
    for agent_id in ("early", "front"):
        ml.set_agent_destination_exit(agent_id, "escalator_c_up")
    q = es.queue["escalator_c_up"]
    q["early"].join_s, q["early"].slot = 0.0, head    # joined first, then pushed back
    q["front"].join_s = 5.0
    es._assign_slots("-1")
    assert q["front"].slot == head
    assert q["early"].slot != head
