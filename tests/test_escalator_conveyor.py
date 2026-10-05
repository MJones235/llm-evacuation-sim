"""Kinematics of the two-lane escalator conveyor (no JuPedSim)."""

import random

import pytest

from evacusim.escalators.conveyor import Conveyor, ConveyorParams

DT = 0.1


def make(length=10.0, belt=0.5, step=0.4, gap_prob=0.0, seed=0):
    return Conveyor(ConveyorParams(length, belt, step, gap_prob), rng=random.Random(seed))


def run_greedy(conv, lane, seconds, walk_speed=0.0):
    """Board on ``lane`` whenever allowed; return boarding times."""
    times, t, n = [], 0.0, 0
    while t < seconds:
        if conv.can_admit(lane):
            conv.board(f"a{n}", lane, walk_speed, t)
            times.append(t)
            n += 1
        for rider in conv.step(DT):
            conv.remove(rider.agent_id)
        t += DT
    return times


def test_stand_lane_rate_never_exceeds_one_per_step():
    conv = make(gap_prob=0.0)
    times = run_greedy(conv, "stand", 60)
    rate = (len(times) - 1) / (times[-1] - times[0])
    assert rate <= conv.max_boarding_rate("stand") + 1e-6
    assert rate == pytest.approx(1.25, rel=0.15)


def test_stander_step_gap_halves_rate():
    always_gap = run_greedy(make(gap_prob=1.0), "stand", 60)
    never_gap = run_greedy(make(gap_prob=0.0), "stand", 60)
    assert len(always_gap) == pytest.approx(len(never_gap) / 2, rel=0.15)


def test_stander_ride_time_is_length_over_belt():
    conv = make(length=10.0, belt=0.5)
    conv.board("a", "stand", 1.0, 0.0)
    t = 0.0
    while not conv.arrived():
        conv.step(DT)
        t += DT
    assert t == pytest.approx(20.0, abs=DT)


def test_walkers_keep_a_free_step_and_never_overtake():
    conv = make(length=30.0)
    rng = random.Random(3)
    t, n = 0.0, 0
    order = []
    for _ in range(600):
        if conv.can_admit("walk"):
            conv.board(f"w{n}", "walk", rng.uniform(0.2, 1.5), t)
            order.append(f"w{n}")
            n += 1
        conv.step(DT)
        walkers = conv.riders["walk"]
        # A rider parked at the far comb is stepped off (or the belt paused)
        # that same step, so spacing applies to riders still on the incline.
        on_incline = [r for r in walkers if r.s < conv.params.length_m]
        for ahead, behind in zip(on_incline, on_incline[1:]):
            assert ahead.s - behind.s >= 2 * conv.params.step_depth - 1e-6
        assert [r.agent_id for r in walkers] == order[len(order) - len(walkers):]
        for rider in conv.arrived():
            conv.remove(rider.agent_id)
        t += DT


def test_walker_is_faster_than_stander_but_never_slower_than_belt():
    conv = make(length=10.0)
    conv.board("s", "stand", 0.0, 0.0)
    conv.board("w", "walk", 0.6, 0.0)
    conv.step(1.0)
    s = {r.agent_id: r.s for r in conv.riders["stand"] + conv.riders["walk"]}
    assert s["s"] == pytest.approx(0.5)
    assert s["w"] == pytest.approx(1.1)


def test_paused_belt_freezes_riders_and_boarding():
    conv = make()
    conv.board("a", "stand", 1.0, 0.0)
    conv.step(1.0)
    conv.pause()
    before = conv.riders["stand"][0].s
    conv.step(5.0)
    assert conv.riders["stand"][0].s == before
    assert not conv.can_admit("walk")
    conv.resume()
    conv.step(1.0)
    assert conv.riders["stand"][0].s > before


def test_close_stops_boarding_but_riders_finish():
    conv = make(length=5.0)
    conv.board("a", "stand", 0.0, 0.0)
    conv.close()
    assert not conv.can_admit("stand") and not conv.can_admit("walk")
    for _ in range(200):
        conv.step(DT)
    assert [r.agent_id for r in conv.arrived()] == ["a"]


def test_stopped_belt_is_a_staircase():
    conv = make(length=10.0)
    conv.board("s", "stand", 0.4, 0.0)
    conv.stop_belt()
    conv.step(1.0)
    assert conv.riders["stand"][0].s == pytest.approx(0.4)
    conv.board("slow", "stand", 0.0, 1.0)
    conv.step(1.0)
    assert conv.riders["stand"][1].s == 0.0


def test_arrived_riders_are_clamped_at_far_comb():
    conv = make(length=1.0)
    conv.board("a", "walk", 2.0, 0.0)
    conv.step(10.0)
    assert conv.riders["walk"][0].s == 1.0
    assert conv.snapshot() == [["a", "walk", 1.0]]


def test_invalid_params_rejected():
    with pytest.raises(ValueError):
        ConveyorParams(length_m=0)
    with pytest.raises(ValueError):
        ConveyorParams(length_m=10, stander_step_gap_prob=1.5)
