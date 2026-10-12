"""The progress watchdog: who counts as wedged in place."""

import math

from evacusim.coordination.progress_watchdog import ProgressWatchdog, sidestep_candidates


def _run(watchdog, seconds, positions, targets, exempt=()):
    stuck = []
    for t in range(seconds + 1):
        stuck += [(t, a) for a in watchdog.stuck(float(t), positions(t), targets, exempt)]
    return stuck


def test_still_with_somewhere_to_go_is_stuck_once_per_period():
    w = ProgressWatchdog(stuck_after_s=45)
    stuck = _run(w, 100, lambda t: {"a": (0.0, 0.0)}, {"a": (10.0, 0.0)})
    assert stuck == [(45, "a"), (90, "a")]
    assert w.freed[0] == {"time_s": 45.0, "agent_id": "a", "position": (0.0, 0.0)}


def test_moving_arrived_unwatched_and_exempt_are_not_stuck():
    w = ProgressWatchdog(stuck_after_s=45)
    positions = lambda t: {  # noqa: E731
        "walker": (0.1 * t, 0.0),  # slow but progressing
        "arrived": (5.0, 5.0),
        "waiting": (7.0, 7.0),  # not passed in targets: not moving
        "queueing": (9.0, 9.0),
    }
    targets = {"walker": (50.0, 0.0), "arrived": (5.5, 5.0), "queueing": (20.0, 9.0)}
    assert _run(w, 120, positions, targets, exempt={"queueing"}) == []


def test_unknown_target_counts_as_somewhere_to_go():
    w = ProgressWatchdog(stuck_after_s=10)
    assert _run(w, 10, lambda t: {"a": (1.0, 1.0)}, {"a": None}) == [(10, "a")]


def test_sidestep_prefers_sideways():
    first = sidestep_candidates((0.0, 0.0), (10.0, 0.0))[0]
    assert math.isclose(first[0], 0.0, abs_tol=1e-9) and math.isclose(abs(first[1]), 1.0)
