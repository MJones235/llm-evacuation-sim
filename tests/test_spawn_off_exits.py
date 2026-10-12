"""Passengers never start inside or next to a street exit."""

from types import SimpleNamespace

from shapely.geometry import box

from evacusim.setup.agent_manager import AgentManager


def _level(*exits):
    return SimpleNamespace(
        geometry_manager=SimpleNamespace(entrance_areas={f"e{i}": p for i, p in enumerate(exits)})
    )


def test_positions_in_or_near_an_exit_on_their_level_are_dropped():
    sim = SimpleNamespace(simulations={"0": _level(box(0, 0, 2, 2)), "-1": _level()})
    positions = [
        (1.0, 1.0, "0"),  # inside the exit
        (2.5, 1.0, "0"),  # within the clearance
        (5.0, 5.0, "0"),  # clear
        (1.0, 1.0, "-1"),  # same spot, but another level has no exit there
    ]
    assert AgentManager._filter_positions_off_exits(sim, positions) == [
        (5.0, 5.0, "0"),
        (1.0, 1.0, "-1"),
    ]


def test_single_level_simulation():
    sim = SimpleNamespace(geometry_manager=_level(box(0, 0, 2, 2)).geometry_manager)
    assert AgentManager._filter_positions_off_exits(sim, [(1.0, 1.0), (9.0, 9.0)]) == [(9.0, 9.0)]
