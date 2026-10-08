from types import SimpleNamespace

from shapely.geometry import Point

from evacusim.decision.situation import GoalTracker
from evacusim.decision.transfer_routing import PostTransferRouting
from evacusim.jps.jupedsim_integration import ConcordiaJuPedSimulation


def _routing(simulation, agent_configs, zones=None, goals=None):
    tracker = GoalTracker(agent_configs)
    tracker.goals.update(goals or {})
    return PostTransferRouting(simulation, zones or {}, agent_configs, tracker)


def test_reaching_egress_distributes_agents_across_assigned_platform():
    platform = Point(-50.0, 35.0).buffer(5.0)
    simulation = SimpleNamespace(
        transfer_escape_waypoints={
            "passenger_1": (-31.80, 40.56),
            "passenger_2": (-31.80, 40.56),
        },
        transfer_platform_waypoints={},
        target=None,
        simulations={},
    )
    simulation.set_agent_target = lambda agent_id, target: setattr(simulation, "target", target)
    simulation.get_agent_level = lambda agent_id: "-1"
    routing = _routing(
        simulation,
        {
            "passenger_1": {"target": "train_platform_4"},
            "passenger_2": {"target": "train_platform_4"},
        },
        {"platform_4": platform},
    )

    for agent_id in ("passenger_1", "passenger_2"):
        assert routing.should_defer(agent_id, (-24.87, 40.56))

    waypoints = simulation.transfer_platform_waypoints
    assert waypoints["passenger_1"] != waypoints["passenger_2"]
    assert all(platform.buffer(-0.3).contains(Point(point)) for point in waypoints.values())
    assert all(Point(point).distance(Point(-24.87, 40.56)) <= 30.0 for point in waypoints.values())
    assert simulation.transfer_escape_waypoints == {}


def test_concourse_transfer_requests_exit_choice_without_stopping():
    simulation = SimpleNamespace(
        transfer_escape_waypoints={"alighter": (42.2, 38.42)},
        transfer_platform_waypoints={},
        routed_exit=None,
        simulations={},
    )
    simulation.set_agent_destination_exit = lambda agent_id, exit_id: setattr(
        simulation, "routed_exit", exit_id
    )
    simulation.get_agent_level = lambda agent_id: "0"
    routing = _routing(
        simulation, {"alighter": {"target": ""}}, goals={"alighter": "Leave the station."}
    )

    # The alighter decides now (to choose a street exit) rather than walking
    # to the egress waypoint first.
    assert not routing.should_defer("alighter", (29.24, 38.48))
    assert simulation.routed_exit is None
    assert simulation.transfer_escape_waypoints == {}


def test_boarded_agent_is_not_reclassified_as_escalator_exit():
    tracker = SimpleNamespace(
        agent_ids={"boarder": 1},
        agent_targets={"boarder": (-7.0, 17.0)},
        get_all_positions=lambda: {},
        remove_agent=lambda agent_id: None,
    )
    simulation = ConcordiaJuPedSimulation.__new__(ConcordiaJuPedSimulation)
    simulation.level_id = "-1"
    simulation.agent_tracker = tracker
    simulation.last_known_positions = {"boarder": (-7.0, 17.0)}
    simulation.agent_assigned_exits = {"boarder": "train_platform_1"}
    simulation.exit_manager = SimpleNamespace(
        exit_coordinates={"escalator_e_up": (-5.0, 16.0)},
        evacuation_exits={"escalator_e_up": object()},
    )

    assert simulation.check_exits() == {"boarder": "train_platform_1"}
