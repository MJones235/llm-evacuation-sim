from types import SimpleNamespace

from shapely.geometry import Point

from evacusim.decision.decision_processor import DecisionProcessor
from evacusim.jps.jupedsim_integration import ConcordiaJuPedSimulation


def _processor(simulation, agent_configs, zones=None):
    processor = DecisionProcessor.__new__(DecisionProcessor)
    processor.jps_sim = simulation
    processor._agent_cfg = agent_configs
    processor.action_translator = SimpleNamespace(zones_polygons=zones or {})
    processor.station_layout = {"street_exits": ["grey_street", "blackett_street"]}
    processor.agent_destinations = {}
    processor._deferred_escalator_agents = set()
    processor._post_transfer_exit_choice_agents = set()
    return processor


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
    processor = _processor(
        simulation,
        {
            "passenger_1": {"target": "train_platform_4"},
            "passenger_2": {"target": "train_platform_4"},
        },
        {"platform_4": platform},
    )

    for agent_id in ("passenger_1", "passenger_2"):
        assert processor._defer_for_post_transfer_route(agent_id, (-24.87, 40.56))

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
    processor = _processor(simulation, {"alighter": {"target": ""}})
    processor.agent_goals = {"alighter": "Leave the station."}

    assert not processor._defer_for_post_transfer_route("alighter", (29.24, 38.48))
    assert simulation.routed_exit is None
    assert processor.agent_destinations == {}
    assert simulation.transfer_escape_waypoints == {}
    assert processor._post_transfer_exit_choice_agents == {"alighter"}


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
