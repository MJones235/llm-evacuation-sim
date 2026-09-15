from types import SimpleNamespace

from shapely.geometry import Point

from evacusim.decision.decision_processor import DecisionProcessor
from evacusim.jps.jupedsim_integration import ConcordiaJuPedSimulation
from evacusim.jps.multi_level_simulation import MultiLevelJuPedSimulation


class _TargetLevelSimulation:
    def __init__(self):
        self.target = None

    def get_all_agent_positions(self):
        return {}

    def add_agent(self, agent_id, position, assign_default_destination):
        self.agent_id = agent_id
        self.position = position

    def set_agent_target(self, agent_id, target):
        self.target = target


class _EscalatorController:
    def __init__(self, edge):
        self.edge = edge

    def get_edge_for_exit(self, current_level, exit_name):
        return self.edge

    def get_zone_polygon(self, zone_name):
        return Point(self.edge.to_spawn_point).buffer(1.0)

    def get_spawn_point_for_edge(self, edge):
        return edge.to_spawn_point


def test_transfer_uses_local_escalator_egress_waypoint():
    edge = SimpleNamespace(
        to_level="-1",
        to_zone_name="esc.A.zone.platform.arrival",
        to_spawn_point=(-24.87, 40.56),
        to_egress_target=(-31.80, 40.56),
    )
    target_sim = _TargetLevelSimulation()
    simulation = MultiLevelJuPedSimulation.__new__(MultiLevelJuPedSimulation)
    simulation.simulations = {"-1": target_sim}
    simulation.escalator_controller = _EscalatorController(edge)
    simulation._pending_spawn_positions = []
    simulation.agent_levels = {}
    simulation.recently_transferred_agents = set()
    simulation._last_transfer_step = {}
    simulation.current_step = 10
    simulation.transfer_escape_waypoints = {}
    simulation.transfer_platform_waypoints = {}
    simulation.transfer_exit_destinations = {}

    simulation._transfer_agent_through_escalator(
        "passenger", "0", "escalator_a_down"
    )

    assert target_sim.target == (-31.80, 40.56)
    assert simulation.transfer_escape_waypoints["passenger"] == (-31.80, 40.56)


def _processor(simulation, agent_configs, zones=None):
    processor = DecisionProcessor.__new__(DecisionProcessor)
    processor.jps_sim = simulation
    processor._agent_cfg = agent_configs
    processor.action_translator = SimpleNamespace(zones_polygons=zones or {})
    processor.station_layout = {"street_exits": ["grey_street", "blackett_street"]}
    processor.agent_destinations = {}
    processor._deferred_escalator_agents = set()
    return processor


def test_reaching_egress_distributes_agents_across_assigned_platform():
    platform = Point(-50.0, 35.0).buffer(5.0)
    simulation = SimpleNamespace(
        transfer_escape_waypoints={
            "passenger_1": (-31.80, 40.56),
            "passenger_2": (-31.80, 40.56),
        },
        transfer_platform_waypoints={},
        transfer_exit_destinations={},
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


def test_reaching_concourse_egress_continues_to_configured_exit():
    simulation = SimpleNamespace(
        transfer_escape_waypoints={"alighter": (42.2, 38.42)},
        transfer_platform_waypoints={},
        transfer_exit_destinations={},
        routed_exit=None,
        simulations={},
    )
    simulation.set_agent_destination_exit = lambda agent_id, exit_id: setattr(
        simulation, "routed_exit", exit_id
    )
    simulation.get_agent_level = lambda agent_id: "0"
    processor = _processor(simulation, {"alighter": {"target": "grey_street"}})

    assert processor._defer_for_post_transfer_route("alighter", (29.24, 38.48))
    assert simulation.routed_exit == "grey_street"
    assert simulation.transfer_exit_destinations["alighter"] == "grey_street"
    assert processor.agent_destinations["alighter"] == "grey_street"
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