"""Regression tests for escalator stall-deferral corridor/zone distinction.

An agent already inside an escalator's physical corridor has no realistic
alternative route (reversing against the flow of a moving escalator isn't an
option), unlike one still in the departure zone approaching it, who can
legitimately switch to a different escalator. `_should_defer_escalator_decision`
must never report "allowing re-decision" for the former — see
decision_processor.py's `_should_defer_escalator_decision` and
`_agent_is_on_escalator`.
"""

from types import SimpleNamespace

from shapely.geometry import Polygon

from evacusim.decision.decision_processor import DecisionProcessor
from evacusim.jps.escalator_controller import EscalatorEndpoint, EscalatorRegistry

LEVEL = "-1"
CORRIDOR_POLY = Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])
ZONE_POLY = Polygon([(10, 10), (12, 10), (12, 12), (10, 12)])


def _processor():
    processor = DecisionProcessor.__new__(DecisionProcessor)
    processor._escalator_deferral_timeout_secs = 12.0
    processor._escalator_progress_threshold_m = 0.4
    processor._escalator_deferral_state = {}
    return processor


def test_should_defer_returns_true_and_logs_but_keeps_deferring_in_corridor():
    processor = _processor()
    # First call establishes the baseline (always defers).
    assert processor._should_defer_escalator_decision("agent1", 0.0, (1.0, 1.0), True) is True
    # Same position, timeout elapsed -> stalled while in_corridor=True.
    result = processor._should_defer_escalator_decision("agent1", 20.0, (1.0, 1.0), True)
    assert result is True


def test_should_defer_allows_redecision_after_stall_outside_corridor():
    processor = _processor()
    assert processor._should_defer_escalator_decision("agent2", 0.0, (11.0, 11.0), False) is True
    # Same position, timeout elapsed -> stalled while in_corridor=False.
    result = processor._should_defer_escalator_decision("agent2", 20.0, (11.0, 11.0), False)
    assert result is False


def test_should_defer_true_while_still_within_timeout_regardless_of_corridor():
    processor = _processor()
    assert processor._should_defer_escalator_decision("agent3", 0.0, (1.0, 1.0), False) is True
    # Not stalled long enough yet (< 12s) -> still defers even outside the corridor.
    result = processor._should_defer_escalator_decision("agent3", 5.0, (1.0, 1.0), False)
    assert result is True


class _FakeGeometryManager:
    def __init__(self, corridors):
        self.escalator_corridors = corridors


class _FakeLevelSim:
    def __init__(self, corridors):
        self.geometry_manager = _FakeGeometryManager(corridors)


def _make_jps_sim(agent_positions, corridors):
    endpoint = EscalatorEndpoint(
        level_id=LEVEL,
        endpoint_id="ep_departure_a",
        role="departure",
        transfer_zone_name="esc.a.zone.platform.departure",
        transfer_zone_polygon=ZONE_POLY,
        corridor_name="corridor_a",
        exit_name="escalator_a_up",
        transfer_to_zone_name="esc.a.zone.concourse.arrival",
        spawn_point=(1.0, 1.0),
        egress_target=(1.0, 1.0),
    )
    registry = EscalatorRegistry(
        endpoints_by_zone={"esc.a.zone.platform.departure": endpoint},
        endpoints_by_level={LEVEL: [endpoint]},
        corridor_endpoints_by_level={(LEVEL, "corridor_a"): [endpoint]},
        edges=[],
        edges_by_exit={},
    )
    level_sim = _FakeLevelSim(corridors)
    return SimpleNamespace(
        get_agent_position=lambda agent_id: agent_positions.get(agent_id),
        agent_levels={agent_id: LEVEL for agent_id in agent_positions},
        simulations={LEVEL: level_sim},
        escalator_controller=SimpleNamespace(
            registry=registry,
            get_zone_polygon=lambda zone_name: ZONE_POLY,
        ),
    )


def test_agent_is_on_escalator_distinguishes_corridor_from_zone():
    processor = _processor()
    processor.jps_sim = _make_jps_sim(
        {"corridor_agent": (1.0, 1.0), "zone_agent": (11.0, 11.0), "elsewhere_agent": (50.0, 50.0)},
        corridors={"corridor_a": CORRIDOR_POLY},
    )

    assert processor._agent_is_on_escalator("corridor_agent") == (True, True)
    assert processor._agent_is_on_escalator("zone_agent") == (True, False)
    assert processor._agent_is_on_escalator("elsewhere_agent") == (False, False)
