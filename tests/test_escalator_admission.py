"""Regression tests for the escalator admission gate and re-entry backoff.

These cover the fix for a deadlock where agents denied entry to a saturated
escalator landing zone were bounced back to the corridor mouth and instantly
re-routed into the same escalator, compounding the jam until no agent could
even reach the transfer trigger anymore (see EscalatorController.try_admit,
MultiLevelJuPedSimulation._drain_escalator_admission_queues, and
EscalatorController.record_transfer_failure).
"""

from shapely.geometry import Polygon

from evacusim.jps.escalator_controller import (
    EscalatorController,
    EscalatorEdge,
    EscalatorEndpoint,
    EscalatorRegistry,
)
from evacusim.jps.multi_level_simulation import MultiLevelJuPedSimulation

FROM_LEVEL = "-1"
TO_LEVEL = "0"
EXIT_NAME = "escalator_a_up"
ZONE_NAME = "esc.a.zone.concourse.arrival"


class _FakeLevelSim:
    def __init__(self, positions=None):
        self._positions = dict(positions or {})
        self.agent_assigned_exits: dict[str, str] = {}
        self.exit_manager = type(
            "ExitManager", (), {"evacuation_exits": {EXIT_NAME: object()}}
        )()
        self.routed: list[tuple[str, str]] = []

    def get_all_agent_positions(self):
        return dict(self._positions)

    def set_agent_destination_exit(self, agent_id, exit_name):
        self.agent_assigned_exits[agent_id] = exit_name
        self.routed.append((agent_id, exit_name))


def _make_controller(
    zone_occupancy_ceiling=3,
    admission_rate_per_sec=1.0,
    admission_burst=2.0,
    dt=0.05,
    zone_positions=None,
):
    zone_polygon = Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])
    endpoint = EscalatorEndpoint(
        level_id=FROM_LEVEL,
        endpoint_id="ep_departure_a",
        role="departure",
        transfer_zone_name="esc.a.zone.platform.departure",
        transfer_zone_polygon=zone_polygon,
        corridor_name="corridor_a",
        exit_name=EXIT_NAME,
        transfer_to_zone_name=ZONE_NAME,
        spawn_point=(1.0, 1.0),
        egress_target=(1.0, 1.0),
    )
    edge = EscalatorEdge(
        from_endpoint_id="ep_departure_a",
        to_endpoint_id="ep_arrival_a",
        from_level=FROM_LEVEL,
        to_level=TO_LEVEL,
        from_exit_name=EXIT_NAME,
        from_zone_name="esc.a.zone.platform.departure",
        to_zone_name=ZONE_NAME,
        from_corridor_name="corridor_a",
        to_corridor_name="corridor_a",
        to_spawn_point=(1.0, 1.0),
        to_egress_target=(1.0, 1.0),
    )
    registry = EscalatorRegistry(
        endpoints_by_zone={
            "esc.a.zone.platform.departure": endpoint,
            ZONE_NAME: endpoint,
        },
        endpoints_by_level={FROM_LEVEL: [endpoint]},
        corridor_endpoints_by_level={},
        edges=[edge],
        edges_by_exit={(FROM_LEVEL, EXIT_NAME): edge},
    )

    controller = EscalatorController.__new__(EscalatorController)
    controller.simulations = {TO_LEVEL: _FakeLevelSim(zone_positions)}
    controller.registry = registry
    controller.dt = dt
    controller.admission_rate_per_sec = admission_rate_per_sec
    controller.admission_burst = admission_burst
    controller.zone_occupancy_ceiling = zone_occupancy_ceiling
    controller._admission_tokens = {}
    controller._admission_last_step = {}
    controller.admission_queues = {}
    controller._reentry_backoff = {}
    controller._agent_motion_context = {}
    controller.agent_states = {}
    return controller, endpoint


def test_try_admit_denies_when_zone_occupancy_at_ceiling():
    # Zone polygon spans (0,0)-(2,2); two agents already sit inside it.
    controller, _ = _make_controller(
        zone_occupancy_ceiling=2,
        zone_positions={"a": (1.0, 1.0), "b": (1.1, 1.1)},
    )
    assert controller.try_admit(FROM_LEVEL, EXIT_NAME, current_step=0) is False


def test_try_admit_rate_limits_and_replenishes_over_time():
    controller, _ = _make_controller(
        zone_occupancy_ceiling=100,  # not the limiting factor here
        admission_rate_per_sec=1.0,
        admission_burst=1.0,
        dt=0.05,
    )
    # First admission consumes the only token.
    assert controller.try_admit(FROM_LEVEL, EXIT_NAME, current_step=0) is True
    # No time has passed — no new tokens.
    assert controller.try_admit(FROM_LEVEL, EXIT_NAME, current_step=0) is False
    # 20 steps * 0.05s = 1.0s elapsed -> exactly one token regenerated.
    assert controller.try_admit(FROM_LEVEL, EXIT_NAME, current_step=20) is True


def test_denied_agent_is_queued_and_admitted_once_capacity_frees():
    controller, _ = _make_controller(
        zone_occupancy_ceiling=100,
        admission_rate_per_sec=1.0,
        admission_burst=1.0,
        dt=0.05,
    )
    # Exhaust the token so a fresh admission attempt is denied and queued.
    assert controller.try_admit(FROM_LEVEL, EXIT_NAME, current_step=0) is True
    controller.enqueue_admission(FROM_LEVEL, EXIT_NAME, "agent1")

    source_sim = _FakeLevelSim()
    simulation = MultiLevelJuPedSimulation.__new__(MultiLevelJuPedSimulation)
    simulation.escalator_controller = controller
    simulation.simulations = {FROM_LEVEL: source_sim}
    simulation.agent_levels = {"agent1": FROM_LEVEL}
    simulation.current_step = 0

    # Not enough time has passed yet -> still queued, not routed.
    simulation._drain_escalator_admission_queues()
    assert source_sim.routed == []
    assert controller.admission_queues[(FROM_LEVEL, EXIT_NAME)] == ["agent1"]

    # Advance past one full token-replenishment interval.
    simulation.current_step = 20
    simulation._drain_escalator_admission_queues()
    assert source_sim.routed == [("agent1", EXIT_NAME)]
    assert controller.admission_queues[(FROM_LEVEL, EXIT_NAME)] == []


def test_reentry_backoff_blocks_immediate_reissue_after_failed_transfer():
    controller, endpoint = _make_controller(
        zone_occupancy_ceiling=100,
        admission_rate_per_sec=1.0,
        admission_burst=2.0,
    )
    level_sim = controller.simulations[TO_LEVEL]
    # Re-key the fake sim to the departure (source) level for this endpoint.
    controller.simulations[FROM_LEVEL] = level_sim

    controller.record_transfer_failure("agent1", EXIT_NAME, until_step=100)

    # Still inside the backoff window -> no reissue.
    controller._apply_endpoint_motion("agent1", endpoint, level_sim, current_step=50)
    assert level_sim.routed == []

    # Past the backoff window -> normal admission logic applies and routes.
    controller._apply_endpoint_motion("agent1", endpoint, level_sim, current_step=150)
    assert level_sim.routed == [("agent1", EXIT_NAME)]
