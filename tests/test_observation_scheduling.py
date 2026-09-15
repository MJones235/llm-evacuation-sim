from types import SimpleNamespace

from evacusim.coordination.observation_coordinator import ObservationCoordinator


class _ObservationGenerator:
    def __init__(self):
        self.generated_for = []

    def generate_observation(self, **kwargs):
        self.generated_for.append(kwargs["agent_id"])
        return f"observation for {kwargs['agent_id']}"


def test_targeted_cycle_formats_only_selected_agent_observations():
    generator = _ObservationGenerator()
    coordinator = ObservationCoordinator(
        concordia_agents={"selected": object(), "other": object()},
        exited_agents=set(),
        observation_generator=generator,
        state_queries=SimpleNamespace(
            get_agent_position=lambda agent_id: (0.0, 0.0),
            get_recent_events=lambda events, current_time: [],
        ),
        event_manager=SimpleNamespace(event_history=[], active_train_exits=set(), blocked_exits=set()),
        message_system=SimpleNamespace(
            get_received_messages=lambda agent_id: [],
            get_conversation_history=lambda agent_id: [],
        ),
        agent_destinations={},
        agent_injured=set(),
        agent_action={},
        agent_last_decision={},
        jps_sim=SimpleNamespace(
            simulations={},
            agent_levels={"selected": "0", "other": "0"},
            get_all_nearby_agents_bulk=lambda radius: {
                "selected": [{"id": "other"}],
                "other": [{"id": "selected"}],
            },
        ),
    )

    observations = coordinator.generate_all_observations(10.0, agent_ids=["selected"])

    assert observations == {"selected": "observation for selected"}
    assert generator.generated_for == ["selected"]