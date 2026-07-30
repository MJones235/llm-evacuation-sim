# Function Reference: evacusim/coordination/observation_coordinator.py

- Source file: [evacusim/coordination/observation_coordinator.py](../../../evacusim/coordination/observation_coordinator.py)
- Top-level classes: 1
- Top-level functions: 0

## Module Summary

Observation Coordinator Coordinates the generation of observations for all agents by gathering information from: - Agent positions and nearby agents (via state queries) - Recent events (via event manager) - Received messages and conversation history (via message system) - Agent destinations (for enriching nearby agent info) This module acts as the glue between multiple systems to create comprehensive observations for agent decision-making.

## Classes and Methods

### Class ObservationCoordinator

- Location: [evacusim/coordination/observation_coordinator.py#L21](../../../evacusim/coordination/observation_coordinator.py#L21)
- Summary: Coordinates observation generation for all agents.
- Method count: 2

#### __init__

- Signature: def __init__(self, concordia_agents, exited_agents, observation_generator, state_queries, event_manager, message_system, agent_destinations, agent_injured, agent_action, agent_last_decision, jps_sim=None, agent_roles=None)
- Location: [evacusim/coordination/observation_coordinator.py#L24](../../../evacusim/coordination/observation_coordinator.py#L24)
- Summary: Initialize observation coordinator.

#### generate_all_observations

- Signature: def generate_all_observations(self, current_sim_time) -> dict[str, str]
- Location: [evacusim/coordination/observation_coordinator.py#L74](../../../evacusim/coordination/observation_coordinator.py#L74)
- Summary: Generate observations for all agents based on simulation state.
- Key calls: set, getattr, self.concordia_agents.keys, hasattr, self.jps_sim.simulations.values, jps_sim_for_bulk.get_all_nearby_agents_bulk, all_train_exits.update, self.state_queries.get_agent_position, self.state_queries.get_recent_events, self.message_system.get_received_messages

## Coverage Notes

- Nested helper functions documented in this file: 0
