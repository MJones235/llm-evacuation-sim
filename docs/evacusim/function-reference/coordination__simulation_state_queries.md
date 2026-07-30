# Function Reference: evacusim/coordination/simulation_state_queries.py

- Source file: [evacusim/coordination/simulation_state_queries.py](../../../evacusim/coordination/simulation_state_queries.py)
- Top-level classes: 1
- Top-level functions: 0

## Module Summary

Simulation state query utilities.

## Classes and Methods

### Class SimulationStateQueries

- Location: [evacusim/coordination/simulation_state_queries.py#L16](../../../evacusim/coordination/simulation_state_queries.py#L16)
- Summary: Helper class for querying simulation state.
- Method count: 4

#### __init__

- Signature: def __init__(self, jps_simulation)
- Location: [evacusim/coordination/simulation_state_queries.py#L19](../../../evacusim/coordination/simulation_state_queries.py#L19)
- Summary: Initialize state queries.

#### get_agent_position

- Signature: def get_agent_position(self, agent_id) -> tuple[float, float] | None
- Location: [evacusim/coordination/simulation_state_queries.py#L28](../../../evacusim/coordination/simulation_state_queries.py#L28)
- Summary: Get agent's current position from JuPedSim.
- Key calls: self.jps_sim.get_agent_position

#### get_nearby_agents

- Signature: def get_nearby_agents(self, agent_id, radius) -> list[dict[str, Any]]
- Location: [evacusim/coordination/simulation_state_queries.py#L40](../../../evacusim/coordination/simulation_state_queries.py#L40)
- Summary: Get information about nearby agents.
- Key calls: self.jps_sim.get_nearby_agents

#### get_recent_events

- Signature: def get_recent_events(self, event_history, current_sim_time, count=3) -> list[str]
- Location: [evacusim/coordination/simulation_state_queries.py#L53](../../../evacusim/coordination/simulation_state_queries.py#L53)
- Summary: Get recent events relevant to agents.

## Coverage Notes

- Nested helper functions documented in this file: 0
