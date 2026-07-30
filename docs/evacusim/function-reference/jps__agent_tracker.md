# Function Reference: evacusim/jps/agent_tracker.py

- Source file: [evacusim/jps/agent_tracker.py](../../../evacusim/jps/agent_tracker.py)
- Top-level classes: 1
- Top-level functions: 0

## Module Summary

Agent tracking and management for JuPedSim simulation.

## Classes and Methods

### Class AgentTracker

- Location: [evacusim/jps/agent_tracker.py#L16](../../../evacusim/jps/agent_tracker.py#L16)
- Summary: Tracks agent state and provides spatial queries.
- Method count: 11

#### __init__

- Signature: def __init__(self, simulation)
- Location: [evacusim/jps/agent_tracker.py#L27](../../../evacusim/jps/agent_tracker.py#L27)
- Summary: Initialize agent tracker.

#### add_agent

- Signature: def add_agent(self, agent_id, jps_id) -> None
- Location: [evacusim/jps/agent_tracker.py#L41](../../../evacusim/jps/agent_tracker.py#L41)
- Summary: Register an agent in the tracker.
- Key calls: logger.debug

#### set_target

- Signature: def set_target(self, agent_id, target) -> None
- Location: [evacusim/jps/agent_tracker.py#L58](../../../evacusim/jps/agent_tracker.py#L58)
- Summary: Record agent's target position.

#### get_position

- Signature: def get_position(self, agent_id) -> tuple[float, float] | None
- Location: [evacusim/jps/agent_tracker.py#L68](../../../evacusim/jps/agent_tracker.py#L68)
- Summary: Get agent's current position.
- Key calls: self.simulation.agents, float

#### get_all_positions

- Signature: def get_all_positions(self) -> dict[str, tuple[float, float]]
- Location: [evacusim/jps/agent_tracker.py#L92](../../../evacusim/jps/agent_tracker.py#L92)
- Summary: Get positions of all agents for visualization.
- Key calls: self.simulation.agents, self.jps_to_concordia.get, float

#### get_nearby_agents

- Signature: def get_nearby_agents(self, agent_id, radius) -> list[dict[str, Any]]
- Location: [evacusim/jps/agent_tracker.py#L111](../../../evacusim/jps/agent_tracker.py#L111)
- Summary: Get information about agents within radius of given agent.
- Key calls: self.get_position, list, self.simulation.agents, self.jps_to_concordia.get, nearby.append, hasattr, float

#### get_all_nearby_agents_bulk

- Signature: def get_all_nearby_agents_bulk(self, radius) -> dict[str, list[dict[str, Any]]]
- Location: [evacusim/jps/agent_tracker.py#L161](../../../evacusim/jps/agent_tracker.py#L161)
- Summary: Compute nearby agents for ALL agents in a single O(n²/2) pass.
- Key calls: list, np.array, len, range, self.simulation.agents, self.jps_to_concordia.get, concordia_ids.append, positions.append, is_moving_flags.append, float

#### is_agent_active

- Signature: def is_agent_active(self, agent_id) -> bool
- Location: [evacusim/jps/agent_tracker.py#L231](../../../evacusim/jps/agent_tracker.py#L231)
- Summary: Check if agent is still in simulation.
- Key calls: self.simulation.agents

#### get_jps_id

- Signature: def get_jps_id(self, agent_id) -> int | None
- Location: [evacusim/jps/agent_tracker.py#L253](../../../evacusim/jps/agent_tracker.py#L253)
- Summary: Get JuPedSim ID for a Concordia agent.
- Key calls: self.agent_ids.get

#### get_concordia_id

- Signature: def get_concordia_id(self, jps_id) -> str | None
- Location: [evacusim/jps/agent_tracker.py#L265](../../../evacusim/jps/agent_tracker.py#L265)
- Summary: Get Concordia ID for a JuPedSim agent.
- Key calls: self.jps_to_concordia.get

#### remove_agent

- Signature: def remove_agent(self, agent_id) -> None
- Location: [evacusim/jps/agent_tracker.py#L277](../../../evacusim/jps/agent_tracker.py#L277)
- Summary: Remove an agent from tracking after they exit simulation.
- Key calls: self.agent_ids.pop, self.agent_targets.pop, self.jps_to_concordia.pop

## Coverage Notes

- Nested helper functions documented in this file: 0
