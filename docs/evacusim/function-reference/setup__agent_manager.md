# Function Reference: evacusim/setup/agent_manager.py

- Source file: [evacusim/setup/agent_manager.py](../../../evacusim/setup/agent_manager.py)
- Top-level classes: 1
- Top-level functions: 0

## Module Summary

Agent manager for Station Concordia simulations.

## Classes and Methods

### Class AgentManager

- Location: [evacusim/setup/agent_manager.py#L26](../../../evacusim/setup/agent_manager.py#L26)
- Summary: Handles complete agent lifecycle management.
- Method count: 3

#### create_and_populate_agents

- Signature: def create_and_populate_agents(jps_sim, config) -> list[dict[str, Any]]
- Location: [evacusim/setup/agent_manager.py#L30](../../../evacusim/setup/agent_manager.py#L30)
- Summary: Create agents and add them to the pedestrian simulation.
- Key calls: config.get, agent_config.get, SpawnManager.generate_spawn_positions, list, len, AgentFactory.create_agents, AgentManager._add_agents_to_jupedsim, logger.info, values, logger.warning

#### _add_agents_to_jupedsim

- Signature: def _add_agents_to_jupedsim(jps_sim, agents_config, spawn_positions, injured_agents, config)
- Location: [evacusim/setup/agent_manager.py#L102](../../../evacusim/setup/agent_manager.py#L102)
- Summary: Add agents to JuPedSim simulation with appropriate walking speeds.
- Key calls: enumerate, get, zone_boundaries.get, level_zones.items, AgentManager._assign_role, roles_config.get, format, role_cfg.get, hasattr, len

#### _assign_role

- Signature: def _assign_role(initial_zone, roles_config) -> str
- Location: [evacusim/setup/agent_manager.py#L193](../../../evacusim/setup/agent_manager.py#L193)
- Summary: Pick a role whose spawn_zones include ``initial_zone``, using configured weights.
- Key calls: list, float, random.choices, roles_config.items, cfg.get

## Coverage Notes

- Nested helper functions documented in this file: 0
