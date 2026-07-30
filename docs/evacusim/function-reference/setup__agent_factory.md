# Function Reference: evacusim/setup/agent_factory.py

- Source file: [evacusim/setup/agent_factory.py](../../../evacusim/setup/agent_factory.py)
- Top-level classes: 1
- Top-level functions: 0

## Module Summary

Agent factory for creating agent configurations in Station Concordia simulations.

## Classes and Methods

### Class AgentFactory

- Location: [evacusim/setup/agent_factory.py#L18](../../../evacusim/setup/agent_factory.py#L18)
- Summary: Handles creation of agent configurations.
- Method count: 4

#### create_agents

- Signature: def create_agents(num_agents, config, seed=42) -> tuple[list[dict], set[int]]
- Location: [evacusim/setup/agent_factory.py#L22](../../../evacusim/setup/agent_factory.py#L22)
- Summary: Create agent configurations with randomized attributes.
- Key calls: random.seed, AgentFactory._determine_injured_agents, config.get, range, logger.info, AgentFactory._select_knowledge_profile, AgentFactory._create_single_agent, agents_config.append, len

#### _determine_injured_agents

- Signature: def _determine_injured_agents(num_agents, config) -> set[int]
- Location: [evacusim/setup/agent_factory.py#L63](../../../evacusim/setup/agent_factory.py#L63)
- Summary: Determine which agents should be injured.
- Key calls: set

#### _create_single_agent

- Signature: def _create_single_agent(agent_index, is_injured, knowledge_profile, agents_section) -> dict
- Location: [evacusim/setup/agent_factory.py#L77](../../../evacusim/setup/agent_factory.py#L77)
- Summary: Create a single agent configuration with randomized attributes.
- Key calls: random.choice, agents_section.get, max, int, random.randint, list, age_cfg.get, PERSONALITY_TYPES.keys, values, role_cfg.get

#### _select_knowledge_profile

- Signature: def _select_knowledge_profile(knowledge_distribution) -> str
- Location: [evacusim/setup/agent_factory.py#L135](../../../evacusim/setup/agent_factory.py#L135)
- Summary: Sample a knowledge profile according to configured distribution.
- Key calls: list, knowledge_distribution.keys, float, random.choices

## Coverage Notes

- Nested helper functions documented in this file: 0
