# Function Reference: evacusim/concordia/agent_builder.py

- Source file: [evacusim/concordia/agent_builder.py](../../../evacusim/concordia/agent_builder.py)
- Top-level classes: 1
- Top-level functions: 0

## Module Summary

Agent builder for creating and initializing Concordia agents.

## Classes and Methods

### Class AgentBuilder

- Location: [evacusim/concordia/agent_builder.py#L22](../../../evacusim/concordia/agent_builder.py#L22)
- Summary: Builds Concordia agents with configured memory and initial knowledge.
- Method count: 6

#### __init__

- Signature: def __init__(self, language_model, embedder, station_layout)
- Location: [evacusim/concordia/agent_builder.py#L25](../../../evacusim/concordia/agent_builder.py#L25)
- Summary: Initialize the agent builder.

#### build_agents

- Signature: async def build_agents(self, agents_config) -> tuple[dict[str, entity_lib.Entity], set[str]]
- Location: [evacusim/concordia/agent_builder.py#L43](../../../evacusim/concordia/agent_builder.py#L43)
- Summary: Build Concordia agents from configurations in parallel.
- Key calls: logger.info, min, set, enumerate, len, ThreadPoolExecutor, isinstance, self._build_single_agent, asyncio.gather, logger.error

#### _build_single_agent

- Signature: async def _build_single_agent(self, agent_config, executor) -> tuple[entity_lib.Entity, str, bool]
- Location: [evacusim/concordia/agent_builder.py#L88](../../../evacusim/concordia/agent_builder.py#L88)
- Summary: Build a single Concordia agent (async for parallel execution).
- Key calls: basic_associative_memory.AssociativeMemoryBank, EvacuationAgent, asyncio.get_event_loop, agent_config.get, loop.run_in_executor

#### _initialize_agent_memory

- Signature: def _initialize_agent_memory(self, agent, config) -> None
- Location: [evacusim/concordia/agent_builder.py#L132](../../../evacusim/concordia/agent_builder.py#L132)
- Summary: Initialize an agent's memory with background knowledge.
- Key calls: str, config.get, self._select_location_memories, self.station_layout.get, _zone_labels.get, self._build_purpose_memories, initial_memories.extend, agent.observe

#### _build_purpose_memories

- Signature: def _build_purpose_memories(config) -> list[str]
- Location: [evacusim/concordia/agent_builder.py#L182](../../../evacusim/concordia/agent_builder.py#L182)
- Summary: Return pre-computed purpose memories stored in the agent config.
- Key calls: config.get

#### _select_location_memories

- Signature: def _select_location_memories(knowledge_config, profile_name, level_id, initial_zone) -> list[str]
- Location: [evacusim/concordia/agent_builder.py#L187](../../../evacusim/concordia/agent_builder.py#L187)
- Summary: Select authored location-specific memories matching the agent context.
- Key calls: knowledge_config.get, isinstance, rule.get, condition.get, selected.extend, str, memory.strip

## Coverage Notes

- Nested helper functions documented in this file: 0
