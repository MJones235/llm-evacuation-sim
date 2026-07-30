# Function Reference: evacusim/concordia/evacuation_agent.py

- Source file: [evacusim/concordia/evacuation_agent.py](../../../evacusim/concordia/evacuation_agent.py)
- Top-level classes: 1
- Top-level functions: 0

## Module Summary

Custom Concordia agent prefab for evacuation scenarios.

## Classes and Methods

### Class EvacuationAgent

- Location: [evacusim/concordia/evacuation_agent.py#L23](../../../evacusim/concordia/evacuation_agent.py#L23)
- Summary: A Concordia agent prefab specialized for evacuation scenarios.
- Method count: 1

#### build

- Signature: def build(self, model, memory_bank) -> entity_agent_with_logging.EntityAgentWithLogging
- Location: [evacusim/concordia/evacuation_agent.py#L51](../../../evacusim/concordia/evacuation_agent.py#L51)
- Summary: Build an evacuation agent with Concordia components.
- Key calls: self.params.get, personality_descriptions.get, agent_components.memory.AssociativeMemory, agent_components.observation.ObservationToMemory, agent_components.observation.LastNObservations, agent_components.constant.Constant, agent_components.concat_act_component.ConcatActComponent, entity_agent_with_logging.EntityAgentWithLogging

## Coverage Notes

- Nested helper functions documented in this file: 0
