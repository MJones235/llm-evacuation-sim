"""
Custom Concordia agent prefab for evacuation scenarios.

This prefab extends Concordia's basic Entity with evacuation-specific components:
- Evacuation memory formation
- Risk perception
- Social influence observation
- Exit knowledge and mental mapping
"""

import dataclasses
from collections.abc import Mapping
from typing import Any

from concordia.agents import entity_agent_with_logging
from concordia.associative_memory import basic_associative_memory
from concordia.components import agent as agent_components
from concordia.language_model import language_model
from concordia.typing import prefab as prefab_lib


@dataclasses.dataclass
class EvacuationAgent(prefab_lib.Prefab):
    """
    A Concordia agent prefab specialized for evacuation scenarios.

    Extends the basic reasoning agent with evacuation-specific components:
    - Enhanced memory for evacuation events
    - Risk perception reasoning
    - Social influence observation
    - Exit knowledge
    """

    description: str = (
        "An evacuation agent that reasons about emergency situations using "
        "memory, personality, risk perception, and social influence."
    )

    params: Mapping[str, Any] = dataclasses.field(
        default_factory=lambda: {
            "name": "Agent",
            "ocean_n": "medium",
            "ocean_o": "medium",
            "ocean_c": "medium",
            "personality_anchor": "",
            "personality_type": "N:medium/O:medium/C:medium",
            "age": 30,
            "gender": "neutral",
            "initial_zone": "platform",
            "destination": "exit",
            "risk_tolerance": "moderate",
        }
    )

    def build(
        self,
        model: language_model.LanguageModel,
        memory_bank: basic_associative_memory.AssociativeMemoryBank,
    ) -> entity_agent_with_logging.EntityAgentWithLogging:
        """
        Build an evacuation agent with Concordia components.

        Args:
            model: The language model to use for reasoning
            memory_bank: The associative memory bank for storing experiences

        Returns:
            A configured EntityAgentWithLogging instance
        """
        # Extract parameters
        name = self.params.get("name", "Agent")
        personality_anchor = self.params.get("personality_anchor", "")
        age = self.params.get("age", 30)
        gender = self.params.get("gender", "neutral")
        risk_tolerance = self.params.get("risk_tolerance", "moderate")
        goal_state = self.params.get("goal_state", "Continue your planned journey.")

        # Core components
        memory_key = agent_components.memory.DEFAULT_MEMORY_COMPONENT_KEY
        memory = agent_components.memory.AssociativeMemory(memory_bank=memory_bank)

        # Observation to memory - automatically stores observations
        observation_to_memory_key = "ObservationToMemory"
        observation_to_memory = agent_components.observation.ObservationToMemory()

        # Recent observations
        observation_key = agent_components.observation.DEFAULT_OBSERVATION_COMPONENT_KEY
        observation = agent_components.observation.LastNObservations(
            history_length=1,
            pre_act_label="\nCurrent Situation",
        )

        # SelfPerception and Goal are intentionally omitted from components_of_agent:
        # profile (age/gender/personality) and current goal are surfaced in the
        # decision prompt template itself, so including them here would duplicate
        # that content in the Concordia context block.
        components_of_agent = {
            observation_to_memory_key: observation_to_memory,
            observation_key: observation,
            memory_key: memory,
        }

        # Component order for prompt construction.
        # Only the observation is included — profile/goal are in the decision
        # prompt template, and memory is handled separately by ObservationToMemory.
        component_order = [
            observation_key,
        ]

        # Action component - generates final action
        act_component = agent_components.concat_act_component.ConcatActComponent(
            model=model,
            component_order=component_order,
            randomize_choices=False,  # Deterministic for evacuation
            prefix_entity_name=False,  # Suppresses bare ":" / "Answer: Agent N" artifacts
        )

        # Create agent
        agent = entity_agent_with_logging.EntityAgentWithLogging(
            agent_name=name,
            act_component=act_component,
            context_components=components_of_agent,
        )

        return agent
