"""
Agent builder for creating and initializing Concordia agents.

This module handles the creation of Concordia agents with their memory banks
and initial knowledge state.
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from concordia.associative_memory import basic_associative_memory
from concordia.language_model import language_model
from concordia.typing import entity as entity_lib

from evacusim.utils.logger import get_logger
from evacusim.concordia.evacuation_agent import EvacuationAgent

logger = get_logger(__name__)


class AgentBuilder:
    """Builds Concordia agents with configured memory and initial knowledge."""

    def __init__(
        self,
        language_model: language_model.LanguageModel,
        embedder: Any,
        station_layout: dict[str, Any],
    ):
        """
        Initialize the agent builder.

        Args:
            language_model: LLM for agent cognition
            embedder: Sentence embedding function
            station_layout: Layout/config data containing authored knowledge packs
        """
        self.model = language_model
        self.embedder = embedder
        self.station_layout = station_layout

    async def build_agents(
        self, agents_config: list[dict[str, Any]]
    ) -> tuple[dict[str, entity_lib.Entity], set[str]]:
        """
        Build Concordia agents from configurations in parallel.

        Args:
            agents_config: List of agent configuration dictionaries

        Returns:
            Tuple of (concordia_agents dict, injured_agents set)
        """
        logger.info(f"Building {len(agents_config)} Concordia agents in parallel...")

        # Create a larger thread pool to handle many agents efficiently
        # Use min(32, agent_count) to avoid creating too many threads
        max_workers = min(32, len(agents_config))

        # Build all agents concurrently with custom executor
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            tasks = [
                self._build_single_agent(agent_config, executor) for agent_config in agents_config
            ]

            results = await asyncio.gather(*tasks, return_exceptions=True)

        # Collect results
        concordia_agents: dict[str, entity_lib.Entity] = {}
        injured_agents: set[str] = set()

        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Failed to build agent {agents_config[i]['id']}: {result}")
                continue

            agent, agent_id, is_injured = result
            concordia_agents[agent_id] = agent
            if is_injured:
                injured_agents.add(agent_id)

        logger.info(
            f"Built {len(concordia_agents)} Concordia agents ({len(injured_agents)} injured)"
        )
        return concordia_agents, injured_agents

    async def _build_single_agent(
        self, agent_config: dict[str, Any], executor: ThreadPoolExecutor
    ) -> tuple[entity_lib.Entity, str, bool]:
        """
        Build a single Concordia agent (async for parallel execution).

        Args:
            agent_config: Agent configuration dictionary
            executor: Thread pool executor to use

        Returns:
            Tuple of (agent, agent_id, is_injured)
        """
        agent_id = agent_config["id"]

        # Create separate memory bank for each agent
        memory_bank = basic_associative_memory.AssociativeMemoryBank(
            sentence_embedder=self.embedder
        )

        # Create agent prefab
        prefab = EvacuationAgent(params=agent_config)

        # Build agent using custom executor
        loop = asyncio.get_event_loop()
        agent = await loop.run_in_executor(
            executor,
            prefab.build,
            self.model,
            memory_bank,
        )

        # Add initial memories using custom executor
        await loop.run_in_executor(
            executor,
            self._initialize_agent_memory,
            agent,
            agent_config,
        )

        is_injured = agent_config.get("is_injured", False)

        return agent, agent_id, is_injured

    def _initialize_agent_memory(self, agent: entity_lib.Entity, config: dict[str, Any]) -> None:
        """
        Initialize an agent's memory with background knowledge.

        Args:
            agent: Concordia agent entity
            config: Agent configuration dictionary
        """
        knowledge_config = self.station_layout["knowledge"]
        profile_name = config["knowledge_profile"]
        level_id = str(config.get("level_id", "0"))
        initial_zone = config.get("initial_zone", "platform_def")

        base_memories = knowledge_config["base_memories"]
        profile_memories = knowledge_config["profiles"][profile_name]
        location_memories = self._select_location_memories(
            knowledge_config=knowledge_config,
            profile_name=profile_name,
            level_id=level_id,
            initial_zone=initial_zone,
        )

        # Zone labels are defined in config under station.zone_labels.
        _zone_labels = self.station_layout.get("zone_labels", {})
        zone_label = _zone_labels.get(initial_zone, f"the {initial_zone} area")

        # Order memories so high-level purpose appears first in recent observations,
        # followed by current location, then platform-navigation specifics.
        purpose_memories = self._build_purpose_memories(config)
        initial_memories = [
            *base_memories,
            *profile_memories,
            *purpose_memories,
            f"I am currently in {zone_label}.",
            *location_memories,
        ]

        # Add injury-specific memories
        if config.get("is_injured", False):
            initial_memories.extend(
                [
                    "I am injured and moving slowly.",
                    "I may need assistance.",
                ]
            )

        for memory in initial_memories:
            agent.observe(memory)

    @staticmethod
    def _build_purpose_memories(config: dict[str, Any]) -> list[str]:
        """Return pre-computed purpose memories stored in the agent config."""
        return config.get("purpose_memories", [])

    @staticmethod
    def _select_location_memories(
        knowledge_config: dict[str, Any],
        profile_name: str,
        level_id: str,
        initial_zone: str,
    ) -> list[str]:
        """Select authored location-specific memories matching the agent context."""
        location_rules = knowledge_config.get("location_memories", [])
        if not isinstance(location_rules, list):
            return []

        selected: list[str] = []
        for rule in location_rules:
            if not isinstance(rule, dict):
                continue

            condition = rule.get("when", {})
            if not isinstance(condition, dict):
                continue

            profiles = condition.get("profiles")
            levels = condition.get("level_ids")
            zones = condition.get("zones")

            profile_match = not profiles or profile_name in profiles
            level_match = not levels or level_id in {str(level) for level in levels}
            zone_match = not zones or initial_zone in zones

            if profile_match and level_match and zone_match:
                memories = rule.get("memories", [])
                if isinstance(memories, list):
                    selected.extend(
                        memory for memory in memories if isinstance(memory, str) and memory.strip()
                    )

        return selected
