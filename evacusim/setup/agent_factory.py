"""
Agent factory for creating agent configurations in Station Concordia simulations.

This module is responsible for:
- Creating agent configurations with randomized attributes
- Determining injured agents based on test scenarios
- Assigning personality types, ages, and other attributes
"""

import random

from evacusim.utils.logger import get_logger
from evacusim.utils.station_agent import OCEAN_ANCHORS, build_personality_anchor

logger = get_logger(__name__)


class AgentFactory:
    """Handles creation of agent configurations."""

    @staticmethod
    def create_agents(
        num_agents: int,
        config: dict,
        seed: int = 42,
    ) -> tuple[list[dict], set[int]]:
        """
        Create agent configurations with randomized attributes.

        Args:
            num_agents: Number of agents to create
            config: Configuration dictionary containing test scenarios
            seed: Random seed for reproducibility

        Returns:
            Tuple of (agents_config, injured_agents)
            - agents_config: List of agent configuration dictionaries
            - injured_agents: Set of agent indices that are injured
        """
        random.seed(seed)

        # Determine which agents are injured (if help scenario enabled)
        injured_agents = AgentFactory._determine_injured_agents(num_agents, config)

        # Create agent configurations
        agents_config = []
        agents_section = config.get("agents", {})
        knowledge_distribution = agents_section["knowledge_profiles"]
        for i in range(num_agents):
            profile = AgentFactory._select_knowledge_profile(knowledge_distribution)
            agent_cfg = AgentFactory._create_single_agent(
                i, i in injured_agents, profile, agents_section
            )
            agents_config.append(agent_cfg)

        logger.info(f"Created {num_agents} agent configuration(s)")
        if injured_agents:
            logger.info(f"  - {len(injured_agents)} agents marked as injured/slow-moving")

        return agents_config, injured_agents

    @staticmethod
    def _determine_injured_agents(num_agents: int, config: dict) -> set[int]:
        """
        Determine which agents should be injured.

        Args:
            num_agents: Total number of agents
            config: Configuration dictionary

        Returns:
            Set of agent indices that should be injured
        """
        return set()

    @staticmethod
    def _create_single_agent(
        agent_index: int,
        is_injured: bool,
        knowledge_profile: str,
        agents_section: dict,
    ) -> dict:
        """
        Create a single agent configuration with randomized attributes.

        Args:
            agent_index: Index of the agent (used for ID)
            is_injured: Whether this agent is injured
            knowledge_profile: Knowledge profile name
            agents_section: The ``agents`` sub-dict from the scenario config

        Returns:
            Agent configuration dictionary
        """
        agent_id = f"agent_{agent_index}"

        # Randomize agent attributes
        # Sample OCEAN personality dimensions from configured weighted distributions.
        # Falls back to uniform sampling when the config key is absent.
        personalities_cfg = agents_section.get("personalities", {})
        ocean_n = AgentFactory._sample_ocean_level(personalities_cfg.get("N", {}))
        ocean_o = AgentFactory._sample_ocean_level(personalities_cfg.get("O", {}))
        ocean_c = AgentFactory._sample_ocean_level(personalities_cfg.get("C", {}))
        personality_anchor = build_personality_anchor(ocean_n, ocean_o, ocean_c)
        # Compact summary used in analytics output (e.g. wait_behavior.txt)
        personality_type = f"N:{ocean_n}/O:{ocean_o}/C:{ocean_c}"

        age_cfg = agents_section.get("age", {})
        age_min = max(18, int(age_cfg.get("min", 18)))
        age_max = int(age_cfg.get("max", 75))
        age = random.randint(age_min, age_max)
        gender = random.choice(["man", "woman"])
        risk_tolerance = random.choice(["low", "moderate", "high"])

        # Sample purpose from config. Target is assigned later by AgentManager
        # after role assignment so it stays role-consistent.
        purposes = agents_section.get("purposes", ["their destination"])
        purpose = random.choice(purposes)

        return {
            "id": agent_id,
            "name": f"Agent {agent_index}",
            "ocean_n": ocean_n,
            "ocean_o": ocean_o,
            "ocean_c": ocean_c,
            "personality_anchor": personality_anchor,
            "personality_type": personality_type,
            "age": age,
            "gender": gender,
            "risk_tolerance": risk_tolerance,
            "knowledge_profile": knowledge_profile,
            "initial_zone": "platform",
            "destination": "exit",
            "is_injured": is_injured,
            "purpose": purpose,
            "target": "",
            # agent_role, goal_state, purpose_memories and initial_goal are assigned
            # later by AgentManager once the spawn zone is known
        }

    @staticmethod
    def _sample_ocean_level(level_cfg: dict) -> str:
        """Sample an OCEAN trait level (high/medium/low) from a weighted config dict.

        Args:
            level_cfg: Dict mapping level name to percentage weight, e.g.
                       {"high": 20.0, "medium": 50.0, "low": 30.0}

        Returns:
            A level string: "high", "medium", or "low".
            Falls back to uniform sampling when level_cfg is empty.
        """
        _defaults = {"high": 1.0, "medium": 1.0, "low": 1.0}
        cfg = level_cfg if level_cfg else _defaults
        levels = [k for k in ("high", "medium", "low") if k in cfg]
        weights = [float(cfg[lvl]) for lvl in levels]
        return random.choices(levels, weights=weights, k=1)[0]

    @staticmethod
    def _select_knowledge_profile(knowledge_distribution: dict[str, float]) -> str:
        """Sample a knowledge profile according to configured distribution."""
        profiles = list(knowledge_distribution.keys())
        weights = [float(knowledge_distribution[p]) for p in profiles]
        return random.choices(profiles, weights=weights, k=1)[0]
