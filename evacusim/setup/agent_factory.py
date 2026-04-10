"""
Agent factory for creating agent configurations in Station Concordia simulations.

This module is responsible for:
- Creating agent configurations with randomized attributes
- Determining injured agents based on test scenarios
- Assigning personality types, ages, and other attributes
"""

import random

from evacusim.utils.logger import get_logger
from evacusim.utils.station_agent import PERSONALITY_TYPES

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
        Determine which agents should be injured based on test scenario config.

        Args:
            num_agents: Total number of agents
            config: Configuration dictionary

        Returns:
            Set of agent indices that should be injured
        """
        help_config = config.get("test_scenarios", {}).get("help_behavior", {})
        injured_agents = set()

        if help_config.get("enabled", False):
            injured_percentage = help_config.get("injured_agent_percentage", 0.2)
            num_injured = max(1, int(num_agents * injured_percentage))
            injured_agents = set(random.sample(range(num_agents), num_injured))
            logger.info(f"Phase 4.1: {num_injured} agents will be injured/slow-moving")

        return injured_agents

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
        personality_type = random.choice(list(PERSONALITY_TYPES.keys()))
        age = random.randint(16, 90)
        gender = random.choice(["male", "female"])
        risk_tolerance = random.choice(["low", "moderate", "high"])

        # Sample purpose from config; sample target from all role target options
        purposes = agents_section.get("purposes", ["their destination"])
        purpose = random.choice(purposes)

        all_targets = [
            v
            for role_cfg in agents_section.get("roles", {}).values()
            for v in role_cfg.get("target", [])
        ]
        target = random.choice(all_targets) if all_targets else ""

        return {
            "id": agent_id,
            "name": f"Agent {agent_index}",
            "personality_type": personality_type,
            "age": age,
            "gender": gender,
            "risk_tolerance": risk_tolerance,
            "knowledge_profile": knowledge_profile,
            "initial_zone": "platform",
            "destination": "exit",
            "is_injured": is_injured,
            "purpose": purpose,
            "target": target,
            # agent_role, goal_state, purpose_memories and initial_goal are assigned
            # later by AgentManager once the spawn zone is known
        }

    @staticmethod
    def _select_knowledge_profile(knowledge_distribution: dict[str, float]) -> str:
        """Sample a knowledge profile according to configured distribution."""
        profiles = list(knowledge_distribution.keys())
        weights = [float(knowledge_distribution[p]) for p in profiles]
        return random.choices(profiles, weights=weights, k=1)[0]
