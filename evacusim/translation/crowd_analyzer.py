"""
Crowd behavior analysis for observation generation.

Analyzes agent behaviors, movement patterns, and exit crowd densities.
"""

from typing import Any

from evacusim.utils.logger import get_logger

logger = get_logger(__name__)


class CrowdAnalyzer:
    """
    Analyzes crowd behavior and movement patterns.

    Handles:
    - Behavior summarization (injured, helping, moving, waiting)
    - Exit crowd density counting
    - Movement pattern categorization
    - Crowd density classification
    """

    @staticmethod
    def _classify_destination(target_exit: str | None) -> str | None:
        """
        Classify a raw destination key into a human-readable category.

        Args:
            target_exit: Raw exit/zone key stored in agent_destinations

        Returns:
            Human-readable destination category, or None if unknown
        """
        if not target_exit:
            return None
        tl = target_exit.lower()
        if "escalator" in tl:
            return "the escalators"
        if tl.startswith("platform_"):
            num = tl.split("_", 1)[1]
            return f"Platform {num}"
        # Street exits and other named exits
        readable = target_exit.replace("_", " ").title()
        return readable

    def __init__(self, exits: dict[str, tuple[float, float]]):
        """
        Initialize crowd analyzer.

        Args:
            exits: Dictionary mapping exit names to coordinates
        """
        self.exits = exits

    def summarize_behaviors(
        self,
        nearby_agents: list[dict[str, Any]],
        agent_injured: set[str],
    ) -> str:
        """
        Summarize what nearby agents are doing using three-dimensional model.

        Args:
            nearby_agents: List of nearby agent info dictionaries
            agent_injured: Set of injured agent IDs
        Returns:
            Natural language summary of behaviors
        """
        # Detect injured/slow-moving agents
        injured_nearby = []

        # Filter out agents that are following THIS agent (they'll be noted separately)
        independent_agents = [a for a in nearby_agents if not a.get("is_following_me", False)]

        for agent in independent_agents:
            agent_id = agent.get("id")
            if agent_id:
                distance = agent.get("distance", 999)

                # Check if injured (physical capability dimension)
                if agent_id in agent_injured and distance < 20.0:
                    injured_nearby.append(agent_id)

        # Build behavior summary
        parts = []

        # Classify where moving agents are actually heading
        moving_agents = [a for a in independent_agents if a.get("is_moving", True)]
        waiting_count = len(independent_agents) - len(moving_agents)

        dest_counts: dict[str, int] = {}
        for agent in moving_agents:
            dest = CrowdAnalyzer._classify_destination(agent.get("target_exit"))
            if dest:
                dest_counts[dest] = dest_counts.get(dest, 0) + 1

        if dest_counts:
            top_dest = max(dest_counts, key=dest_counts.__getitem__)
            if len(moving_agents) > waiting_count:
                if len(dest_counts) == 1:
                    parts.append(f"Most people nearby are heading toward {top_dest}.")
                else:
                    dest_list = sorted(dest_counts, key=dest_counts.__getitem__, reverse=True)
                    parts.append(f"People nearby are heading toward {' and '.join(dest_list[:2])}.")
            else:
                parts.append(f"Some people are heading toward {top_dest}; others are waiting.")
        elif moving_agents and len(moving_agents) > waiting_count:
            parts.append("Most people nearby are moving.")
        elif independent_agents:
            parts.append("Many people are waiting or stationary.")

        # Note injured agents nearby
        if injured_nearby:
            if len(injured_nearby) == 1:
                parts.append(
                    f"You notice {injured_nearby[0]} appears injured or moving very slowly."
                )
            else:
                parts.append(
                    f"You notice {len(injured_nearby)} people nearby appear injured or moving very slowly: {', '.join(injured_nearby[:3])}"
                )

        return " ".join(parts)

    def count_agents_per_exit(self, nearby_agents: list[dict[str, Any]]) -> dict[str, int]:
        """
        Count how many nearby agents appear to be heading toward each exit.

        Args:
            nearby_agents: List of nearby agent info dictionaries

        Returns:
            Dict mapping exit name to approximate agent count
        """
        exit_counts: dict[str, int] = {}

        for agent in nearby_agents:
            target_exit = agent.get("target_exit")
            if target_exit and target_exit in self.exits:
                exit_counts[target_exit] = exit_counts.get(target_exit, 0) + 1

        return exit_counts

    @staticmethod
    def categorize_density(num_nearby: int) -> str:
        """
        Categorize crowd density.

        Args:
            num_nearby: Number of nearby agents

        Returns:
            Density category string
        """
        if num_nearby == 0:
            return "empty (no one nearby)"
        elif num_nearby <= 3:
            return "sparse (a few people nearby)"
        elif num_nearby <= 10:
            return "moderate crowd nearby"
        else:
            return "crowded (many people nearby)"

    @staticmethod
    def categorize_count(count: int) -> str:
        """
        Categorize people count to prevent minor changes from triggering LLM.

        Args:
            count: Number of people

        Returns:
            Count category string
        """
        if count == 0:
            return "empty"
        elif count <= 3:
            return "sparse (few people)"
        elif count <= 10:
            return "moderate crowd"
        else:
            return "crowded (many people)"

    @staticmethod
    def analyze_movement_pattern(nearby_agents: list[dict[str, Any]]) -> str:
        """
        Analyze overall movement pattern of nearby agents.

        Args:
            nearby_agents: List of nearby agent info dictionaries

        Returns:
            Movement pattern description
        """
        if not nearby_agents:
            return ""

        moving_count = sum(1 for a in nearby_agents if a.get("is_moving", True))
        moving_pct = (moving_count / len(nearby_agents)) * 100

        dest_counts: dict[str, int] = {}
        for agent in nearby_agents:
            if agent.get("is_moving", True):
                dest = CrowdAnalyzer._classify_destination(agent.get("target_exit"))
                if dest:
                    dest_counts[dest] = dest_counts.get(dest, 0) + 1

        if moving_pct > 70:
            if dest_counts:
                top_dest = max(dest_counts, key=dest_counts.__getitem__)
                if len(dest_counts) == 1:
                    return f"Most people around you are heading toward {top_dest}."
                dest_list = sorted(dest_counts, key=dest_counts.__getitem__, reverse=True)
                return f"Most people around you are heading toward {' and '.join(dest_list[:2])}."
            return "Most people around you are on the move."
        elif moving_pct > 40:
            if dest_counts:
                dest_list = sorted(dest_counts, key=dest_counts.__getitem__, reverse=True)
                top_two = " and ".join(dest_list[:2])
                return f"The crowd is mixed — some heading toward {top_two}, others waiting."
            return "The crowd is mixed — some moving, others waiting."
        else:
            return "Many people around you are waiting or stationary."
