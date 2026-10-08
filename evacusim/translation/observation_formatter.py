"""
Natural language observation formatting.

Formats simulation data into natural language observations for agents.
"""

from typing import Any

from evacusim.utils.logger import get_logger

logger = get_logger(__name__)


class ObservationFormatter:
    """
    Formats observations into natural language.

    Handles:
    - Message display formatting
    - Conversation history formatting
    - Event formatting
    - Nearby agent list formatting
    """

    @staticmethod
    def format_received_messages(received_messages: list[dict[str, Any]]) -> list[str]:
        """
        Format received messages for display.

        Args:
            received_messages: List of message dictionaries

        Returns:
            List of formatted message strings
        """
        # Show recent unique messages (last 5)
        unique_messages = []
        seen_texts = set()
        for msg in reversed(received_messages):
            msg_key = msg["text"][:30].lower()
            if msg_key not in seen_texts:
                unique_messages.append(msg)
                seen_texts.add(msg_key)
            if len(unique_messages) >= 5:
                break

        if not unique_messages:
            return []

        lines = ["What people just said to you:"]
        for msg in reversed(unique_messages):
            sender_id = msg["from"]
            role = msg.get("sender_role")
            # For concordia agents (agent_N) show "Person N (role)".
            # For director agents (rci_*, pa_*, etc.) use the role as the full
            # display name — the technical ID is meaningless to passengers.
            if role and not sender_id.startswith("agent_"):
                sender_name = role
            elif role:
                sender_name = f"{sender_id.replace('agent_', 'Person ')} ({role})"
            else:
                sender_name = sender_id.replace("agent_", "Person ")
            msg_type = msg.get("message_type", "")
            type_indicator = {
                "directed": " (to you)",
                "quiet": " (quietly)",
                "shout": " (shouting)",
                "directive": " (directing you)",
                "pa": "",
            }.get(msg_type, "")
            lines.append(f'{sender_name}{type_indicator} said: "{msg["text"]}"')

        return lines

    @staticmethod
    def format_nearby_agent_ids(nearby_agents: list[dict[str, Any]]) -> list[str]:
        """
        Format nearby agent IDs for targeting messages.
        Includes role label when present (e.g. director agents).

        Args:
            nearby_agents: List of nearby agent info

        Returns:
            List with single formatted string, or empty list
        """
        # Only list IDs when there are a few people (not in crowds)
        if len(nearby_agents) > 0 and len(nearby_agents) <= 5:
            nearby_people = []
            for agent in nearby_agents[:5]:
                aid = agent.get("id")
                if aid:
                    person_name = aid.replace("agent_", "Person ")
                    role = agent.get("role")
                    if role:
                        nearby_people.append(f"{person_name} ({role}, {aid})")
                    else:
                        nearby_people.append(f"{person_name} ({aid})")

            if nearby_people:
                return [f"Nearby people: {', '.join(nearby_people)}."]
        return []

    @staticmethod
    def format_blocked_exits(visible_blocked: list[dict[str, Any]]) -> list[str]:
        """
        Format visible blocked exits.

        Args:
            visible_blocked: List of blocked exit info dicts

        Returns:
            List of formatted blocked exit strings
        """
        if not visible_blocked:
            return []

        lines = ["Visual observations:"]
        for blocked in visible_blocked:
            dist = blocked["distance"]
            if dist == "remembered":
                # Exit was seen as blocked earlier in this evacuation — keep it
                # in the agent's mental model even when they've moved away.
                lines.append(
                    f"The {blocked['name']} appears blocked or obstructed "
                    f"(remembered from earlier observation)."
                )
            else:
                lines.append(
                    f"The {blocked['name']} appears blocked or obstructed (seen now: {dist})."
                )

        return lines

    @staticmethod
    def format_own_status(
        agent_id: str,
        agent_injured: set[str],
        agent_action: dict[str, str],
        state_queries=None,
    ) -> list[str]:
        """
        Format agent's own status from three-dimensional model.

        Args:
            agent_id: ID of the agent
            agent_injured: Set of injured agent IDs
            agent_action: Dict of agent_id -> action ("moving"|"waiting")
            state_queries: SimulationStateQueries for position lookups (optional)

        Returns:
            List of formatted status strings (may be empty)
        """
        lines = []

        # Physical capability dimension
        if agent_id in agent_injured:
            lines.append("You are injured and moving slowly.")

        # Action dimension (waiting for assistance is special case)
        action = agent_action.get(agent_id, "moving")
        if action == "waiting":
            # Only mention waiting if they're not already mentioned as injured/helping
            if agent_id not in agent_injured:
                lines.append("You are waiting.")

        return lines
