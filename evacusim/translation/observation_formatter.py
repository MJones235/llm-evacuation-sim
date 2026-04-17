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
    def format_conversation_history(
        agent_id: str,
        conversation_history: dict[str, list[dict]],
        nearby_agents: list[dict[str, Any]],
    ) -> list[str]:
        """
        Format conversation history for active conversations.

        Args:
            agent_id: ID of the observing agent
            conversation_history: Dict mapping other_agent_id to conversation messages
            nearby_agents: List of nearby agent info

        Returns:
            List of formatted conversation strings
        """
        # Only show conversations with nearby people who have exchanged multiple messages
        active_conversations = []
        nearby_ids = {a.get("id") for a in nearby_agents}

        for other_agent_id, messages in conversation_history.items():
            if other_agent_id in nearby_ids and len(messages) >= 2:
                # Get last 3 messages in this conversation
                recent = messages[-3:]
                convo_summary = []
                for m in recent:
                    direction = (
                        "You"
                        if m["from"] == agent_id
                        else other_agent_id.replace("agent_", "Person ")
                    )
                    convo_summary.append(f'{direction}: "{m["text"]}"')

                active_conversations.append(
                    {
                        "other": other_agent_id.replace("agent_", "Person "),
                        "summary": " → ".join(convo_summary),
                    }
                )

        if not active_conversations:
            return []

        lines = ["Recent conversation context:"]
        for convo in active_conversations[:2]:  # Max 2 to keep it concise
            lines.append(f"  - With {convo['other']}: {convo['summary']}")

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
    def format_exit_crowds(exit_crowds: dict[str, int], categorize_func) -> list[str]:
        """
        Format exit crowd information.

        Args:
            exit_crowds: Dict mapping exit name to agent count
            categorize_func: Function to categorize count into string

        Returns:
            List of formatted exit crowd strings
        """
        if not exit_crowds:
            return []

        lines = ["People heading toward exits:"]
        for exit_name, count in sorted(exit_crowds.items()):
            lines.append(f"  - {exit_name}: {categorize_func(count)}")

        return lines

    @staticmethod
    def format_events(events: list[str]) -> list[str]:
        """
        Format recent events.

        Args:
            events: List of event strings

        Returns:
            List of formatted event strings
        """
        if not events:
            return []

        # Normalize spacing to avoid odd double-spaces in prompt text.
        recent = [" ".join(str(event).split()) for event in events[-3:]]  # Last 3 events
        if len(recent) == 1:
            return [f"Recent events: {recent[0]}"]

        lines = ["Recent events:"]
        for event in recent:
            lines.append(f"- {event}")
        return lines

    @staticmethod
    def format_visible_exits(visible_exits: list[dict[str, str]]) -> list[str]:
        """
        Format visible exits for observation.

        Args:
            visible_exits: List of visible exit info dicts with name and distance

        Returns:
            List of formatted visible exit strings
        """
        if not visible_exits:
            return []

        # Group by distance category for cleaner presentation
        by_distance = {"very close": [], "nearby": [], "visible in distance": []}
        for exit_info in visible_exits:
            dist_cat = exit_info.get("distance", "visible in distance")
            if dist_cat in by_distance:
                by_distance[dist_cat].append(exit_info["name"])

        lines = []
        if by_distance["very close"]:
            exits_str = ", ".join(by_distance["very close"])
            lines.append(f"You can see {exits_str} very close to you.")
        if by_distance["nearby"]:
            exits_str = ", ".join(by_distance["nearby"])
            lines.append(f"You can see {exits_str} nearby.")
        if by_distance["visible in distance"]:
            exits_str = ", ".join(by_distance["visible in distance"])
            lines.append(f"You can see {exits_str} in the distance.")

        return lines

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
            lines.append(
                f"The {blocked['name']} appears blocked or obstructed "
                f"({blocked['distance']})."
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

    @staticmethod
    def format_last_decision(
        agent_id: str,
        last_decision: dict[str, Any],
    ) -> list[str]:
        """
        Format agent's previous decision as a natural language memory statement.

        Allows agents to see and reason about their prior commitment, supporting consistent behavior.

        Args:
            agent_id: ID of the agent
            last_decision: Dict containing the last translated_action details

        Returns:
            List of formatted decision memory strings
        """
        lines = []

        action_type = last_decision.get("action_type", "unknown")
        target_type = last_decision.get("target_type", "")
        target_agent = last_decision.get("target_agent")
        target_exit = last_decision.get("target_exit")
        zone_name = last_decision.get("zone_name")
        speed = last_decision.get("speed")
        reasoning = last_decision.get("reasoning", "")
        wait_reason = last_decision.get("wait_reason")
        decision_time = last_decision.get("time", 0)

        # Build natural language memory of decision
        if action_type == "move":
            target_desc = ""
            if target_type == "agent" and target_agent:
                target_desc = f"toward {target_agent.replace('agent_', 'Person ')}"
            elif target_type == "exit" and target_exit:
                target_desc = f"to {target_exit}"
            elif target_type == "zone" and zone_name:
                target_desc = f"to {zone_name}"
            else:
                target_desc = "to a location"

            speed_label = speed.replace("_", " ") if speed else ""
            speed_desc = f" at {speed_label}" if speed_label else ""
            lines.append(
                f"At t={decision_time:.0f}s, your previous decision was to move {target_desc}{speed_desc}."
            )

            if reasoning:
                lines.append(f"You reasoned: {reasoning}")

        elif action_type == "wait":
            wait_desc = wait_reason if wait_reason else "at your current location"
            if wait_reason:
                lines.append(f"At t={decision_time:.0f}s, your previous decision was to wait.")
                lines.append(f"You waited because: {wait_desc}")
            else:
                lines.append(
                    f"At t={decision_time:.0f}s, your previous decision was to wait {wait_desc}."
                )
            if reasoning:
                lines.append(f"You reasoned: {reasoning}")

        return lines
