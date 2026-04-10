"""
Conversation tracking for agent-to-agent communication.

Maintains dialogue history between pairs of agents.
"""

from evacusim.utils.logger import get_logger

logger = get_logger(__name__)


class ConversationTracker:
    """
    Tracks conversation history between agent pairs.

    Maintains bidirectional conversation logs for each pair of agents
    that have communicated.
    """

    def __init__(self):
        """Initialize conversation tracker."""
        # agent_id -> {other_agent_id -> conversation}
        self.agent_conversations: dict[str, dict[str, list[dict]]] = {}

    def track_message(
        self, sender_id: str, recipient_id: str, message_text: str, current_time: float
    ) -> None:
        """
        Track a message in the conversation between two agents.

        Updates conversation history for both sender and recipient.

        Args:
            sender_id: ID of the sending agent
            recipient_id: ID of the receiving agent
            message_text: Text of the message
            current_time: Current simulation time
        """
        message_entry = {
            "time": current_time,
            "from": sender_id,
            "to": recipient_id,
            "text": message_text,
        }

        # Track on sender's side
        if sender_id not in self.agent_conversations:
            self.agent_conversations[sender_id] = {}
        if recipient_id not in self.agent_conversations[sender_id]:
            self.agent_conversations[sender_id][recipient_id] = []

        self.agent_conversations[sender_id][recipient_id].append(message_entry)

        # Also track on recipient's side
        if recipient_id not in self.agent_conversations:
            self.agent_conversations[recipient_id] = {}
        if sender_id not in self.agent_conversations[recipient_id]:
            self.agent_conversations[recipient_id][sender_id] = []

        self.agent_conversations[recipient_id][sender_id].append(message_entry)

    def get_conversation_history(self, agent_id: str) -> dict[str, list[dict]]:
        """
        Get conversation history for an agent.

        Args:
            agent_id: ID of the agent

        Returns:
            Dictionary mapping other_agent_id to list of conversation entries
        """
        return self.agent_conversations.get(agent_id, {})
