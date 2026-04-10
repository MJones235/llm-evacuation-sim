"""
Message memory and deduplication for agent communication.

Prevents agents from repeating messages and hearing the same message multiple times.
"""

from typing import Any

from evacusim.utils.logger import get_logger

logger = get_logger(__name__)


class MessageMemory:
    """
    Manages message memory and deduplication.

    Features:
    - Prevents agents from repeating recently sent messages
    - Prevents agents from hearing duplicate messages
    - Time-based memory window for cleanup
    """

    def __init__(self, memory_window: float = 60.0):
        """
        Initialize message memory.

        Args:
            memory_window: How long to remember sent messages (seconds)
        """
        self.memory_window = memory_window

        # Track sent messages per agent
        self.agent_sent_messages: dict[str, list[dict[str, Any]]] = {}

        # Track heard messages per agent (set of message keys)
        self.agent_heard_messages: dict[str, set[str]] = {}

    def is_repeat_message(self, sender_id: str, message_text: str, current_time: float) -> bool:
        """
        Check if this message was recently sent by the same agent.

        Args:
            sender_id: ID of the agent
            message_text: Text of the message
            current_time: Current simulation time

        Returns:
            True if message is a repeat, False otherwise
        """
        if sender_id not in self.agent_sent_messages:
            return False

        # Clean old messages outside memory window
        recent_cutoff = current_time - self.memory_window
        self.agent_sent_messages[sender_id] = [
            msg for msg in self.agent_sent_messages[sender_id] if msg["time"] >= recent_cutoff
        ]

        # Check if similar message was recently sent
        for recent_msg in self.agent_sent_messages[sender_id]:
            if recent_msg["text"].lower() == message_text.lower():
                logger.debug(
                    f"{sender_id} suppressed repeat message: '{message_text}' "
                    f"(last sent {current_time - recent_msg['time']:.0f}s ago)"
                )
                return True

        return False

    def record_sent_message(self, sender_id: str, message_text: str, current_time: float) -> None:
        """
        Record a message in sender's sent history.

        Args:
            sender_id: ID of the agent
            message_text: Text of the message
            current_time: Current simulation time
        """
        if sender_id not in self.agent_sent_messages:
            self.agent_sent_messages[sender_id] = []
        self.agent_sent_messages[sender_id].append({"time": current_time, "text": message_text})

    def has_heard_message(self, recipient_id: str, message_text: str) -> bool:
        """
        Check if recipient has already heard this message.

        Args:
            recipient_id: ID of the recipient
            message_text: Text of the message

        Returns:
            True if already heard, False otherwise
        """
        if recipient_id not in self.agent_heard_messages:
            self.agent_heard_messages[recipient_id] = set()

        # Clean old heard messages if too many accumulated
        if len(self.agent_heard_messages[recipient_id]) > 50:
            self.agent_heard_messages[recipient_id].clear()

        # Check if already heard (use first 30 chars normalized as key)
        msg_key = f"{message_text.lower()[:30]}"
        return msg_key in self.agent_heard_messages[recipient_id]

    def mark_as_heard(self, recipient_id: str, message_text: str) -> None:
        """
        Mark a message as heard by recipient.

        Args:
            recipient_id: ID of the recipient
            message_text: Text of the message
        """
        if recipient_id not in self.agent_heard_messages:
            self.agent_heard_messages[recipient_id] = set()

        msg_key = f"{message_text.lower()[:30]}"
        self.agent_heard_messages[recipient_id].add(msg_key)
