"""
Message parsing and validation for agent communication.

Extracts and validates messages from agent action JSON strings.
"""

from evacusim.utils.logger import get_logger

logger = get_logger(__name__)


class MessageParser:
    """
    Parses and validates messages from agent actions.

    Handles:
    - JSON extraction from action strings
    - Message field validation
    - Type checking for message components
    """

    @staticmethod
    def get_type_emoji(message_type: str | None) -> str:
        """
        Get emoji indicator for message type.

        Args:
            message_type: Type of message

        Returns:
            Emoji string for logging
        """
        return {"directed": "💬", "shout": "📢", "quiet": "🤫"}.get(message_type, "📣")
