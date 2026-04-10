"""
Message parsing and validation for agent communication.

Extracts and validates messages from agent action JSON strings.
"""

import json
from typing import Any

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
    def extract_message_data(action: str) -> dict[str, Any] | None:
        """
        Extract message data from agent action JSON.

        Args:
            action: JSON action string from agent

        Returns:
            Dictionary with message_text, message_type, and target_agent if valid,
            None if no valid message found
        """
        try:
            # Parse action JSON to extract message
            json_start = action.find("{")
            if json_start > 0:
                action = action[json_start:]

            data = json.loads(action)
            message_text = data.get("message")
            message_type = data.get("message_type")  # directed, shout, quiet
            target_agent = data.get("target_agent")  # agent_id or null

            if not message_text or message_text == "null":
                return None

            return {
                "message_text": message_text,
                "message_type": message_type,
                "target_agent": target_agent,
            }

        except json.JSONDecodeError:
            return None
        except Exception as e:
            logger.debug(f"Error parsing message: {e}")
            return None

    @staticmethod
    def get_message_radius(message_type: str | None, default_radius: float) -> float:
        """
        Determine message radius based on type.

        Args:
            message_type: Type of message (quiet, shout, or None for default)
            default_radius: Default radius for unspecified type

        Returns:
            Radius in meters
        """
        if message_type == "quiet":
            return 3.0  # Only very close people
        elif message_type == "shout":
            return 15.0  # Wider range for warnings
        else:
            return default_radius  # Default 10m

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
