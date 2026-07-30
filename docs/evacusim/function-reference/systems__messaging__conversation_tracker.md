# Function Reference: evacusim/systems/messaging/conversation_tracker.py

- Source file: [evacusim/systems/messaging/conversation_tracker.py](../../../evacusim/systems/messaging/conversation_tracker.py)
- Top-level classes: 1
- Top-level functions: 0

## Module Summary

Conversation tracking for agent-to-agent communication.

## Classes and Methods

### Class ConversationTracker

- Location: [evacusim/systems/messaging/conversation_tracker.py#L12](../../../evacusim/systems/messaging/conversation_tracker.py#L12)
- Summary: Tracks conversation history between agent pairs.
- Method count: 3

#### __init__

- Signature: def __init__(self)
- Location: [evacusim/systems/messaging/conversation_tracker.py#L20](../../../evacusim/systems/messaging/conversation_tracker.py#L20)
- Summary: Initialize conversation tracker.

#### track_message

- Signature: def track_message(self, sender_id, recipient_id, message_text, current_time) -> None
- Location: [evacusim/systems/messaging/conversation_tracker.py#L25](../../../evacusim/systems/messaging/conversation_tracker.py#L25)
- Summary: Track a message in the conversation between two agents.
- Key calls: append

#### get_conversation_history

- Signature: def get_conversation_history(self, agent_id) -> dict[str, list[dict]]
- Location: [evacusim/systems/messaging/conversation_tracker.py#L62](../../../evacusim/systems/messaging/conversation_tracker.py#L62)
- Summary: Get conversation history for an agent.
- Key calls: self.agent_conversations.get

## Coverage Notes

- Nested helper functions documented in this file: 0
