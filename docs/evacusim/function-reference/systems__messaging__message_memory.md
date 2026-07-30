# Function Reference: evacusim/systems/messaging/message_memory.py

- Source file: [evacusim/systems/messaging/message_memory.py](../../../evacusim/systems/messaging/message_memory.py)
- Top-level classes: 1
- Top-level functions: 0

## Module Summary

Message memory and deduplication for agent communication.

## Classes and Methods

### Class MessageMemory

- Location: [evacusim/systems/messaging/message_memory.py#L14](../../../evacusim/systems/messaging/message_memory.py#L14)
- Summary: Manages message memory and deduplication.
- Method count: 5

#### __init__

- Signature: def __init__(self, memory_window=60.0)
- Location: [evacusim/systems/messaging/message_memory.py#L24](../../../evacusim/systems/messaging/message_memory.py#L24)
- Summary: Initialize message memory.

#### is_repeat_message

- Signature: def is_repeat_message(self, sender_id, message_text, current_time) -> bool
- Location: [evacusim/systems/messaging/message_memory.py#L39](../../../evacusim/systems/messaging/message_memory.py#L39)
- Summary: Check if this message was recently sent by the same agent.
- Key calls: lower, message_text.lower, logger.debug

#### record_sent_message

- Signature: def record_sent_message(self, sender_id, message_text, current_time) -> None
- Location: [evacusim/systems/messaging/message_memory.py#L71](../../../evacusim/systems/messaging/message_memory.py#L71)
- Summary: Record a message in sender's sent history.
- Key calls: append

#### has_heard_message

- Signature: def has_heard_message(self, recipient_id, message_text) -> bool
- Location: [evacusim/systems/messaging/message_memory.py#L84](../../../evacusim/systems/messaging/message_memory.py#L84)
- Summary: Check if recipient has already heard this message.
- Key calls: set, len, clear, message_text.lower

#### mark_as_heard

- Signature: def mark_as_heard(self, recipient_id, message_text) -> None
- Location: [evacusim/systems/messaging/message_memory.py#L106](../../../evacusim/systems/messaging/message_memory.py#L106)
- Summary: Mark a message as heard by recipient.
- Key calls: add, set, message_text.lower

## Coverage Notes

- Nested helper functions documented in this file: 0
