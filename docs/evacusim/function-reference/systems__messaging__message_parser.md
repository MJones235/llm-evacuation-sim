# Function Reference: evacusim/systems/messaging/message_parser.py

- Source file: [evacusim/systems/messaging/message_parser.py](../../../evacusim/systems/messaging/message_parser.py)
- Top-level classes: 1
- Top-level functions: 0

## Module Summary

Message parsing and validation for agent communication.

## Classes and Methods

### Class MessageParser

- Location: [evacusim/systems/messaging/message_parser.py#L15](../../../evacusim/systems/messaging/message_parser.py#L15)
- Summary: Parses and validates messages from agent actions.
- Method count: 3

#### extract_message_data

- Signature: def extract_message_data(action) -> dict[str, Any] | None
- Location: [evacusim/systems/messaging/message_parser.py#L26](../../../evacusim/systems/messaging/message_parser.py#L26)
- Summary: Extract message data from agent action JSON.
- Key calls: action.find, json.loads, data.get, logger.debug

#### get_message_radius

- Signature: def get_message_radius(message_type, default_radius) -> float
- Location: [evacusim/systems/messaging/message_parser.py#L64](../../../evacusim/systems/messaging/message_parser.py#L64)
- Summary: Determine message radius based on type.

#### get_type_emoji

- Signature: def get_type_emoji(message_type) -> str
- Location: [evacusim/systems/messaging/message_parser.py#L83](../../../evacusim/systems/messaging/message_parser.py#L83)
- Summary: Get emoji indicator for message type.
- Key calls: get

## Coverage Notes

- Nested helper functions documented in this file: 0
