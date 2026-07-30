# Function Reference: evacusim/translation/observation_formatter.py

- Source file: [evacusim/translation/observation_formatter.py](../../../evacusim/translation/observation_formatter.py)
- Top-level classes: 1
- Top-level functions: 0

## Module Summary

Natural language observation formatting.

## Classes and Methods

### Class ObservationFormatter

- Location: [evacusim/translation/observation_formatter.py#L14](../../../evacusim/translation/observation_formatter.py#L14)
- Summary: Formats observations into natural language.
- Method count: 9

#### format_received_messages

- Signature: def format_received_messages(received_messages) -> list[str]
- Location: [evacusim/translation/observation_formatter.py#L26](../../../evacusim/translation/observation_formatter.py#L26)
- Summary: Format received messages for display.
- Key calls: set, reversed, lower, msg.get, get, lines.append, unique_messages.append, seen_texts.add, len, sender_id.startswith

#### format_conversation_history

- Signature: def format_conversation_history(agent_id, conversation_history, nearby_agents) -> list[str]
- Location: [evacusim/translation/observation_formatter.py#L76](../../../evacusim/translation/observation_formatter.py#L76)
- Summary: Format conversation history for active conversations.
- Key calls: conversation_history.items, a.get, lines.append, active_conversations.append, len, convo_summary.append, other_agent_id.replace, join

#### format_nearby_agent_ids

- Signature: def format_nearby_agent_ids(nearby_agents) -> list[str]
- Location: [evacusim/translation/observation_formatter.py#L126](../../../evacusim/translation/observation_formatter.py#L126)
- Summary: Format nearby agent IDs for targeting messages.
- Key calls: len, agent.get, aid.replace, nearby_people.append, join

#### format_exit_crowds

- Signature: def format_exit_crowds(exit_crowds, categorize_func) -> list[str]
- Location: [evacusim/translation/observation_formatter.py#L155](../../../evacusim/translation/observation_formatter.py#L155)
- Summary: Format exit crowd information.
- Key calls: sorted, exit_crowds.items, lines.append, categorize_func

#### format_events

- Signature: def format_events(events) -> list[str]
- Location: [evacusim/translation/observation_formatter.py#L176](../../../evacusim/translation/observation_formatter.py#L176)
- Summary: Format recent events.
- Key calls: join, len, lines.append, split, str

#### format_visible_exits

- Signature: def format_visible_exits(visible_exits) -> list[str]
- Location: [evacusim/translation/observation_formatter.py#L200](../../../evacusim/translation/observation_formatter.py#L200)
- Summary: Format visible exits for observation.
- Key calls: exit_info.get, join, lines.append, append

#### format_blocked_exits

- Signature: def format_blocked_exits(visible_blocked) -> list[str]
- Location: [evacusim/translation/observation_formatter.py#L234](../../../evacusim/translation/observation_formatter.py#L234)
- Summary: Format visible blocked exits.
- Key calls: lines.append

#### format_own_status

- Signature: def format_own_status(agent_id, agent_injured, agent_action, state_queries=None) -> list[str]
- Location: [evacusim/translation/observation_formatter.py#L266](../../../evacusim/translation/observation_formatter.py#L266)
- Summary: Format agent's own status from three-dimensional model.
- Key calls: agent_action.get, lines.append

#### format_last_decision

- Signature: def format_last_decision(agent_id, last_decision) -> list[str]
- Location: [evacusim/translation/observation_formatter.py#L300](../../../evacusim/translation/observation_formatter.py#L300)
- Summary: Format agent's previous decision as a natural language memory statement.
- Key calls: last_decision.get, lines.append, speed.replace, target_agent.replace

## Coverage Notes

- Nested helper functions documented in this file: 0
