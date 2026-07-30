# Function Reference: evacusim/systems/messaging/message_system.py

- Source file: [evacusim/systems/messaging/message_system.py](../../../evacusim/systems/messaging/message_system.py)
- Top-level classes: 1
- Top-level functions: 0

## Module Summary

Message system for agent-to-agent communication in evacuation scenarios.

## Classes and Methods

### Class MessageSystem

- Location: [evacusim/systems/messaging/message_system.py#L22](../../../evacusim/systems/messaging/message_system.py#L22)
- Summary: Manages agent-to-agent messaging with memory and deduplication.
- Method count: 13

#### __init__

- Signature: def __init__(self, default_radius=10.0, memory_window=60.0, shout_cooldown=45.0, local_duplicate_window=20.0, local_duplicate_radius=12.0, max_shouts_per_timestep=2)
- Location: [evacusim/systems/messaging/message_system.py#L34](../../../evacusim/systems/messaging/message_system.py#L34)
- Summary: Initialize the message system.
- Key calls: MessageMemory, ConversationTracker

#### extract_and_deliver_message

- Signature: def extract_and_deliver_message(self, sender_id, action, sender_position, current_sim_time, state_queries, exited_agents) -> dict[str, Any] | None
- Location: [evacusim/systems/messaging/message_system.py#L71](../../../evacusim/systems/messaging/message_system.py#L71)
- Summary: Extract message from action JSON and deliver to nearby agents.
- Key calls: MessageParser.extract_message_data, self.memory.is_repeat_message, MessageParser.get_message_radius, state_queries.get_nearby_agents, self._find_recipients, self._deliver_to_recipients, self._record_message_constraints_state, self.message_history.append, self.memory.record_sent_message, MessageParser.get_type_emoji

#### _should_send_message

- Signature: def _should_send_message(self, sender_id, message_text, message_type, target_agent, sender_position, current_sim_time) -> bool
- Location: [evacusim/systems/messaging/message_system.py#L173](../../../evacusim/systems/messaging/message_system.py#L173)
- Summary: Apply lightweight social realism constraints before message delivery.
- Key calls: self._last_shout_time_by_agent.get, int, self._shout_count_by_bucket.get, self._is_generic_alert, self._has_nearby_recent_alert

#### _record_message_constraints_state

- Signature: def _record_message_constraints_state(self, sender_id, message_text, message_type, sender_position, current_sim_time) -> None
- Location: [evacusim/systems/messaging/message_system.py#L205](../../../evacusim/systems/messaging/message_system.py#L205)
- Summary: Update shout throttling state after a message is accepted.
- Key calls: int, self._is_generic_alert, self._shout_count_by_bucket.get, self._recent_local_alerts.append, self._prune_recent_local_alerts

#### _prune_recent_local_alerts

- Signature: def _prune_recent_local_alerts(self, current_sim_time) -> None
- Location: [evacusim/systems/messaging/message_system.py#L230](../../../evacusim/systems/messaging/message_system.py#L230)
- Summary: Drop stale local alert records outside suppression window.

#### _has_nearby_recent_alert

- Signature: def _has_nearby_recent_alert(self, sender_position, current_sim_time) -> bool
- Location: [evacusim/systems/messaging/message_system.py#L237](../../../evacusim/systems/messaging/message_system.py#L237)
- Summary: True if another generic alert was shouted nearby very recently.
- Key calls: self._prune_recent_local_alerts

#### _is_generic_alert

- Signature: def _is_generic_alert(self, text) -> bool
- Location: [evacusim/systems/messaging/message_system.py#L254](../../../evacusim/systems/messaging/message_system.py#L254)
- Summary: Detect low-information evacuation shout patterns.
- Key calls: text.lower, bool, re.search

#### deliver_directive

- Signature: def deliver_directive(self, sender_id, message_text, sender_position, current_sim_time, state_queries, exited_agents, radius=None, messages_by_zone=None, zone_id_for_agent_fn=None) -> None
- Location: [evacusim/systems/messaging/message_system.py#L264](../../../evacusim/systems/messaging/message_system.py#L264)
- Summary: Deliver a rule-based directive from a director agent to nearby agents.
- Key calls: state_queries.get_nearby_agents, MessageParser.get_type_emoji, logger.debug, append, self.conversation_tracker.track_message, logger.info, zone_id_for_agent_fn, len

#### deliver_pa

- Signature: def deliver_pa(self, sender_label, message_text, current_sim_time, all_agent_ids, exited_agents, messages_by_zone=None, zone_id_for_agent_fn=None) -> None
- Location: [evacusim/systems/messaging/message_system.py#L349](../../../evacusim/systems/messaging/message_system.py#L349)
- Summary: Deliver a station-wide PA announcement to all agents.
- Key calls: append, logger.info, zone_id_for_agent_fn

#### get_received_messages

- Signature: def get_received_messages(self, agent_id) -> list[dict[str, Any]]
- Location: [evacusim/systems/messaging/message_system.py#L402](../../../evacusim/systems/messaging/message_system.py#L402)
- Summary: Get messages received by an agent and clear them.
- Key calls: self.agent_messages.get

#### get_conversation_history

- Signature: def get_conversation_history(self, agent_id) -> dict[str, list[dict]]
- Location: [evacusim/systems/messaging/message_system.py#L408](../../../evacusim/systems/messaging/message_system.py#L408)
- Summary: Get conversation history for an agent.
- Key calls: self.conversation_tracker.get_conversation_history

#### _find_recipients

- Signature: def _find_recipients(self, target_agent, nearby_agents, exited_agents, sender_id) -> list[str]
- Location: [evacusim/systems/messaging/message_system.py#L412](../../../evacusim/systems/messaging/message_system.py#L412)
- Summary: Find recipient agent IDs based on targeting.
- Key calls: any

#### _deliver_to_recipients

- Signature: def _deliver_to_recipients(self, sender_id, message_text, message_type, recipient_ids, current_time)
- Location: [evacusim/systems/messaging/message_system.py#L435](../../../evacusim/systems/messaging/message_system.py#L435)
- Summary: Deliver message to all recipients with deduplication.
- Key calls: self.memory.has_heard_message, self.memory.mark_as_heard, append, self.conversation_tracker.track_message

## Coverage Notes

- Nested helper functions documented in this file: 0
