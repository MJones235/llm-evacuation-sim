# Function Reference: evacusim/translation/observation_generator.py

- Source file: [evacusim/translation/observation_generator.py](../../../evacusim/translation/observation_generator.py)
- Top-level classes: 1
- Top-level functions: 0

## Module Summary

Generates natural language observations from JuPedSim simulation state.

## Classes and Methods

### Class ObservationGenerator

- Location: [evacusim/translation/observation_generator.py#L22](../../../evacusim/translation/observation_generator.py#L22)
- Summary: Generates natural language observations from JuPedSim simulation state.
- Method count: 9

#### __init__

- Signature: def __init__(self, station_layout, jps_sim=None)
- Location: [evacusim/translation/observation_generator.py#L30](../../../evacusim/translation/observation_generator.py#L30)
- Summary: Initialize the observation generator.
- Key calls: station_layout.get, build_registry_from_station_layout, SpatialAnalyzer, CrowdAnalyzer, self._build_exit_name_lookups

#### _build_exit_name_lookups

- Signature: def _build_exit_name_lookups(self) -> None
- Location: [evacusim/translation/observation_generator.py#L71](../../../evacusim/translation/observation_generator.py#L71)
- Summary: Populate case-insensitive exit name → canonical key and reverse lookups.
- Key calls: self.spatial_analyzer._canonical_visible_exit_key, self._update_canonical_display, self.exit_registry.get_display_name, raw_name.lower, display.lower

#### _update_canonical_display

- Signature: def _update_canonical_display(self, canonical, display) -> None
- Location: [evacusim/translation/observation_generator.py#L87](../../../evacusim/translation/observation_generator.py#L87)
- Summary: Keep the most descriptive display name for *canonical* in the lookup.
- Key calls: self._canonical_to_display.get, existing.lower, display.lower, len

#### _learn_exits_from_messages

- Signature: def _learn_exits_from_messages(self, agent_id, messages) -> None
- Location: [evacusim/translation/observation_generator.py#L100](../../../evacusim/translation/observation_generator.py#L100)
- Summary: Scan received messages for exit name mentions and add them to known exits.
- Key calls: self.agent_known_exits.setdefault, set, lower, self._exit_name_to_canonical.items, msg.get, known.add

#### generate_observation

- Signature: def generate_observation(self, agent_id, position, nearby_agents, events, sim_time, blocked_exits=None, agent_injured=None, agent_action=None, agent_level=None, agent_last_decision=None, state_queries=None, received_messages=None, conversation_history=None, inactive_exits=None, known_blocked_exits=None) -> str
- Location: [evacusim/translation/observation_generator.py#L109](../../../evacusim/translation/observation_generator.py#L109)
- Summary: Generate a natural language observation for an agent.
- Key calls: self._format_new_information, observations.extend, self.spatial_analyzer.identify_zone, self.station_layout.get, zone_labels.get, observations.append, ObservationFormatter.format_own_status, self.spatial_analyzer.get_visible_exits, self.agent_known_exits.setdefault, set

#### _format_new_information

- Signature: def _format_new_information(self, agent_id, events, received_messages) -> list[str]
- Location: [evacusim/translation/observation_generator.py#L411](../../../evacusim/translation/observation_generator.py#L411)
- Summary: Format only new information since this agent's previous observation.
- Key calls: self._last_event_signatures.get, self._last_message_signatures.get, updates.extend, lines.append, join, set, msg.get, strip, updates.append, msg_sig.split

#### _humanize_exit_crowd_keys

- Signature: def _humanize_exit_crowd_keys(self, exit_crowds) -> dict[str, int]
- Location: [evacusim/translation/observation_generator.py#L476](../../../evacusim/translation/observation_generator.py#L476)
- Summary: Translate technical exit IDs into display names and merge equivalent escalator IDs.
- Key calls: exit_crowds.items, self._canonical_exit_key, self.exit_registry.get_display_name, self._prefer_exit_label, sorted, merged.values, lower

#### _canonical_exit_key

- Signature: def _canonical_exit_key(self, exit_id) -> str
- Location: [evacusim/translation/observation_generator.py#L502](../../../evacusim/translation/observation_generator.py#L502)
- Summary: Map equivalent exit IDs (zone IDs vs escalator IDs) to one canonical key.
- Key calls: re.match, escalator_match.groups, zone_escalator_match.groups

#### _prefer_exit_label

- Signature: def _prefer_exit_label(self, current_label, candidate_label) -> str
- Location: [evacusim/translation/observation_generator.py#L516](../../../evacusim/translation/observation_generator.py#L516)
- Summary: Prefer more descriptive display labels when combining equivalent exits.
- Key calls: current_label.lower, candidate_label.lower, len

## Coverage Notes

- Nested helper functions documented in this file: 0
