# Function Reference: evacusim/decision/prompt_cache.py

- Source file: [evacusim/decision/prompt_cache.py](../../../evacusim/decision/prompt_cache.py)
- Top-level classes: 2
- Top-level functions: 0

## Module Summary

Prompt Cache & Change Detection Implements intelligent prompt caching to reduce unnecessary LLM calls by: 1.

## Classes and Methods

### Class PromptCache

- Location: [evacusim/decision/prompt_cache.py#L66](../../../evacusim/decision/prompt_cache.py#L66)
- Summary: Content-aware prompt caching for agent decision-making.
- Method count: 15

#### __init__

- Signature: def __init__(self, enable_detailed_logging=False, stable_skip_threshold=5)
- Location: [evacusim/decision/prompt_cache.py#L75](../../../evacusim/decision/prompt_cache.py#L75)
- Summary: Initialize prompt cache.

#### should_call_llm

- Signature: def should_call_llm(self, agent_id, observation, action_spec_text, received_messages=None, blocked_exits=None, recent_events=None) -> tuple[bool, str | None]
- Location: [evacusim/decision/prompt_cache.py#L95](../../../evacusim/decision/prompt_cache.py#L95)
- Summary: Determine if LLM call is needed based on prompt changes.
- Key calls: self._extract_significant_content, self._consecutive_hits.get, bool, self._compute_content_hash, get, self._record_cache, self._filter_stable_agent_content, self.agent_decisions.get, self._describe_change_detailed, logger.info

#### cache_decision

- Signature: def cache_decision(self, agent_id, decision_json) -> None
- Location: [evacusim/decision/prompt_cache.py#L182](../../../evacusim/decision/prompt_cache.py#L182)
- Summary: Store decision for future reuse if prompt unchanged.
- Key calls: logger.debug

#### get_cached_decision

- Signature: def get_cached_decision(self, agent_id) -> str | None
- Location: [evacusim/decision/prompt_cache.py#L194](../../../evacusim/decision/prompt_cache.py#L194)
- Summary: Retrieve cached decision if available.
- Key calls: self.agent_decisions.get

#### clear_agent

- Signature: def clear_agent(self, agent_id) -> None
- Location: [evacusim/decision/prompt_cache.py#L206](../../../evacusim/decision/prompt_cache.py#L206)
- Summary: Clear cache for specific agent (e.
- Key calls: self.agent_prompts.pop, self.agent_decisions.pop, self._consecutive_hits.pop, logger.debug

#### get_statistics

- Signature: def get_statistics(self) -> dict[str, Any]
- Location: [evacusim/decision/prompt_cache.py#L219](../../../evacusim/decision/prompt_cache.py#L219)
- Summary: Return cache statistics (useful for monitoring).
- Key calls: len, sum, self.agent_change_history.values

#### _filter_stable_agent_content

- Signature: def _filter_stable_agent_content(self, significant_content) -> str
- Location: [evacusim/decision/prompt_cache.py#L236](../../../evacusim/decision/prompt_cache.py#L236)
- Summary: Apply extra normalisation for stable agents (those with a long hit streak).
- Key calls: significant_content.split, join, line.startswith, _RE_CROWD_COUNT.sub, obs.replace, strip, cleaned.append, _RE_MULTI_SPACE.sub

#### _extract_significant_content

- Signature: def _extract_significant_content(self, observation, messages=None, blocked_exits=None, events=None) -> str
- Location: [evacusim/decision/prompt_cache.py#L261](../../../evacusim/decision/prompt_cache.py#L261)
- Summary: Extract only semantically significant parts of the prompt.
- Key calls: join, self._filter_observation, significant_parts.append, self._summarize_events, json.dumps, sorted

#### _filter_observation

- Signature: def _filter_observation(self, observation) -> str
- Location: [evacusim/decision/prompt_cache.py#L296](../../../evacusim/decision/prompt_cache.py#L296)
- Summary: Filter observation to remove volatile components that shouldn't trigger new LLM calls.
- Key calls: _RE_PREV_DECISION.sub, _RE_YOU_REASONED.sub, _RE_YOU_ARE_IN.sub, _RE_ESC_DIRECTION.sub, _RE_EXIT_QUALIFIER.sub, _RE_YOU_CAN_SEE_EXIT.sub, _RE_PROXIMITY_ADVERBS.sub, _RE_AGENT_IDS.sub, _RE_NEARBY_EMPTY.sub, _RE_NEARBY_PEOPLE.sub

#### _summarize_events

- Signature: def _summarize_events(self, events) -> str
- Location: [evacusim/decision/prompt_cache.py#L361](../../../evacusim/decision/prompt_cache.py#L361)
- Summary: Create a concise summary of events (reduces volatility).
- Key calls: set, isinstance, summary_parts.append, join, event_types.add, event.get, event_locations.add, sorted

#### _compute_content_hash

- Signature: def _compute_content_hash(self, prompt_text, significant_content) -> str
- Location: [evacusim/decision/prompt_cache.py#L384](../../../evacusim/decision/prompt_cache.py#L384)
- Summary: Create hash of prompt + significant content.
- Key calls: hexdigest, hashlib.sha256, combined.encode

#### _record_cache

- Signature: def _record_cache(self, agent_id, content_hash, prompt_text, significant_content) -> None
- Location: [evacusim/decision/prompt_cache.py#L393](../../../evacusim/decision/prompt_cache.py#L393)
- Summary: Record cache entry for agent.
- Key calls: len

#### _describe_change

- Signature: def _describe_change(self, agent_id, old_hash, new_hash) -> str
- Location: [evacusim/decision/prompt_cache.py#L409](../../../evacusim/decision/prompt_cache.py#L409)
- Summary: Generate human-readable description of what changed.
- Key calls: get, change_hints.append, self.agent_prompts.get, str, join

#### _describe_change_detailed

- Signature: def _describe_change_detailed(self, agent_id, old_hash, new_hash, old_content, new_content) -> str
- Location: [evacusim/decision/prompt_cache.py#L438](../../../evacusim/decision/prompt_cache.py#L438)
- Summary: Describe what changed between prompts with detailed content comparison.
- Key calls: parse_sections, join, content.split, old_sec.get, new_sec.get, changes.append, line.startswith, strip, len, new_msgs.split
- Nested function count: 1

##### Nested helpers inside _describe_change_detailed

###### _describe_change_detailed.parse_sections

- Signature: def parse_sections(content) -> dict
- Location: [evacusim/decision/prompt_cache.py#L458](../../../evacusim/decision/prompt_cache.py#L458)
- Summary: No nested-function docstring summary available.
- Key calls: content.split, line.startswith, strip

#### _track_change

- Signature: def _track_change(self, agent_id, change_desc, significant_content) -> None
- Location: [evacusim/decision/prompt_cache.py#L509](../../../evacusim/decision/prompt_cache.py#L509)
- Summary: Track change in history for this agent.
- Key calls: append, len

### Class SignificantChangeDetector

- Location: [evacusim/decision/prompt_cache.py#L523](../../../evacusim/decision/prompt_cache.py#L523)
- Summary: Detects if a prompt change is truly "significant" (warrants LLM call).
- Method count: 1

#### score_change_significance

- Signature: def score_change_significance(old_observation, new_observation, received_messages=None, blocked_exits_changed=False, recent_events=None) -> float
- Location: [evacusim/decision/prompt_cache.py#L531](../../../evacusim/decision/prompt_cache.py#L531)
- Summary: Score how significant the change is (0.
- Key calls: min

## Coverage Notes

- Nested helper functions documented in this file: 1
