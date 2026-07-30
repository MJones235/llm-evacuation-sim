# Function Reference: evacusim/decision/decision_processor.py

- Source file: [evacusim/decision/decision_processor.py](../../../evacusim/decision/decision_processor.py)
- Top-level classes: 1
- Top-level functions: 0

## Module Summary

Decision Processor Handles agent decision-making logic including: - Parallel processing of agent decisions using asyncio - LLM calls with comprehensive prompts for action selection - JSON response parsing and reasoning extraction - Action translation and decision recording - Route change detection and tracking - Intelligent prompt caching to reduce redundant LLM calls This module coordinates the cognitive layer (Concordia) decision-making process.

## Classes and Methods

### Class DecisionProcessor

- Location: [evacusim/decision/decision_processor.py#L52](../../../evacusim/decision/decision_processor.py#L52)
- Summary: Processes agent decision-making with parallel LLM calls.
- Method count: 13

#### __init__

- Signature: def __init__(self, concordia_agents, exited_agents, action_translator, action_executor, message_system, state_queries, station_layout, agent_decisions, agent_destinations, last_observations, last_actions, perf_timer, jps_sim=None, agent_configs=None, enable_group_decisions=False, group_decision_min_size=3, llm_semaphore_limit=10, per_agent_timeout_secs=30.0, wait_nudge_enabled=False)
- Location: [evacusim/decision/decision_processor.py#L55](../../../evacusim/decision/decision_processor.py#L55)
- Summary: Initialize decision processor.
- Key calls: max, PromptCache, station_layout.get, logger.debug, int

#### _build_zones_section

- Signature: def _build_zones_section(self, zones, zone_id=None) -> str
- Location: [evacusim/decision/decision_processor.py#L165](../../../evacusim/decision/decision_processor.py#L165)
- Summary: Build the AVAILABLE ZONES block for the prompt.
- Key calls: self.station_layout.get, set, join, self._zones_hidden_for_zone.get, zone_labels.get, lines.append, title, z.replace

#### _get_valid_exits_section

- Signature: def _get_valid_exits_section(self, agent_id, observation, zone_id) -> str
- Location: [evacusim/decision/decision_processor.py#L191](../../../evacusim/decision/decision_processor.py#L191)
- Summary: Build a prompt section listing the valid exit names for this agent.
- Key calls: set, self._agent_cfg.get, cfg.get, _extract_visible_distance_ranks, _RE_BLOCKED_EXIT_LINE.finditer, _RE_VISUAL_FRAGMENTS.findall, _RE_VISIBLE_EXITS_LINE.findall, join, get, registry.get_all_ids
- Nested function count: 3

##### Nested helpers inside _get_valid_exits_section

###### _get_valid_exits_section._is_valid_departure

- Signature: def _is_valid_departure(eid) -> bool
- Location: [evacusim/decision/decision_processor.py#L210](../../../evacusim/decision/decision_processor.py#L210)
- Summary: No nested-function docstring summary available.

###### _get_valid_exits_section._extract_visible_distance_ranks

- Signature: def _extract_visible_distance_ranks(text) -> dict[str, int]
- Location: [evacusim/decision/decision_processor.py#L213](../../../evacusim/decision/decision_processor.py#L213)
- Summary: No nested-function docstring summary available.
- Key calls: _RE_VISIBLE_EXITS_LINE.search, _RE_EXITS_LINE_ENTRY.finditer, line_match.group, strip, _cat_rank.get, m.group

###### _get_valid_exits_section._sort_display_exits

- Signature: def _sort_display_exits(display_names) -> list[str]
- Location: [evacusim/decision/decision_processor.py#L235](../../../evacusim/decision/decision_processor.py#L235)
- Summary: No nested-function docstring summary available.
- Key calls: list, enumerate, visible_distance_rank.get, name.lower, sorted

#### _get_following_constraint_text

- Signature: def _get_following_constraint_text(self, observation) -> str
- Location: [evacusim/decision/decision_processor.py#L295](../../../evacusim/decision/decision_processor.py#L295)
- Summary: Build a hard movement constraint block when followers are detected.
- Key calls: _RE_CIRCULAR_FOLLOW.search, circular.group, constraints.append, _RE_FOLLOWER_SINGULAR.findall, _RE_FOLLOWER_PLURAL.search, join, re.findall, plural.group

#### _identify_zone

- Signature: def _identify_zone(self, position) -> str | None
- Location: [evacusim/decision/decision_processor.py#L346](../../../evacusim/decision/decision_processor.py#L346)
- Summary: Return the zone_id containing *position*, or None if not in any named zone.
- Key calls: _Point, float, self.action_translator.zones_polygons.items, poly.covers, poly.contains

#### _agent_is_on_escalator

- Signature: def _agent_is_on_escalator(self, agent_id) -> bool
- Location: [evacusim/decision/decision_processor.py#L373](../../../evacusim/decision/decision_processor.py#L373)
- Summary: Return True if the agent is currently inside an escalator corridor polygon.
- Key calls: self.jps_sim.get_agent_position, hasattr, getattr, Point, any, self.jps_sim.agent_levels.get, self.jps_sim.simulations.get, poly.contains, corridors.values

#### process_all_agents

- Signature: def process_all_agents(self, observations, current_sim_time, agent_ids=None) -> float
- Location: [evacusim/decision/decision_processor.py#L402](../../../evacusim/decision/decision_processor.py#L402)
- Summary: Process decision-making for all agents (parallel processing).
- Key calls: asyncio.run, logger.info, list, self._process_agents_parallel, self.action_translator.zones_polygons.keys, self.action_translator.zones.keys, len

#### _process_agents_parallel

- Signature: async def _process_agents_parallel(self, observations, zones, current_sim_time, agent_ids=None)
- Location: [evacusim/decision/decision_processor.py#L436](../../../evacusim/decision/decision_processor.py#L436)
- Summary: Process all agents in parallel using async/await.
- Key calls: asyncio.Lock, asyncio.Semaphore, getattr, list, self._process_single_agent, set, groups.items, self.perf_timer.measure, self.concordia_agents.keys, self.state_queries.get_agent_position

#### _process_single_agent

- Signature: async def _process_single_agent(self, agent_id, agent, observations, zones, current_sim_time, cycle_zone_cache=None)
- Location: [evacusim/decision/decision_processor.py#L554](../../../evacusim/decision/decision_processor.py#L554)
- Summary: Process a single agent's decision (async) with intelligent prompt caching.
- Key calls: self.state_queries.get_agent_position, self._agent_wait_since.get, observations.get, self._get_valid_exits_section, self._get_following_constraint_text, bool, self._agent_cfg.get, cfg.get, entity_lib.ActionSpec, self.prompt_cache.should_call_llm

#### _parse_json_response

- Signature: def _parse_json_response(self, response) -> dict[str, str]
- Location: [evacusim/decision/decision_processor.py#L901](../../../evacusim/decision/decision_processor.py#L901)
- Summary: Parse JSON response from agent, extracting reasoning components.
- Key calls: response.find, json.loads, data.get, logger.warning

#### on_agent_exit

- Signature: def on_agent_exit(self, agent_id) -> None
- Location: [evacusim/decision/decision_processor.py#L925](../../../evacusim/decision/decision_processor.py#L925)
- Summary: Clean up cache when agent exits simulation.
- Key calls: self.prompt_cache.clear_agent, self._agent_wait_since.pop, logger.debug

#### get_cache_statistics

- Signature: def get_cache_statistics(self) -> dict[str, Any]
- Location: [evacusim/decision/decision_processor.py#L936](../../../evacusim/decision/decision_processor.py#L936)
- Summary: Get statistics about LLM call optimization.
- Key calls: self.prompt_cache.get_statistics

#### log_cache_summary

- Signature: def log_cache_summary(self) -> None
- Location: [evacusim/decision/decision_processor.py#L954](../../../evacusim/decision/decision_processor.py#L954)
- Summary: Log summary of cache statistics to console and logger.
- Key calls: self.get_cache_statistics, logger.info

## Coverage Notes

- Nested helper functions documented in this file: 3
