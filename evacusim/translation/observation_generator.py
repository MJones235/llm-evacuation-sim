"""
Generates natural language observations from JuPedSim simulation state.

Converts geometric and simulation data into observations that Concordia
agents can reason about.
"""

import re
from typing import Any

from evacusim.utils.logger import get_logger
from evacusim.translation.crowd_analyzer import CrowdAnalyzer
from evacusim.translation.exit_name_registry import (
    build_registry_from_station_layout,
)
from evacusim.translation.observation_formatter import ObservationFormatter
from evacusim.translation.spatial_analyzer import SpatialAnalyzer

logger = get_logger(__name__)


class ObservationGenerator:
    """
    Generates natural language observations from JuPedSim simulation state.

    Converts geometric and simulation data into observations that Concordia
    agents can reason about.
    """

    def __init__(self, station_layout: dict[str, Any], jps_sim=None):
        """
        Initialize the observation generator.

        Args:
            station_layout: Station geometry and zone information (level 0 for multi-level)
            jps_sim: JuPedSim simulation instance (for multi-level exit access)
        """
        self.station_layout = station_layout
        self.exits = station_layout.get("exits", {})
        self.jps_sim = jps_sim

        # Build exit name registry for natural language display
        self.exit_registry = build_registry_from_station_layout(station_layout, jps_sim)

        # Initialize analyzers
        self.spatial_analyzer = SpatialAnalyzer(station_layout, self.exit_registry)
        self.crowd_analyzer = CrowdAnalyzer(self.exits)

        # Per-agent novelty tracking so prompts contain genuinely new information.
        self._last_event_signatures: dict[str, set[str]] = {}
        self._last_message_signatures: dict[str, set[str]] = {}

    def generate_observation(
        self,
        agent_id: str,
        position: tuple[float, float],
        nearby_agents: list[dict[str, Any]],
        events: list[str],
        sim_time: float,
        blocked_exits: set[str] | None = None,
        agent_injured: set[str] | None = None,
        agent_action: dict[str, str] | None = None,
        agent_level: str | None = None,
        agent_last_decision: dict[str, dict] | None = None,
        state_queries=None,
        received_messages: list[dict[str, Any]] | None = None,
        conversation_history: dict[str, list[dict]] | None = None,
    ) -> str:
        """
        Generate a natural language observation for an agent.

        Args:
            agent_id: ID of the observing agent
            position: Agent's current (x, y) position
            nearby_agents: List of nearby agents with their info
            events: Recent events (announcements, alarms, etc.)
            sim_time: Current simulation time
            blocked_exits: Set of blocked exit names (for visual observation)
            agent_injured: Set of injured agent IDs (physical capability dimension)
            agent_action: Dict of agent_id -> action ("moving"|"waiting")
            agent_level: Current level ID for multi-level simulations (e.g., "0", "-1")
            agent_last_decision: Dict of agent_id -> last decision (for memory)
            state_queries: SimulationStateQueries for position lookups (optional)
            received_messages: List of messages received from nearby agents
            conversation_history: Dict mapping other_agent_id to conversation history

        Returns:
            Natural language observation string
        """
        observations = []
        if blocked_exits is None:
            blocked_exits = set()
        if agent_injured is None:
            agent_injured = set()
        if agent_action is None:
            agent_action = {}
        if agent_last_decision is None:
            agent_last_decision = {}
        if received_messages is None:
            received_messages = []
        if conversation_history is None:
            conversation_history = {}

        # Surface only genuinely new information first.
        new_info_lines = self._format_new_information(agent_id, events, received_messages)
        observations.extend(new_info_lines)

        # Note: Station layout is now in agent formative memory, not observations
        # This keeps observations stable when nothing changes
        # Time is also omitted as it's not meaningful for agent decisions

        # EARLY CHECK: Detect circular following BEFORE other observations
        # This makes it the most salient new information
        followers = [a for a in nearby_agents if a.get("is_following_me", False)]

        # CIRCULAR FOLLOWING DETECTION: Describe observable pattern without prescribing behavior
        if agent_id in agent_last_decision:
            last_decision = agent_last_decision[agent_id]
            if (
                last_decision.get("target_type") == "agent"
                and last_decision.get("target_agent") is not None
            ):
                # We're following someone - check if they're also following us
                our_target = last_decision.get("target_agent")
                circular_detected = False
                for follower in followers:
                    if follower.get("id") == our_target:
                        # Describe the observable situation without prescribing action
                        target_person = our_target.replace("agent_", "Person ")
                        observations.append(
                            f"You are following {target_person}, and {target_person} is following YOU. "
                            f"Neither of you is heading toward an exit."
                        )
                        circular_detected = True
                        break

                if not circular_detected and followers:
                    # Not circular but still being followed - show normal follower alerts
                    follower_ids = [a.get("id").replace("agent_", "Person ") for a in followers]
                    if len(followers) == 1:
                        observations.append(f"⚠️ {follower_ids[0]} is trying to follow YOU.")
                    else:
                        follower_list = ", ".join(follower_ids)
                        observations.append(f"⚠️ {follower_list} are trying to follow YOU.")
            else:
                # Not following anyone - show normal follower alerts only
                if followers:
                    follower_ids = [a.get("id").replace("agent_", "Person ") for a in followers]
                    if len(followers) == 1:
                        observations.append(f"⚠️ {follower_ids[0]} is trying to follow YOU.")
                    else:
                        follower_list = ", ".join(follower_ids)
                        observations.append(f"⚠️ {follower_list} are trying to follow YOU.")
        else:
            # No previous decision recorded - show normal follower alerts
            if followers:
                follower_ids = [a.get("id").replace("agent_", "Person ") for a in followers]
                if len(followers) == 1:
                    observations.append(f"⚠️ {follower_ids[0]} is trying to follow YOU.")
                else:
                    follower_list = ", ".join(follower_ids)
                    observations.append(f"⚠️ {follower_list} are trying to follow YOU.")

        # Current location
        zone = self.spatial_analyzer.identify_zone(position)
        if zone == "unknown area" and agent_level:
            # Level 0 walkable_areas only covers the concourse; look up the
            # agent's actual level so platform agents get a sensible description.
            level_walkable = self.station_layout.get("walkable_areas_by_level", {}).get(
                str(agent_level), {}
            )
            if level_walkable:
                try:
                    from shapely.geometry import Point

                    pt = Point(position)
                    for area_name, polygon in level_walkable.items():
                        try:
                            if polygon.covers(pt) or polygon.contains(pt):
                                zone = area_name
                                break
                        except Exception:
                            pass
                except Exception:
                    pass

        # Map technical area/level names to human-readable descriptions.
        # Labels are defined in config under station.zone_labels.
        zone_labels = self.station_layout.get("zone_labels", {})
        zone_label = zone_labels.get(zone, zone)
        observations.append(f"You are in {zone_label}.")

        # Current situation: include own status (injury/waiting), but avoid repeating
        # full previous-decision recap every timestep.
        status_lines = ObservationFormatter.format_own_status(
            agent_id, agent_injured, agent_action, state_queries
        )
        observations.extend(status_lines)

        # Visual discovery of exits (efficient distance-based line of sight)
        visible_exits = self.spatial_analyzer.get_visible_exits(
            position, agent_level=agent_level, jps_sim=self.jps_sim, visual_range=25.0
        )
        if visible_exits:
            visible_names = [exit_info["name"] for exit_info in visible_exits]
            observations.append(f"Visible exits right now: {', '.join(visible_names)}.")
        else:
            observations.append("Visible exits right now: none.")

        # Crowd density (categorized to prevent constant LLM calls as people move)
        num_nearby = len(nearby_agents)
        density = CrowdAnalyzer.categorize_density(num_nearby)
        observations.append(f"The area is {density}.")

        # Nearby agent behaviors
        if nearby_agents:
            behaviors = self.crowd_analyzer.summarize_behaviors(nearby_agents, agent_injured)
            observations.append(behaviors)

            # Phase 5.1: List nearby agent IDs for targeting messages
            nearby_ids = ObservationFormatter.format_nearby_agent_ids(nearby_agents)
            observations.extend(nearby_ids)

        # Phase 5: Messages from nearby people
        message_lines = ObservationFormatter.format_received_messages(received_messages)
        observations.extend(message_lines)

        # Add conversation history context for active conversations
        if conversation_history:
            # Only show conversations with nearby people who have exchanged multiple messages
            active_conversations = []
            nearby_ids = {a.get("id") for a in nearby_agents}

            for other_agent_id, messages in conversation_history.items():
                if other_agent_id in nearby_ids and len(messages) >= 2:  # At least 2 exchanges
                    # Get last 3 messages in this conversation
                    recent = messages[-3:]
                    convo_summary = []
                    for m in recent:
                        direction = (
                            "You"
                            if m["from"] == agent_id
                            else other_agent_id.replace("agent_", "Person ")
                        )
                        convo_summary.append(f'{direction}: "{m["text"]}"')

                    active_conversations.append(
                        {
                            "other": other_agent_id.replace("agent_", "Person "),
                            "summary": " → ".join(convo_summary),
                        }
                    )

            if active_conversations:
                observations.append("Recent conversation context:")
                for convo in active_conversations[:2]:  # Max 2 to keep it concise
                    observations.append(f"  - With {convo['other']}: {convo['summary']}")

        # Visual observation of blocked exits (Phase 4.2: Realistic discovery)
        visible_blocked = self.spatial_analyzer.get_visible_blocked_exits(position, blocked_exits)
        blocked_lines = ObservationFormatter.format_blocked_exits(visible_blocked)
        observations.extend(blocked_lines)

        cleaned_lines = [line.strip() for line in observations if line and line.strip()]
        return "\n".join(cleaned_lines)

    def _format_new_information(
        self,
        agent_id: str,
        events: list[str],
        received_messages: list[dict[str, Any]],
    ) -> list[str]:
        """Format only new information since this agent's previous observation."""
        event_sigs = {" ".join(str(event).split()) for event in events[-3:]}
        msg_sigs = {
            f"{msg.get('from', 'unknown')}::{msg.get('text', '').strip()}"
            for msg in (received_messages or [])
            if msg.get("text", "").strip()
        }

        prev_events = self._last_event_signatures.get(agent_id, set())
        prev_msgs = self._last_message_signatures.get(agent_id, set())

        new_events = [evt for evt in sorted(event_sigs) if evt not in prev_events]
        new_msgs = [msig for msig in sorted(msg_sigs) if msig not in prev_msgs]

        self._last_event_signatures[agent_id] = event_sigs
        self._last_message_signatures[agent_id] = msg_sigs

        lines = ["What is NEW since your last decision:"]
        if not new_events and not new_msgs:
            lines.append("No significant new information.")
            return lines

        updates: list[str] = []
        updates.extend(new_events)

        for msg_sig in new_msgs[-3:]:
            sender, text = msg_sig.split("::", 1)
            sender_name = sender.replace("agent_", "Person ")
            updates.append(f'{sender_name} said: "{text}"')

        lines.append("; ".join(updates))

        return lines

    def _humanize_exit_crowd_keys(self, exit_crowds: dict[str, int]) -> dict[str, int]:
        """Translate technical exit IDs into display names and merge equivalent escalator IDs."""
        if not exit_crowds:
            return exit_crowds

        merged: dict[str, dict[str, Any]] = {}

        for exit_id, count in exit_crowds.items():
            canonical_key = self._canonical_exit_key(exit_id)
            display_name = (
                self.exit_registry.get_display_name(exit_id) if self.exit_registry else exit_id
            )

            if canonical_key not in merged:
                merged[canonical_key] = {"display_name": display_name, "count": count}
            else:
                merged[canonical_key]["count"] += count
                merged[canonical_key]["display_name"] = self._prefer_exit_label(
                    merged[canonical_key]["display_name"], display_name
                )

        return {
            data["display_name"]: data["count"]
            for data in sorted(merged.values(), key=lambda item: item["display_name"].lower())
        }

    def _canonical_exit_key(self, exit_id: str) -> str:
        """Map equivalent exit IDs (zone IDs vs escalator IDs) to one canonical key."""
        escalator_match = re.match(r"^escalator_([a-z])_(up|down)$", exit_id)
        if escalator_match:
            letter, direction = escalator_match.groups()
            return f"escalator_{letter}_{direction}"

        zone_escalator_match = re.match(r"^L[^_]+_esc_([a-z])_(up|down)$", exit_id)
        if zone_escalator_match:
            letter, direction = zone_escalator_match.groups()
            return f"escalator_{letter}_{direction}"

        return exit_id

    def _prefer_exit_label(self, current_label: str, candidate_label: str) -> str:
        """Prefer more descriptive display labels when combining equivalent exits."""
        current_specific = " to " in current_label.lower()
        candidate_specific = " to " in candidate_label.lower()

        if candidate_specific and not current_specific:
            return candidate_label
        if current_specific and not candidate_specific:
            return current_label

        if len(candidate_label) > len(current_label):
            return candidate_label
        return current_label
