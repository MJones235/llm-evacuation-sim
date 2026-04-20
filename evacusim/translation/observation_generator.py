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

        # Per-agent persistent exit knowledge.
        # An agent becomes aware of an exit when they see it (LOS) or hear about it
        # from another agent.  Knowledge is never forgotten once acquired.
        # Maps agent_id -> set of canonical exit keys.
        self.agent_known_exits: dict[str, set[str]] = {}

        # Canonical exit key -> human-readable display name (populated lazily from
        # get_visible_exits() results and the level-0 exit registry at init).
        self._canonical_to_display: dict[str, str] = {}
        # Case-insensitive lookup: any exit name variant -> canonical key.
        # Used to extract exit references from natural language messages.
        self._exit_name_to_canonical: dict[str, str] = {}
        self._build_exit_name_lookups()

    # ------------------------------------------------------------------
    # Exit knowledge helpers
    # ------------------------------------------------------------------

    def _build_exit_name_lookups(self) -> None:
        """Populate case-insensitive exit name → canonical key and reverse lookups.

        Only covers exits in the level-0 station_layout.  Multi-level exits
        discovered at runtime are added lazily via _update_canonical_display().
        """
        for raw_name in self.exits:
            canonical = self.spatial_analyzer._canonical_visible_exit_key(raw_name)
            if self.exit_registry:
                display = self.exit_registry.get_display_name(raw_name)
            else:
                display = raw_name
            self._exit_name_to_canonical[raw_name.lower()] = canonical
            self._exit_name_to_canonical[display.lower()] = canonical
            self._update_canonical_display(canonical, display)

    def _update_canonical_display(self, canonical: str, display: str) -> None:
        """Keep the most descriptive display name for *canonical* in the lookup."""
        existing = self._canonical_to_display.get(canonical, "")
        # Prefer names that contain " to " (e.g. "Escalator A (up to concourse)")
        existing_specific = " to " in existing.lower()
        candidate_specific = " to " in display.lower()
        if (
            canonical not in self._canonical_to_display
            or (candidate_specific and not existing_specific)
            or (candidate_specific == existing_specific and len(display) > len(existing))
        ):
            self._canonical_to_display[canonical] = display

    def _learn_exits_from_messages(self, agent_id: str, messages: list[dict[str, Any]]) -> None:
        """Scan received messages for exit name mentions and add them to known exits."""
        known = self.agent_known_exits.setdefault(agent_id, set())
        for msg in messages:
            text_lower = msg.get("text", "").lower()
            for name_key, canonical in self._exit_name_to_canonical.items():
                if name_key and name_key in text_lower:
                    known.add(canonical)

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
        inactive_exits: set[str] | None = None,
        known_blocked_exits: set[str] | None = None,
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

        # Attach sender_role to received messages so ObservationFormatter can label them.
        # The role is carried in nearby_agents[*]["role"] which ObservationCoordinator
        # populates from the shared agent_roles dict.
        _role_by_id = {a["id"]: a["role"] for a in nearby_agents if a.get("role") and a.get("id")}
        for msg in received_messages:
            sender = msg.get("from", "")
            if sender in _role_by_id and "sender_role" not in msg:
                msg["sender_role"] = _role_by_id[sender]

        # Surface only genuinely new information first.
        new_info_lines = self._format_new_information(agent_id, events, received_messages)
        observations.extend(new_info_lines)

        # Always repeat every currently-active event as a persistent reminder.
        # The novelty filter above only flags events as "new" once; without this
        # block the alarm (or any other ongoing situation) disappears from all
        # subsequent observations and fades from the agent's reasoning context.
        if events:
            observations.append(f"⚠️ Ongoing situation: {'; '.join(events)}")

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

        # --- Exit awareness (line-of-sight + persistent memory) ---

        # 1. Discover exits visible right now via LOS geometry.
        #    Pass blocked_exits so that currently-blocked exits are excluded from
        #    the visible list — they appear instead as blocked visual observations
        #    once the agent is within discovery range.
        visible_exits = self.spatial_analyzer.get_visible_exits(
            position, agent_level=agent_level, jps_sim=self.jps_sim,
            inactive_exits=inactive_exits,
            blocked_exits=blocked_exits if blocked_exits else None,
        )

        # 2. Update: learn exits seen this tick.
        known = self.agent_known_exits.setdefault(agent_id, set())
        visible_canonical: set[str] = set()
        for exit_info in visible_exits:
            canonical = exit_info["id"]
            visible_canonical.add(canonical)
            known.add(canonical)
            # Keep display name lookup up to date (picks up multi-level exits lazily).
            self._exit_name_to_canonical[exit_info["name"].lower()] = canonical
            self._update_canonical_display(canonical, exit_info["name"])

        # 3. Update: learn exits mentioned in messages from others.
        self._learn_exits_from_messages(agent_id, received_messages)

        # 4. Format observation lines.
        if visible_exits:
            # Group by distance category for a natural description.
            by_dist: dict[str, list[str]] = {"very close": [], "nearby": [], "visible in distance": []}
            for exit_info in visible_exits:
                cat = exit_info.get("distance", "visible in distance")
                by_dist.setdefault(cat, []).append(exit_info["name"])
            parts = []
            for cat in ("very close", "nearby", "visible in distance"):
                if by_dist[cat]:
                    parts.append(f"{', '.join(by_dist[cat])} ({cat})")
            observations.append(f"Exits visible right now: {'; '.join(parts)}.")
        else:
            observations.append("Exits visible right now: none.")

        # 5. Exits the agent knows about from prior observation but cannot see right now.
        recalled = known - visible_canonical
        # Suppress inactive exits (e.g. trains not yet arrived) from recalled list too.
        if inactive_exits:
            recalled = {c for c in recalled if c not in inactive_exits}
        # Filter recalled exits to only those accessible from the agent's current level.
        # This prevents agents who have transitioned levels from recalling exits that
        # only exist on another level (e.g. a platform agent recalling "Escalator A
        # (down to platforms)" which is only accessible from the concourse level).
        if agent_level and self.jps_sim and hasattr(self.jps_sim, "simulations"):
            level_sim = self.jps_sim.simulations.get(str(agent_level))
            if level_sim and hasattr(level_sim, "exit_manager"):
                level_exit_ids = set(level_sim.exit_manager.exit_coordinates.keys())
                recalled = {c for c in recalled if c in level_exit_ids}
        if recalled:
            recalled_names = sorted(
                self._canonical_to_display.get(c, c) for c in recalled
            )
            observations.append(
                f"Exits you know about but cannot currently see: {', '.join(recalled_names)}."
            )

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

        # Visual observation of blocked exits (Phase 4.2: Realistic discovery).
        # Proximity-based discovery: only exits within 8 m are newly visible.
        visible_blocked = self.spatial_analyzer.get_visible_blocked_exits(
            position, blocked_exits, agent_level=agent_level, jps_sim=self.jps_sim
        )
        # Merge with exits the agent already knows are blocked (persistent memory).
        # These are shown as "remembered" so the agent doesn't forget and oscillate
        # back toward blocked escalators after walking away from them.
        if known_blocked_exits:
            visible_names = {e["name"] for e in visible_blocked}
            registry = getattr(self, "exit_registry", None) or getattr(
                getattr(self, "action_translator", None), "exit_registry", None
            )
            for cid in known_blocked_exits:
                display = (
                    registry.get_display_name(cid) if registry else cid
                )
                if display not in visible_names:
                    visible_blocked.append({"name": display, "distance": "remembered"})
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
        """Format only new information since this agent's previous observation.

        Directive and PA messages are always treated as new — they are repeated
        intentionally and must not be silently dropped by the novelty filter.
        """
        event_sigs = {" ".join(str(event).split()) for event in events[-3:]}

        # Split messages: authority broadcasts (always fresh) vs regular chat
        _authority_types = {"directive", "pa"}
        authority_msgs = [
            m for m in (received_messages or [])
            if m.get("message_type") in _authority_types and m.get("text", "").strip()
        ]
        regular_msgs = [
            m for m in (received_messages or [])
            if m.get("message_type") not in _authority_types and m.get("text", "").strip()
        ]

        # Novelty filter applies only to regular chat messages.
        msg_sigs = {
            f"{msg.get('from', 'unknown')}::{msg.get('text', '').strip()}"
            for msg in regular_msgs
        }

        prev_events = self._last_event_signatures.get(agent_id, set())
        prev_msgs = self._last_message_signatures.get(agent_id, set())

        new_events = [evt for evt in sorted(event_sigs) if evt not in prev_events]
        new_msgs = [msig for msig in sorted(msg_sigs) if msig not in prev_msgs]

        self._last_event_signatures[agent_id] = event_sigs
        self._last_message_signatures[agent_id] = msg_sigs

        lines = ["What is NEW since your last decision:"]
        if not new_events and not new_msgs and not authority_msgs:
            lines.append("No significant new information.")
            return lines

        updates: list[str] = []
        updates.extend(new_events)

        # Authority broadcasts: shown every cycle, sender name uses their role
        for msg in authority_msgs[-2:]:
            sender = msg.get("from", "unknown")
            text = msg.get("text", "").strip()
            role = msg.get("sender_role")
            sender_name = role if role else sender.replace("agent_", "Person ")
            updates.append(f'{sender_name} (directing you) said: "{text}"')

        # Regular new chat messages
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
