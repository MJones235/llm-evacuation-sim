"""Delivery of spoken messages to agents: PA announcements and staff directives.

Messages reach agents within a radius of the speaker (or everyone, for the
PA), optionally with different text per zone. Each agent's received messages
feed its next observation, and the LLM engine's prompt cache.
"""

from typing import Any

from evacusim.systems.messaging.conversation_tracker import ConversationTracker
from evacusim.systems.messaging.message_parser import MessageParser
from evacusim.utils.logger import get_logger

logger = get_logger(__name__)


class MessageSystem:
    """Delivers PA announcements and staff directives to agents.

    Args:
        default_radius: Reach of a directive when none is given (m).
    """

    def __init__(self, default_radius: float = 10.0):
        self.default_radius = default_radius
        self.agent_messages: dict[str, list[dict[str, Any]]] = {}  # agent_id -> received messages
        self.message_history: list[dict[str, Any]] = []  # every message delivered
        self.conversation_tracker = ConversationTracker()
        # Warning cues (for the rule-based engine), kept for the whole run:
        # station-wide ones (the alarm) and those each agent received.
        self._broadcast_cues: list[dict[str, Any]] = []
        self._agent_cues: dict[str, list[dict[str, Any]]] = {}

    def broadcast_cue(self, cue: dict[str, Any], source: str, current_sim_time: float) -> None:
        """Record a warning everyone perceives from now on (e.g. the alarm)."""
        self._broadcast_cues.append({"time": current_sim_time, "source": source, **cue})

    def cues_for(self, agent_id: str) -> list[dict[str, Any]]:
        """Every warning cue the agent has perceived, oldest first.

        Each is ``{time, source, strength, instruction}``; ``source`` is
        ``alarm``, ``pa`` or ``staff``.
        """
        cues = self._broadcast_cues + self._agent_cues.get(agent_id, [])
        return sorted(cues, key=lambda c: c["time"])

    def _record_cue(
        self,
        agent_id: str,
        source: str,
        current_sim_time: float,
        cue: dict[str, Any] | None,
        cues_by_zone: dict[str, dict[str, Any]] | None,
        zone: str | None,
    ) -> None:
        chosen = (cues_by_zone or {}).get(zone or "") or cue
        if chosen:
            self._agent_cues.setdefault(agent_id, []).append(
                {"time": current_sim_time, "source": source, **chosen}
            )

    def deliver_directive(  # noqa: PLR0913
        self,
        sender_id: str,
        message_text: str,
        sender_position: tuple[float, float],
        current_sim_time: float,
        state_queries: Any,
        exited_agents: set[str],
        radius: float | None = None,
        messages_by_zone: dict[str, str] | None = None,
        zone_id_for_agent_fn: Any | None = None,
        cue: dict[str, Any] | None = None,
        cues_by_zone: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        """Deliver a rule-based directive from a director agent to nearby agents.

        Staff broadcast unconditionally, at the cadence set by ``DirectorSystem``.

        When ``messages_by_zone`` and ``zone_id_for_agent_fn`` are both supplied,
        each recipient receives the message matched to their current zone (or the
        ``message_text`` default when their zone has no override).

        Args:
            sender_id: ID of the director agent sending the message.
            message_text: Default message text.
            sender_position: Current (x, y) position of the sender.
            current_sim_time: Current simulation time in seconds.
            state_queries: SimulationStateQueries for nearby-agent lookup.
            exited_agents: Set of agent IDs that have already evacuated.
            radius: Broadcast radius in metres (defaults to ``default_radius``).
            messages_by_zone: Optional mapping of zone_id → message text override.
            zone_id_for_agent_fn: Optional callable(agent_id) -> zone_id | None.
        """
        effective_radius = radius if radius is not None else self.default_radius
        nearby_agents = state_queries.get_nearby_agents(sender_id, effective_radius)
        recipient_ids = [
            a["id"] for a in nearby_agents if a["id"] != sender_id and a["id"] not in exited_agents
        ]

        if not recipient_ids:
            logger.debug(
                f"[directive] {sender_id} found no recipients within {effective_radius}m "
                f"at t={current_sim_time:.1f}s (nearby={len(nearby_agents)} agents on same level)"
            )
            return

        type_emoji = MessageParser.get_type_emoji("shout")
        delivered = 0
        for recipient_id in recipient_ids:
            # Resolve per-zone message override
            text = message_text
            zone = None
            if (messages_by_zone or cues_by_zone) and zone_id_for_agent_fn is not None:
                zone = zone_id_for_agent_fn(recipient_id)
            if messages_by_zone and zone and zone in messages_by_zone:
                text = messages_by_zone[zone]

            if not text:
                continue
            self._record_cue(recipient_id, "staff", current_sim_time, cue, cues_by_zone, zone)

            if recipient_id not in self.agent_messages:
                self.agent_messages[recipient_id] = []
            self.agent_messages[recipient_id].append(
                {
                    "time": current_sim_time,
                    "from": sender_id,
                    "text": text,
                    "message_type": "directive",
                }
            )
            self.conversation_tracker.track_message(sender_id, recipient_id, text, current_sim_time)
            delivered += 1

        if delivered:
            logger.info(
                f"{type_emoji} {sender_id} (directive) \u2192 {delivered} people: '{message_text}'"
            )
        else:
            logger.debug(
                f"[directive] {sender_id}: {len(recipient_ids)} candidate(s) found but all had empty text"
            )

    def deliver_pa(
        self,
        sender_label: str,
        message_text: str,
        current_sim_time: float,
        all_agent_ids: list[str],
        exited_agents: set[str],
        messages_by_zone: dict[str, str] | None = None,
        zone_id_for_agent_fn: Any | None = None,
        cue: dict[str, Any] | None = None,
        cues_by_zone: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        """Deliver a station-wide PA announcement to all agents.

        PA announcements are not radius-limited — they reach every active agent.
        When ``messages_by_zone`` is provided, each agent receives the message
        matching their zone.

        Args:
            sender_label: Human-readable source label (e.g. "PA system").
            message_text: Default broadcast text.
            current_sim_time: Current simulation time.
            all_agent_ids: List of all Concordia agent IDs.
            exited_agents: Set of agent IDs that have already evacuated.
            messages_by_zone: Optional per-zone message overrides.
            zone_id_for_agent_fn: Optional callable(agent_id) -> zone_id | None.
        """
        delivered = 0
        for agent_id in all_agent_ids:
            if agent_id in exited_agents:
                continue
            text = message_text
            zone = None
            if (messages_by_zone or cues_by_zone) and zone_id_for_agent_fn is not None:
                zone = zone_id_for_agent_fn(agent_id)
            if messages_by_zone:
                if zone and zone in messages_by_zone:
                    text = messages_by_zone[zone]
                elif not zone and "default" in messages_by_zone:
                    text = messages_by_zone["default"]
            if not text:
                continue
            self._record_cue(agent_id, "pa", current_sim_time, cue, cues_by_zone, zone)
            if agent_id not in self.agent_messages:
                self.agent_messages[agent_id] = []
            self.agent_messages[agent_id].append(
                {
                    "time": current_sim_time,
                    "from": sender_label,
                    "text": text,
                    "message_type": "pa",
                }
            )
            delivered += 1

        if delivered:
            logger.info(f"📢 PA ({sender_label}) → {delivered} agents: '{message_text}'")

    def get_received_messages(self, agent_id: str) -> list[dict[str, Any]]:
        """Get messages received by an agent and clear them."""
        messages = self.agent_messages.get(agent_id, [])
        self.agent_messages[agent_id] = []  # Clear after retrieval
        return messages

    def get_conversation_history(self, agent_id: str) -> dict[str, list[dict]]:
        """Get conversation history for an agent."""
        return self.conversation_tracker.get_conversation_history(agent_id)
