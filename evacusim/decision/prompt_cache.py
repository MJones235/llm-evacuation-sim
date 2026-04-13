"""
Prompt Cache & Change Detection

Implements intelligent prompt caching to reduce unnecessary LLM calls by:
1. Creating Content-Aware Hashes: Hashes only semantically significant parts of prompts
2. Tracking Changes: Compares hashes across decision cycles
3. Detecting Significant Info: Identifies messages, events, or key state changes
4. Reusing Decisions: Returns cached decisions when prompt hasn't changed

Purpose: Only invoke LLM when "significant new information" arrives (messages, events,
         blocked exits), not for minor positional/observational updates.

Author: Developed to reduce Agent 12-style flip-flopping by ~60-70%
"""

import hashlib
import json
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Precompiled regex patterns used in _filter_observation (hot path)
# ---------------------------------------------------------------------------
_RE_PREV_DECISION = re.compile(
    r"(?:\[Your previous decision[^\]]+\].*?"
    r"|At t=\d+(?:\.\d+)?s, your previous decision was to .*?)"
    r"(?=You are in|You can see|The area|Nearby:|Nearby people:|Recent events:|$)",
    re.DOTALL | re.IGNORECASE,
)
_RE_YOU_REASONED = re.compile(
    r"You reasoned:\s.*?(?=You are in|You can see|The area|Nearby:|Nearby people:|Recent events:|$)",
    re.DOTALL | re.IGNORECASE,
)
_RE_YOU_ARE_IN = re.compile(r"You are in (the )?[\w\s]+\.", re.IGNORECASE)
_RE_ESC_DIRECTION = re.compile(r"(Escalator [A-Z])\s*\(going (?:up|down)\)")
_RE_EXIT_QUALIFIER = re.compile(
    r"(Escalator [A-Z]|[A-Z][a-z]+(?: [A-Z][a-z]+)* Street|Eldon Square)"
    r"(?:\s+(?:is|visible|nearby|to you|in the distance|very close|close|at a distance|"
    r"some distance away|quite close))+",
    re.IGNORECASE,
)
_RE_YOU_CAN_SEE_EXIT = re.compile(
    r"(?:You can see|You visible)\s+(Escalator [A-Z]|[A-Z][a-z]+(?: [A-Z][a-z]+)* Street)",
    re.IGNORECASE,
)
_RE_PROXIMITY_ADVERBS = re.compile(
    r"\b(very close|quite close|close by|close|nearby|near|in the distance|"
    r"at a distance|some distance away|a short distance|visible)\b",
    re.IGNORECASE,
)
_RE_AGENT_IDS = re.compile(r"\bagent_\d+\b(?:[,\s]+agent_\d+\b)*", re.IGNORECASE)
_RE_NEARBY_EMPTY = re.compile(r"\bNearby:\s*(?:[,\s]*)(\n|$)")
_RE_NEARBY_PEOPLE = re.compile(r"\bNearby people:\s*(?:[^\n]*)(\n|$)")
_RE_CROWD_COUNT = re.compile(r"\b\d+\s+(?:people|agents?)\b", re.IGNORECASE)
_RE_TIMESTAMP = re.compile(r"\bat t=[\d.]+s\b")
_RE_MOVEMENT_ADVERBS = re.compile(
    r"\b(purposefully|quickly|slowly|calmly|briskly)\b", re.IGNORECASE
)
_RE_MULTI_SPACE = re.compile(r"[ \t]+")
_RE_MULTI_NEWLINE = re.compile(r"\n\s*\n")


class PromptCache:
    """
    Content-aware prompt caching for agent decision-making.

    Tracks which parts of the prompt contain significant information that should
    trigger new LLM decisions (messages, events, route changes) vs. routine updates
    (minor position changes, nearby agent proximity fluctuations).
    """

    def __init__(self, enable_detailed_logging: bool = False, stable_skip_threshold: int = 5):
        """
        Initialize prompt cache.

        Args:
            enable_detailed_logging: If True, logs detailed change information
            stable_skip_threshold: After this many consecutive cache hits with no new
                messages/events, treat the agent as "stable" and apply an even more
                aggressive observation filter (ignores crowd-level fluctuations).
                Set to 0 to disable.  Default 5 (≈ 25 s at a 5 s decision interval).
        """
        self.agent_prompts: dict[str, dict[str, Any]] = {}  # agent_id -> {hash, content, timestamp}
        self.agent_decisions: dict[str, str] = {}  # agent_id -> cached decision JSON
        self.agent_change_history: dict[str, list[dict]] = {}  # agent_id -> list of changes
        self.enable_logging = enable_detailed_logging
        self.stable_skip_threshold = stable_skip_threshold
        # Per-agent count of consecutive decision cycles where the cache was hit
        # without any new messages/events ("stable" streak counter).
        self._consecutive_hits: dict[str, int] = {}

    def should_call_llm(
        self,
        agent_id: str,
        observation: str,
        action_spec_text: str,
        received_messages: list | None = None,
        blocked_exits: set | None = None,
        recent_events: list | None = None,
    ) -> tuple[bool, str | None]:
        """
        Determine if LLM call is needed based on prompt changes.

        Args:
            agent_id: Agent identifier
            observation: Current observation string for agent
            action_spec_text: The full action spec/prompt text
            received_messages: New messages for this agent (significant!)
            blocked_exits: Currently blocked exits (significant!)
            recent_events: Recent events affecting this agent (significant!)

        Returns:
            Tuple of (should_call_llm: bool, cached_decision: str | None)

        Example:
            should_call, cached_action = cache.should_call_llm(
                "agent_5",
                observation_str,
                prompt_text,
                received_messages=["Help!", "Are you okay?"],
                recent_events=[...]
            )
            if not should_call:
                return cached_action  # Reuse previous decision
        """
        # Extract semantically significant content
        significant_content = self._extract_significant_content(
            observation=observation,
            messages=received_messages,
            blocked_exits=blocked_exits,
            events=recent_events,
        )

        # Stable-agent optimisation (Opt 7a): when an agent has had N consecutive
        # cache hits with no external triggers (messages / events / blocked exits),
        # apply an extra level of filtering that strips crowd-level fluctuations too.
        # This prevents gradual crowd-count drift from re-triggering the LLM.
        hits = self._consecutive_hits.get(agent_id, 0)
        has_external_trigger = bool(received_messages or blocked_exits or recent_events)
        if self.stable_skip_threshold > 0 and hits >= self.stable_skip_threshold and not has_external_trigger:
            significant_content = self._filter_stable_agent_content(significant_content)

        # Create hash of significant content + prompt
        content_hash = self._compute_content_hash(action_spec_text, significant_content)

        # Check if this is the first decision for this agent
        if agent_id not in self.agent_prompts:
            if self.enable_logging:
                logger.info(f"{agent_id}: First decision, calling LLM")
            self._record_cache(agent_id, content_hash, action_spec_text, significant_content)
            self._consecutive_hits[agent_id] = 0
            return (True, None)

        # Compare with last decision
        last_hash = self.agent_prompts[agent_id].get("content_hash")
        last_content = self.agent_prompts[agent_id].get("significant_content", "")

        if content_hash == last_hash:
            # Prompt hasn't changed significantly — increment stability streak.
            if not has_external_trigger:
                self._consecutive_hits[agent_id] = hits + 1
            if self.enable_logging:
                logger.debug(f"{agent_id}: Prompt unchanged (hash match), reusing decision")
            cached = self.agent_decisions.get(agent_id)
            return (False, cached)

        # Significant change detected — reset stability streak.
        self._consecutive_hits[agent_id] = 0
        if self.enable_logging:
            change_info = self._describe_change_detailed(
                agent_id, last_hash, content_hash, last_content, significant_content
            )
            logger.info(f"{agent_id}: Significant change: {change_info}")
            self._track_change(agent_id, change_info, significant_content)

        self._record_cache(agent_id, content_hash, action_spec_text, significant_content)
        return (True, None)

    def cache_decision(self, agent_id: str, decision_json: str) -> None:
        """
        Store decision for future reuse if prompt unchanged.

        Args:
            agent_id: Agent identifier
            decision_json: JSON decision string to cache
        """
        self.agent_decisions[agent_id] = decision_json
        if self.enable_logging:
            logger.debug(f"{agent_id}: Decision cached for reuse")

    def get_cached_decision(self, agent_id: str) -> str | None:
        """
        Retrieve cached decision if available.

        Args:
            agent_id: Agent identifier

        Returns:
            Cached decision JSON string or None
        """
        return self.agent_decisions.get(agent_id)

    def clear_agent(self, agent_id: str) -> None:
        """
        Clear cache for specific agent (e.g., on exit).

        Args:
            agent_id: Agent identifier
        """
        self.agent_prompts.pop(agent_id, None)
        self.agent_decisions.pop(agent_id, None)
        self._consecutive_hits.pop(agent_id, None)
        if self.enable_logging:
            logger.debug(f"{agent_id}: Cache cleared")

    def get_statistics(self) -> dict[str, Any]:
        """
        Return cache statistics (useful for monitoring).

        Returns:
            Dictionary with cache statistics
        """
        return {
            "cached_agents": len(self.agent_prompts),
            "cached_decisions": len(self.agent_decisions),
            "change_history_size": sum(
                len(changes) for changes in self.agent_change_history.values()
            ),
        }

    # ==================== PRIVATE METHODS ====================

    def _filter_stable_agent_content(self, significant_content: str) -> str:
        """
        Apply extra normalisation for stable agents (those with a long hit streak).

        Removes "several" crowd descriptors added by _filter_observation, since even
        normalised crowd counts can fluctuate slightly as agents flow through zones.
        After this filter, only exit visibility / blocked exits / events cause a change.
        """
        # Strip lines that are observation-only (start with "OBS:") and replace
        # with a fixed placeholder so the hash cannot be affected by crowd churn.
        lines = significant_content.split("\n")
        cleaned = []
        for line in lines:
            if line.startswith("OBS:"):
                # Keep only exit-visibility fragments; drop crowd sentences.
                obs = line[4:]
                obs = _RE_CROWD_COUNT.sub("", obs)
                # Also strip any remaining "several" tokens left over from prior pass.
                obs = obs.replace("several", "")
                obs = _RE_MULTI_SPACE.sub(" ", obs).strip()
                cleaned.append(f"OBS:{obs}")
            else:
                cleaned.append(line)
        return "\n".join(cleaned)

    def _extract_significant_content(
        self,
        observation: str,
        messages: list | None = None,
        blocked_exits: set | None = None,
        events: list | None = None,
    ) -> str:
        """
        Extract only semantically significant parts of the prompt.

        Filters out volatile components (memory, exact positions) while keeping
        critical decision-making information (messages, events, situation).
        """
        significant_parts = []

        # Filter observation to remove memory/history component
        if observation:
            filtered_obs = self._filter_observation(observation)
            significant_parts.append(f"OBS:{filtered_obs}")

        # Include messages (IMPORTANT - agent received new info!)
        if messages:
            significant_parts.append(f"MSG:{json.dumps(sorted(messages))}")

        # Include blocked exits (IMPORTANT - route must change!)
        if blocked_exits:
            significant_parts.append(f"BLOCKED:{json.dumps(sorted(blocked_exits))}")

        # Include events (IMPORTANT - situation may have changed!)
        if events:
            event_summary = self._summarize_events(events)
            significant_parts.append(f"EVENTS:{event_summary}")

        return "\n".join(significant_parts)

    def _filter_observation(self, observation: str) -> str:
        """
        Filter observation to remove volatile components that shouldn't trigger new LLM calls.

        Removes:
        - Memory of previous decisions (e.g., "[Your previous decision at t=15s]...")
        - Current location/area ("You are in the concourse") — agent chose to move there,
          so changing location alone shouldn't re-trigger a decision
        - Distance qualifiers on exits — "Escalator B visible nearby" vs "visible in the
          distance" is NOT a substantial change; what matters is whether the exit is visible
        - Specific agent IDs (keep crowd level, not names)
        - Exact crowd counts ("5 people" → "several")
        - Timestamps within observations

        Keeps (these DO constitute a substantial change):
        - Which exits are visible (set, without distances)
        - Crowd level descriptors (empty / sparse / moderate / crowded)
        - Emergency alerts / evacuation events
        - Blocked/closed exits
        - Whether agent is waiting or moving
        - Injured agents nearby
        """
        filtered = observation

        # 1. Remove the previous-decision memory block (both wording styles).
        filtered = _RE_PREV_DECISION.sub("", filtered)

        # Remove explicit reasoning lines tied to previous-decision memory.
        filtered = _RE_YOU_REASONED.sub("", filtered)

        # 2. Remove current location sentence entirely.
        filtered = _RE_YOU_ARE_IN.sub("", filtered)

        # 3. Normalise exit visibility: strip direction/distance qualifiers.
        # Replace parenthetical direction tags on escalators.
        filtered = _RE_ESC_DIRECTION.sub(r"\1", filtered)
        # Collapse any distance/proximity qualifier after an exit name.
        filtered = _RE_EXIT_QUALIFIER.sub(r"\1 visible", filtered)
        # "You can see / You visible Escalator B" → "Escalator B visible"
        filtered = _RE_YOU_CAN_SEE_EXIT.sub(r"\1 visible", filtered)

        # 4. Remove all remaining distance/proximity adverbs.
        filtered = _RE_PROXIMITY_ADVERBS.sub("", filtered)

        # 5. Remove specific agent IDs; keep crowd-level descriptions.
        filtered = _RE_AGENT_IDS.sub("", filtered)
        filtered = _RE_NEARBY_EMPTY.sub("", filtered)
        filtered = _RE_NEARBY_PEOPLE.sub("", filtered)

        # 6. Normalise exact crowd counts to bands ("5 people" → "several").
        filtered = _RE_CROWD_COUNT.sub("several", filtered)

        # 7. Remove bare timestamps.
        filtered = _RE_TIMESTAMP.sub("", filtered)

        # 8. Normalise movement adverbs.
        filtered = _RE_MOVEMENT_ADVERBS.sub("", filtered)

        # 9. Collapse whitespace left by removals.
        filtered = _RE_MULTI_SPACE.sub(" ", filtered)
        filtered = _RE_MULTI_NEWLINE.sub("\n", filtered)
        filtered = filtered.strip()

        return filtered

    def _summarize_events(self, events: list) -> str:
        """
        Create a concise summary of events (reduces volatility).

        Focuses on event types/locations, not timestamps or details that change.
        """
        event_types = set()
        event_locations = set()

        for event in events:
            if isinstance(event, dict):
                event_types.add(event.get("type", "unknown"))
                if "location" in event:
                    event_locations.add(event.get("location", ""))

        summary_parts = []
        if event_types:
            summary_parts.append(f"types:{sorted(event_types)}")
        if event_locations:
            summary_parts.append(f"locations:{sorted(event_locations)}")

        return "|".join(summary_parts) if summary_parts else "no_events"

    def _compute_content_hash(self, prompt_text: str, significant_content: str) -> str:
        """
        Create hash of prompt + significant content.

        Uses SHA256 for collisions resistance.
        """
        combined = f"{prompt_text}\n---\n{significant_content}"
        return hashlib.sha256(combined.encode()).hexdigest()

    def _record_cache(
        self,
        agent_id: str,
        content_hash: str,
        prompt_text: str,
        significant_content: str,
    ) -> None:
        """Record cache entry for agent."""
        self.agent_prompts[agent_id] = {
            "content_hash": content_hash,
            "prompt_length": len(prompt_text),
            "content_preview": significant_content[:400],  # Store more for debugging
            "significant_content": significant_content,  # Store full for comparison
            "timestamp": None,  # Can be updated by caller if needed
        }

    def _describe_change(self, agent_id: str, old_hash: str, new_hash: str) -> str:
        """Generate human-readable description of what changed."""
        # Get old and new content previews for comparison
        old_preview = self.agent_prompts.get(agent_id, {}).get("content_preview", "")

        # Basic hash change info
        basic_info = f"hash {old_hash[:8]}... → {new_hash[:8]}..."

        # Try to identify what likely changed
        # (This is heuristic since we don't store full old content)
        change_hints = []

        # Check if this looks like a message-triggered change
        if "MSG:" in old_preview or "MSG:" in str(self.agent_prompts.get(agent_id, {})):
            change_hints.append("messages")

        # Check if blocked exits changed
        if "BLOCKED:" in old_preview:
            change_hints.append("blocked_exits")

        # Check if events changed
        if "EVENTS:" in old_preview:
            change_hints.append("events")

        if change_hints:
            return f"{basic_info} (likely: {', '.join(change_hints)})"
        else:
            return f"{basic_info} (observation changed)"

    def _describe_change_detailed(
        self, agent_id: str, old_hash: str, new_hash: str, old_content: str, new_content: str
    ) -> str:
        """
        Describe what changed between prompts with detailed content comparison.

        Args:
            agent_id: Agent identifier
            old_hash: Previous content hash
            new_hash: New content hash
            old_content: Previous significant content
            new_content: New significant content

        Returns:
            Human-readable change description with specific differences
        """
        old_preview = old_hash[:8] if old_hash else "none"
        new_preview = new_hash[:8] if new_hash else "none"

        # Parse content sections
        def parse_sections(content: str) -> dict:
            sections = {}
            for line in content.split("\n"):
                if line.startswith("FILTERED_OBS:"):
                    sections["obs"] = line[13:].strip()
                elif line.startswith("MSG:"):
                    sections["msg"] = line[4:].strip()
                elif line.startswith("BLOCKED:"):
                    sections["blocked"] = line[8:].strip()
                elif line.startswith("EVENT:"):
                    sections["event"] = line[6:].strip()
            return sections

        old_sec = parse_sections(old_content)
        new_sec = parse_sections(new_content)

        # Identify what changed
        changes = []

        if old_sec.get("obs") != new_sec.get("obs"):
            changes.append("observation")

        if old_sec.get("msg") != new_sec.get("msg"):
            old_msgs = old_sec.get("msg", "")
            new_msgs = new_sec.get("msg", "")
            if not old_msgs and new_msgs:
                changes.append(f"new_messages({len(new_msgs.split(','))})")
            elif old_msgs and not new_msgs:
                changes.append("messages_cleared")
            else:
                changes.append("messages_changed")

        if old_sec.get("blocked") != new_sec.get("blocked"):
            old_blocked = old_sec.get("blocked", "")
            new_blocked = new_sec.get("blocked", "")
            if not old_blocked and new_blocked:
                changes.append(f"exits_blocked({new_blocked})")
            elif old_blocked and not new_blocked:
                changes.append("exits_unblocked")
            else:
                changes.append("blocked_exits_changed")

        if old_sec.get("event") != new_sec.get("event"):
            changes.append("events")

        if not changes:
            changes.append("unknown")

        change_str = ", ".join(changes)
        return f"hash {old_preview}→{new_preview}: {change_str}"

    def _track_change(self, agent_id: str, change_desc: str, significant_content: str) -> None:
        """Track change in history for this agent."""
        if agent_id not in self.agent_change_history:
            self.agent_change_history[agent_id] = []

        self.agent_change_history[agent_id].append(
            {"description": change_desc, "content_preview": significant_content[:100]}
        )

        # Keep only last 20 changes per agent to avoid unbounded growth
        if len(self.agent_change_history[agent_id]) > 20:
            self.agent_change_history[agent_id] = self.agent_change_history[agent_id][-20:]


class SignificantChangeDetector:
    """
    Detects if a prompt change is truly "significant" (warrants LLM call).

    This is a more sophisticated detector that can score how important a change is.
    """

    @staticmethod
    def score_change_significance(
        old_observation: str,
        new_observation: str,
        received_messages: list | None = None,
        blocked_exits_changed: bool = False,
        recent_events: list | None = None,
    ) -> float:
        """
        Score how significant the change is (0.0 to 1.0).

        Returns:
            Significance score. 1.0 = definitely call LLM, 0.0 = reuse cached decision
        """
        score = 0.0

        # Messages are HIGH priority (agent received new info)
        if received_messages:
            score += 0.9

        # Blocked exits changed is HIGH priority (must reroute!)
        if blocked_exits_changed:
            score += 0.8

        # Recent events are MEDIUM-HIGH priority
        if recent_events:
            score += 0.6

        # Observation differences...
        # For now just check if it changed (not character-by-character diff)
        if old_observation != new_observation:
            # Only add small amount for observation changes
            # (many changes are just nearby agents moving)
            score += 0.1

        return min(score, 1.0)  # Cap at 1.0
