"""
Decision Processor

Handles agent decision-making logic including:
- Parallel processing of agent decisions using asyncio
- LLM calls with comprehensive prompts for action selection
- JSON response parsing and reasoning extraction
- Action translation and decision recording
- Route change detection and tracking
- Intelligent prompt caching to reduce redundant LLM calls

This module coordinates the cognitive layer (Concordia) decision-making process.
"""

import asyncio
import json
import re
from pathlib import Path
from string import Template
from typing import Any

from concordia.typing import entity as entity_lib
from shapely.geometry import Point

from evacusim.utils.logger import get_logger
from evacusim.decision.action_utils import extract_exit_name
from evacusim.decision.prompt_cache import PromptCache
from evacusim.concordia.azure_llm_concordia import llm_current_agent_id, llm_current_sim_time

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Precompiled regex patterns (module-level to avoid per-call recompilation)
# ---------------------------------------------------------------------------
_RE_VISUAL_CLOSE = re.compile(r"You can see\s+([^\.]+?)\s+very close to you\.")
_RE_VISUAL_NEARBY = re.compile(r"You can see\s+([^\.]+?)\s+nearby\.")
_RE_VISUAL_DISTANCE = re.compile(r"You can see\s+([^\.]+?)\s+in the distance\.")
_RE_VISUAL_FRAGMENTS = re.compile(r"You can see\s+([^\.]+)\.")
# Matches the observation format: "Exits visible right now: Name (cat); ...."
_RE_VISIBLE_EXITS_LINE = re.compile(r"Exits visible right now:\s*([^\.]+)\.")
# Parses each semicolon-delimited entry: "Exit Name (going up) (nearby)"
_RE_EXITS_LINE_ENTRY = re.compile(r"(.+?)\s+\((very close|nearby|visible in distance)\)")
# Matches blocked exit lines from ObservationFormatter.format_blocked_exits
_RE_BLOCKED_EXIT_LINE = re.compile(r"The (.+?) appears blocked or obstructed")
_RE_CIRCULAR_FOLLOW = re.compile(
    r"You are following (Person (\w+)), and Person \2 is following YOU"
)
_RE_FOLLOWER_SINGULAR = re.compile(
    r"\u26a0\ufe0f (Person (\w+)) is trying to follow YOU"
)
_RE_FOLLOWER_PLURAL = re.compile(r"\u26a0\ufe0f (.+?) are trying to follow YOU")


class DecisionProcessor:
    """Processes agent decision-making with parallel LLM calls."""

    def __init__(
        self,
        concordia_agents: dict[str, entity_lib.Entity],
        exited_agents: set[str],
        action_translator,
        action_executor,
        message_system,
        state_queries,
        station_layout: dict[str, Any],
        agent_decisions: dict[str, dict[str, Any]],
        agent_destinations: dict[str, str],
        last_observations: dict[str, str],
        last_actions: dict[str, str],
        perf_timer,
        jps_sim=None,
        agent_configs: list[dict] | None = None,
        enable_group_decisions: bool = False,
        group_decision_min_size: int = 3,
        llm_semaphore_limit: int = 10,
        per_agent_timeout_secs: float | None = 30.0,
        min_redecision_interval_secs: float = 0.0,
        wait_nudge_enabled: bool = False,
        decision_prompt_template_path: str | None = None,
    ):
        """
        Initialize decision processor.

        Args:
            concordia_agents: Dict of agent_id -> Concordia entity
            exited_agents: Set of agent IDs who have exited
            action_translator: ActionTranslator for NL to JuPedSim commands
            action_executor: ActionExecutor for applying actions
            message_system: MessageSystem for agent communication
            state_queries: Simulation state query interface
            station_layout: Station geometry and exit information
            agent_decisions: Dict tracking all agent decisions
            agent_destinations: Dict of agent_id -> current exit name
            last_observations: Cache of last observations for change detection
            last_actions: Cache of last actions to reuse
            perf_timer: Performance monitoring timer
            jps_sim: JuPedSim simulation instance (for multi-level support)
        """
        self.concordia_agents = concordia_agents
        self.exited_agents = exited_agents
        self.action_translator = action_translator
        self.action_executor = action_executor
        self.message_system = message_system
        self.state_queries = state_queries
        self.station_layout = station_layout
        self.agent_decisions = agent_decisions
        self.agent_destinations = agent_destinations
        self.last_observations = last_observations
        self.last_actions = last_actions
        self.perf_timer = perf_timer
        self.jps_sim = jps_sim
        self._agent_cfg: dict[str, dict] = {cfg["id"]: cfg for cfg in (agent_configs or [])}
        self.enable_group_decisions = enable_group_decisions
        self.group_decision_min_size = group_decision_min_size
        # NOTE: _state_lock and _llm_semaphore are NOT created here.
        # asyncio primitives bind to the event loop that is current at creation
        # time.  process_all_agents() calls asyncio.run() each decision cycle,
        # which creates a fresh event loop every time.  Creating these objects in
        # __init__ would bind them to a loop that is later closed, causing the
        # "bound to a different event loop" RuntimeError on the second cycle.
        # Instead they are created lazily at the top of _process_agents_parallel,
        # which always runs inside the correct (current) loop.
        self._state_lock: asyncio.Lock | None = None
        self._llm_semaphore: asyncio.Semaphore | None = None
        # Configurable concurrency cap — alter before running if needed.
        self._llm_semaphore_limit: int = max(1, int(llm_semaphore_limit))
        # Per-agent wall-clock timeout (seconds).  If an agent's full decision
        # pipeline (including the LLM call) exceeds this limit the task is
        # cancelled and a warning is logged.  The agent keeps their existing
        # JuPedSim waypoint so the simulation is not blocked.
        # Set to None to disable (matches old behaviour).
        self._per_agent_timeout_secs: float | None = per_agent_timeout_secs
        # Optional simulation-time throttle for repeated LLM decisions when
        # observations still report no significant new information.
        self._min_redecision_interval_secs: float = max(0.0, float(min_redecision_interval_secs))
        self._last_llm_decision_time: dict[str, float] = {}

        # Optional wait-duration nudge in prompt context.
        self._wait_nudge_enabled: bool = wait_nudge_enabled
        self._decision_prompt_template_path = decision_prompt_template_path
        self._decision_prompt_template = self._load_decision_prompt_template(
            decision_prompt_template_path
        )

        # Escalator deferral guardrail: if an agent stays inside escalator
        # departure geometry without making positional progress for too long,
        # stop deferring decisions so the agent can reconsider route choice.
        self._escalator_deferral_timeout_secs: float = 12.0
        self._escalator_progress_threshold_m: float = 0.4
        self._escalator_deferral_state: dict[str, dict[str, Any]] = {}

        # Initialize prompt cache for intelligent LLM call reduction
        self.prompt_cache = PromptCache(enable_detailed_logging=True)
        self.llm_calls_skipped = 0  # Statistics tracking
        self.llm_calls_made = 0

        # Per-agent persistent goal text (plain string, updated via goal_update field).
        # Seeded from agent_cfg["initial_goal"] on first decision.
        self.agent_goals: dict[str, str] = {}

        # Tracks the sim-time when each agent last issued a non-wait action.
        # Used to inject an escalating wait-duration nudge into the observation so
        # that the prompt hash changes every minute, busting the cache and forcing
        # the LLM to re-evaluate agents stuck in a repetitive-wait deadlock.
        self._agent_wait_since: dict[str, float] = {}

        # Config-driven exit knowledge — replaces hardcoded level-number comparisons.
        # All keys are defined in station.yaml under the station: section.
        self._arrival_exits_by_zone: dict[str, list[str]] = station_layout.get(
            "arrival_exits_by_zone", {}
        )
        self._zone_known_exits: dict[str, dict[str, list[str]]] = station_layout.get(
            "zone_known_exits_by_profile", {}
        )
        self._zones_hidden_for_zone: dict[str, list[str]] = station_layout.get(
            "zones_hidden_for_zone", {}
        )
        self._zone_goal_keywords: dict[str, list[str]] = station_layout.get(
            "zone_goal_keywords", {}
        )
        self._exit_semantic_tags: dict[str, list[str]] = station_layout.get(
            "exit_semantic_tags", {}
        )
        self._goal_semantic_policies: list[dict[str, Any]] = station_layout.get(
            "goal_semantic_policies", []
        )

        logger.debug("DecisionProcessor initialized for parallel async processing")

    def _load_decision_prompt_template(self, template_path: str | None) -> Template:
        """Load the decision prompt template from disk.

        Raises:
            FileNotFoundError: If the configured template file does not exist.
            RuntimeError: If the template file cannot be read.
        """
        default_path = Path(__file__).with_name("templates") / "decision_prompt.template.txt"
        chosen_path = Path(template_path).expanduser() if template_path else default_path

        if not chosen_path.exists():
            raise FileNotFoundError(
                f"Decision prompt template file not found: {chosen_path}"
            )

        try:
            template_text = chosen_path.read_text(encoding="utf-8")
        except Exception as e:
            raise RuntimeError(
                f"Failed to read decision prompt template from {chosen_path}: {e}"
            ) from e

        logger.info(f"Loaded decision prompt template from {chosen_path}")
        return Template(template_text)

    def _render_decision_prompt(
        self,
        *,
        age: str,
        gender: str,
        personality_profile: str,
        last_decision_summary: str,
        situation_framing: str,
        zone_option: str,
        zone_name_hint: str,
        zone_constraint: str,
        zones_section: str,
        following_constraint_text: str,
        valid_exits_text: str,
    ) -> str:
        """Render the decision prompt text from the configured template."""
        return self._decision_prompt_template.safe_substitute(
            age=age,
            gender=gender,
            personality_profile=personality_profile,
            last_decision_summary=last_decision_summary,
            situation_framing=situation_framing,
            zone_option=zone_option,
            zone_name_hint=zone_name_hint,
            zone_constraint=zone_constraint,
            zones_section=zones_section,
            following_constraint_text=following_constraint_text,
            valid_exits_text=valid_exits_text,
        )

    def _build_zones_section(self, zones: list[str], zone_id: str | None = None) -> str:
        """Build the AVAILABLE ZONES block for the prompt.

        Uses zone_labels from station config to show human-readable names alongside
        the technical zone_name the LLM must copy verbatim into its JSON response.
        Returns an empty string when no zones are configured.

        Zones that cannot be reached directly without first using an exit (e.g. platform
        zones from the concourse) are hidden via the zones_hidden_for_zone config key,
        keeping the list meaningful for reorientation.
        """
        if not zones:
            return ""
        zone_labels: dict[str, str] = self.station_layout.get("zone_labels", {})
        hidden = set(self._zones_hidden_for_zone.get(zone_id or "", []))
        lines = []
        for z in zones:
            if z in hidden:
                continue
            label = zone_labels.get(z, z.replace("_", " ").title())
            lines.append(f"  \u2022 {label} (zone_name='{z}')")
        if not lines:
            return ""
        bullets = "\n".join(lines)
        return "\nAvailable zones:\n" f"{bullets}\n\n"

    def _get_valid_exits_section(self, agent_id: str, observation: str, zone_id: str | None) -> str:
        """Build a prompt section listing the valid exit names for this agent.

        Uses config-driven knowledge tables rather than hard-coded level numbers:
        - arrival_exits_by_zone: exits that arrive INTO this zone (filtered out)
        - zone_known_exits_by_profile: exits recalled from memory per zone/profile

                Commuters recall memorized exits for their current zone.
                Novices rely on what they can see; they have a small fallback list of
                exits they could read from signs (e.g. named street exits on the concourse).

                Goal-aware routing is handled through config-driven prompt guidance:
                - station.exit_semantic_tags: map of exit_id -> semantic tags
                - station.goal_semantic_policies: rules that map goal keywords to
                    preferred/discouraged exit tags in specific zones
                This keeps behavior reusable across geometries without hardcoded IDs.
        """
        registry = self.action_translator.exit_registry
        valid_ids = set(registry.get_all_ids())
        cfg = self._agent_cfg.get(agent_id, {})
        profile = cfg.get("knowledge_profile", "novice")

        goal_text = self.agent_goals.get(agent_id, "")
        goal_policy = self._get_goal_semantic_policy(zone_id, goal_text)

        # Exits that arrive INTO this zone are one-way arrivals, not departure choices.
        arrival_exits = set(self._arrival_exits_by_zone.get(zone_id or "", []))

        def _is_valid_departure(eid: str) -> bool:
            if eid in arrival_exits:
                return False
            return True

        # Optional goal-driven ranking preferences from station config.
        preferred_ids: set[str] = set()
        discouraged_ids: set[str] = set()
        if goal_policy:
            prefer_tags = {
                str(tag).strip().lower()
                for tag in goal_policy.get("prefer_exit_tags", [])
                if str(tag).strip()
            }
            avoid_tags = {
                str(tag).strip().lower()
                for tag in goal_policy.get("avoid_exit_tags", [])
                if str(tag).strip()
            }
            for eid, tags in self._exit_semantic_tags.items():
                if eid not in valid_ids or not _is_valid_departure(eid):
                    continue
                tag_set = {str(tag).strip().lower() for tag in (tags or []) if str(tag).strip()}
                if prefer_tags and (tag_set & prefer_tags):
                    preferred_ids.add(eid)
                if avoid_tags and (tag_set & avoid_tags):
                    discouraged_ids.add(eid)

        def _extract_visible_distance_ranks(text: str) -> dict[str, int]:
            ranks: dict[str, int] = {}
            _cat_rank = {"very close": 0, "nearby": 1, "visible in distance": 2}
            line_match = _RE_VISIBLE_EXITS_LINE.search(text)
            if line_match:
                for m in _RE_EXITS_LINE_ENTRY.finditer(line_match.group(1)):
                    name = m.group(1).strip()
                    rank = _cat_rank.get(m.group(2), 99)
                    if name and name not in ranks:
                        ranks[name] = rank
            return ranks

        visible_distance_rank = _extract_visible_distance_ranks(observation)

        # Exits the agent currently knows are blocked (could be from proximity
        # discovery this tick, or from persistent remembered knowledge injected
        # by ObservationCoordinator).  These are stripped from every exit list so
        # the agent never chooses a confirmed-blocked exit regardless of profile.
        blocked_display: set[str] = set()
        for _m in _RE_BLOCKED_EXIT_LINE.finditer(observation):
            blocked_display.add(_m.group(1).strip())

        display_to_id: dict[str, str] = {}
        for _eid in valid_ids:
            if not _is_valid_departure(_eid):
                continue
            _name = registry.get_display_name(_eid)
            if _name not in display_to_id:
                display_to_id[_name] = _eid

        def _sort_display_exits(display_names: list[str]) -> list[str]:
            indexed = list(enumerate(display_names))

            def key_fn(item: tuple[int, str]) -> tuple[int, int, int, str]:
                idx, name = item
                is_visible = 0 if name in visible_distance_rank else 1
                dist_rank = visible_distance_rank.get(name, 99)
                eid = display_to_id.get(name)
                preference_rank = 1
                if eid in preferred_ids:
                    preference_rank = 0
                elif eid in discouraged_ids:
                    preference_rank = 2
                return (is_visible, dist_rank, preference_rank, idx, name.lower())

            return [name for _, name in sorted(indexed, key=key_fn)]

        departure_ids = [eid for eid in valid_ids if _is_valid_departure(eid)]
        semantics_text = self._build_exit_semantics_text(goal_policy, departure_ids)

        # ── Commuter: recall memorized exits for this zone ───────────────────
        if profile == "commuter":
            mem_ids = self._zone_known_exits.get(zone_id or "", {}).get("commuter", [])
            exits = [
                registry.get_display_name(eid)
                for eid in mem_ids
                if eid in valid_ids
                and _is_valid_departure(eid)
                and registry.get_display_name(eid) not in blocked_display
            ]
            exits = _sort_display_exits(exits)
            if exits:
                bullets = "".join(f"\n  \u2022 {e}" for e in exits)
                return "\nAllowed exits now:" f"{bullets}\n" + semantics_text
            # Fall through to visible-exit check if no memorized exits for this zone

        # ── Visible exits (both profiles) ────────────────────────────────────
        # Restrict to explicit visual sentences to avoid false positives.
        visual_fragments = _RE_VISUAL_FRAGMENTS.findall(observation)
        visual_fragments += _RE_VISIBLE_EXITS_LINE.findall(observation)
        visual_text = " ".join(visual_fragments)

        all_departure = [
            registry.get_display_name(eid) for eid in valid_ids if _is_valid_departure(eid)
        ]
        visible = [n for n in all_departure if n in visual_text and n not in blocked_display]
        if visible:
            visible = _sort_display_exits(visible)
            bullets = "".join(f"\n  \u2022 {n}" for n in visible)
            return "\nVisible exits right now:" f"{bullets}\n" + semantics_text

        # ── No visible exits: check profile's memorized fallback ─────────────
        fallback_ids = self._zone_known_exits.get(zone_id or "", {}).get(profile, [])
        fallback = [
            registry.get_display_name(eid)
            for eid in fallback_ids
            if eid in valid_ids
            and _is_valid_departure(eid)
            and registry.get_display_name(eid) not in blocked_display
        ]
        if fallback:
            bullets = "".join(f"\n  \u2022 {n}" for n in fallback)
            return "\nAllowed exits now:" f"{bullets}\n" + semantics_text

        return semantics_text + (
            "\nVisible exits right now: none.\n"
            "Do NOT use target_type='exit'. Follow someone or wait.\n\n"
        )

    def _get_goal_semantic_policy(
        self, zone_id: str | None, goal_text: str
    ) -> dict[str, Any] | None:
        """Return the first matching goal-to-exit semantic policy from config.

        Policy schema (all keys optional except when_goal_contains_any):
          - when_goal_contains_any: list[str]
          - applies_in_zones: list[str]
          - prefer_exit_tags: list[str]
          - avoid_exit_tags: list[str]
          - instruction: str
        """
        if not goal_text or not self._goal_semantic_policies:
            return None

        goal_lower = goal_text.lower()
        zone_lower = (zone_id or "").lower()

        for policy in self._goal_semantic_policies:
            if not isinstance(policy, dict):
                continue

            keywords = [
                str(k).strip().lower()
                for k in policy.get("when_goal_contains_any", [])
                if str(k).strip()
            ]
            if not keywords:
                continue
            if not any(keyword in goal_lower for keyword in keywords):
                continue

            zones = {
                str(z).strip().lower()
                for z in policy.get("applies_in_zones", [])
                if str(z).strip()
            }
            if zones and zone_lower not in zones:
                continue

            return policy

        return None

    def _build_exit_semantics_text(
        self,
        goal_policy: dict[str, Any] | None,
        candidate_exit_ids: list[str],
    ) -> str:
        """Build an optional prompt block with goal-driven exit semantics guidance."""
        if not goal_policy:
            return "\n"

        registry = self.action_translator.exit_registry
        candidate_ids = set(candidate_exit_ids)

        prefer_tags = {
            str(tag).strip().lower()
            for tag in goal_policy.get("prefer_exit_tags", [])
            if str(tag).strip()
        }
        avoid_tags = {
            str(tag).strip().lower()
            for tag in goal_policy.get("avoid_exit_tags", [])
            if str(tag).strip()
        }

        preferred_names: list[str] = []
        avoided_names: list[str] = []
        for eid, tags in self._exit_semantic_tags.items():
            if eid not in candidate_ids:
                continue
            tag_set = {str(tag).strip().lower() for tag in (tags or []) if str(tag).strip()}
            if prefer_tags and (tag_set & prefer_tags):
                preferred_names.append(registry.get_display_name(eid))
            if avoid_tags and (tag_set & avoid_tags):
                avoided_names.append(registry.get_display_name(eid))

        instruction = str(goal_policy.get("instruction", "")).strip()
        lines = ["\nGoal-aligned exit guidance:"]
        if instruction:
            lines.append(f"- {instruction}")
        if preferred_names:
            shown = ", ".join(sorted(set(preferred_names))[:8])
            lines.append(f"- Prefer exits like: {shown}")
        if avoided_names:
            shown = ", ".join(sorted(set(avoided_names))[:8])
            lines.append(f"- Avoid exits like: {shown} (unless your goal is to leave the station)")

        if len(lines) == 1:
            return "\n"
        return "\n" + "\n".join(lines) + "\n\n"

    def _get_following_constraint_text(self, observation: str) -> str:
        """Build a hard movement constraint block when followers are detected.

        Two cases:
        1. Circular following — both agents follow each other: force exit choice.
        2. One-way follower   — someone is following this agent: forbid targeting them.
        """
        constraints = []

        # Case 1: circular — "You are following Person X, and Person X is following YOU."
        circular = _RE_CIRCULAR_FOLLOW.search(observation)
        if circular:
            person_label = circular.group(1)  # e.g. "Person 46"
            agent_id_str = f"agent_{circular.group(2)}"  # e.g. "agent_46"
            constraints.append(
                f"\n\u26d4 CIRCULAR FOLLOWING — BREAK THE LOOP: You and {person_label} are "
                f"following each other. Neither of you is heading toward an exit. "
                f"You MUST do something different this turn. "
                f"Do NOT set target_agent='{agent_id_str}'. "
                f"Choose one of: (a) target_type='exit' if an exit is listed in VALID EXIT OPTIONS, "
                f"(b) target_type='agent' toward a DIFFERENT person who is heading toward an exit, "
                f"or (c) target_type='current_position' to stop and reorient."
            )
        else:
            # Case 2: one-way followers — "⚠\ufe0f Person X is trying to follow YOU."
            followers = _RE_FOLLOWER_SINGULAR.findall(observation)
            # Also handle plural: "Person X, Person Y are trying to follow YOU"
            if not followers:
                plural = _RE_FOLLOWER_PLURAL.search(observation)
                if plural:
                    followers = [
                        (f"Person {m}", m) for m in re.findall(r"Person (\w+)", plural.group(1))
                    ]
            if followers:
                names = ", ".join(p for p, _ in followers)
                avoid = ", ".join(f"'agent_{n}'" for _, n in followers)
                constraints.append(
                    f"\n\u26a0\ufe0f FOLLOWER RULE: {names} is following YOU — you are their leader. "
                    f"Do NOT set target_agent={avoid}. "
                    f"You must lead: choose target_type='exit' using your own knowledge, "
                    f"or follow a DIFFERENT person who is actually heading toward an exit."
                )

        if constraints:
            return (
                "\n\u2550\u2550\u2550 MOVEMENT CONSTRAINTS (READ FIRST) \u2550\u2550\u2550"
                + "".join(constraints)
                + "\n\n"
            )
        return ""

    def _identify_zone(self, position: tuple[float, float]) -> str | None:
        """Return the zone_id containing *position*, or None if not in any named zone."""
        try:
            from shapely.geometry import Point as _Point

            pt = _Point(position)
            # Collect all matching zones, then return the one with the smallest
            # area.  This ensures that specific zones (e.g. "platform_1") take
            # priority over coarse background polygons (e.g. "level_-1") that
            # may cover the entire level and would otherwise shadow every
            # specific zone for agents who happen to be inside both.
            best_zone: str | None = None
            best_area: float = float("inf")
            for zone_id, poly in self.action_translator.zones_polygons.items():
                try:
                    if poly.covers(pt) or poly.contains(pt):
                        area = poly.area
                        if area < best_area:
                            best_area = area
                            best_zone = zone_id
                except Exception:
                    pass
            return best_zone
        except Exception:
            pass
        return None

    def _agent_is_on_escalator(self, agent_id: str) -> bool:
        """Return True if the agent is actively traversing an escalator departure path.

        Only **departure** zones and escalator corridors warrant deferral: an
        agent in those areas is physically mid-journey on an escalator and
        interrupting them would stall the flow.

        **Arrival** zones are explicitly excluded: an agent in an arrival zone
        has already completed the escalator journey and is awaiting a fresh
        decision.  Deferring them there is the primary cause of post-transfer
        blocking because they end up silently skipped every cycle.
        """
        if not self.jps_sim:
            return False
        pos = self.jps_sim.get_agent_position(agent_id)
        if pos is None:
            return False
        level_id = None
        if hasattr(self.jps_sim, "agent_levels"):
            level_id = self.jps_sim.agent_levels.get(agent_id)
        # For multi-level simulations look up the per-level sim; for single-level use jps_sim.
        if level_id is not None and hasattr(self.jps_sim, "simulations"):
            sim = self.jps_sim.simulations.get(level_id)
        else:
            sim = self.jps_sim
        if sim is None:
            return False
        p = Point(pos)

        # Corridor check — any corridor is a departure path regardless of role.
        corridors = getattr(getattr(sim, "geometry_manager", None), "escalator_corridors", {})
        for poly in corridors.values():
            if poly.covers(p) or poly.contains(p):
                return True

        # Transfer-zone check — departure zones only (multi-level).
        controller = getattr(self.jps_sim, "escalator_controller", None)
        if controller is not None and level_id is not None:
            for ep in controller.registry.endpoints_by_level.get(str(level_id), []):
                if ep.role != "departure":
                    # Arrival zones are skipped: agents there need decisions, not deferral.
                    continue
                zone_poly = controller.get_zone_polygon(ep.transfer_zone_name)
                if zone_poly is not None and (zone_poly.covers(p) or zone_poly.contains(p)):
                    return True

        return False

    def _should_defer_escalator_decision(
        self,
        agent_id: str,
        current_sim_time: float,
        position: tuple[float, float],
    ) -> bool:
        """Return True when escalator-context decision deferral should continue.

        Agents can queue in escalator departure geometry during congestion.
        If they are not making progress for too long, indefinite deferral causes
        a deadlock where they never re-evaluate alternatives. This guardrail
        allows periodic re-decisions only for stalled agents.
        """
        state = self._escalator_deferral_state.get(agent_id)
        if state is None:
            self._escalator_deferral_state[agent_id] = {
                "entered_time": current_sim_time,
                "last_progress_time": current_sim_time,
                "last_position": position,
            }
            return True

        last_pos = state.get("last_position", position)
        moved = ((position[0] - last_pos[0]) ** 2 + (position[1] - last_pos[1]) ** 2) ** 0.5
        if moved >= self._escalator_progress_threshold_m:
            state["last_progress_time"] = current_sim_time
            state["last_position"] = position
            return True

        state["last_position"] = position
        stalled_for = current_sim_time - float(state.get("last_progress_time", current_sim_time))
        if stalled_for < self._escalator_deferral_timeout_secs:
            return True

        # Let this cycle through so the LLM can break a potential escalator jam.
        logger.info(
            f"{agent_id}: escalator-context stall ({stalled_for:.1f}s) — allowing re-decision"
        )
        state["last_progress_time"] = current_sim_time
        return False

    def process_all_agents(
        self,
        observations: dict[str, str],
        current_sim_time: float,
        agent_ids: list[str] | None = None,
    ) -> float:
        """
        Process decision-making for all agents (parallel processing).

        Args:
            observations: Dict of agent_id -> observation string
            current_sim_time: Current simulation time in seconds
            agent_ids: Optional subset of agent IDs to process

        Returns:
            Current simulation time (for updating last_decision_time)
        """
        if agent_ids is None:
            logger.info(f"Agent decisions at t={current_sim_time:.1f}s")
        else:
            logger.info(
                f"Targeted agent decisions at t={current_sim_time:.1f}s for {len(agent_ids)} agents"
            )

        # Get available zones (same for all levels)
        zones = list(self.action_translator.zones_polygons.keys()) or list(
            self.action_translator.zones.keys()
        )

        # Run agent processing in parallel using asyncio
        asyncio.run(self._process_agents_parallel(observations, zones, current_sim_time, agent_ids))

        return current_sim_time

    async def _process_agents_parallel(
        self,
        observations: dict,
        zones: list,
        current_sim_time: float,
        agent_ids: list[str] | None = None,
    ):
        """Process all agents in parallel using async/await."""
        # Create event-loop-bound primitives here so they are always attached to
        # the loop that asyncio.run() just started (a new loop each decision cycle).
        self._state_lock = asyncio.Lock()
        self._llm_semaphore = asyncio.Semaphore(self._llm_semaphore_limit)

        # Filter and create tasks using list comprehension for efficiency
        candidate_agents = (
            agent_ids if agent_ids is not None else list(self.concordia_agents.keys())
        )
        agents_to_process = [
            agent_id
            for agent_id in candidate_agents
            if agent_id in self.concordia_agents
            and agent_id not in self.exited_agents
        ]

        # --- Per-cycle zone cache ---
        # Pre-compute zone_id for every active agent once (avoids N × M Shapely
        # contains-checks spread across N concurrent coroutines).
        # Use smallest-area match so specific sub-zones (e.g. "platform_1")
        # take priority over large background polygons (e.g. "level_-1").
        cycle_zone_cache: dict[str, str | None] = {}
        zones_polygons = getattr(self.action_translator, "zones_polygons", {})
        if zones_polygons:
            from shapely.geometry import Point as _Point

            for agent_id in agents_to_process:
                pos = self.state_queries.get_agent_position(agent_id)
                if pos is None:
                    cycle_zone_cache[agent_id] = None
                    continue
                pt = _Point(pos)
                found: str | None = None
                found_area: float = float("inf")
                for z_id, poly in zones_polygons.items():
                    try:
                        if poly.covers(pt) or poly.contains(pt):
                            area = poly.area
                            if area < found_area:
                                found_area = area
                                found = z_id
                    except Exception:
                        pass
                cycle_zone_cache[agent_id] = found

        tasks = [
            self._process_single_agent(
                agent_id,
                self.concordia_agents[agent_id],
                observations,
                zones,
                current_sim_time,
                cycle_zone_cache,
            )
            for agent_id in agents_to_process
        ]

        # --- Group-level LLM decisions (Opt 7c) ---
        # When enabled, agents sharing the same zone + destination + knowledge profile
        # are grouped.  Only the first agent ("representative") in each qualifying
        # group makes a real LLM call; the others get the same decision injected into
        # their prompt cache so they skip the LLM on this cycle.
        # This is an approximation: individual memory / injury state are ignored.
        # Enable via DecisionProcessor(enable_group_decisions=True).
        if self.enable_group_decisions and len(agents_to_process) >= self.group_decision_min_size:
            groups: dict[tuple, list[str]] = {}
            for agent_id in agents_to_process:
                zone = cycle_zone_cache.get(agent_id)
                dest = self.agent_destinations.get(agent_id, "")
                profile = self._agent_cfg.get(agent_id, {}).get("knowledge_profile", "novice")
                key = (zone, dest, profile)
                groups.setdefault(key, []).append(agent_id)

            # For each qualifying group, force-cache the representative's decision
            # into all follower agents' prompt caches *before* tasks run.
            # The representative's task will call LLM naturally; followers will hit
            # the cache (because should_call_llm will see their hash unchanged if
            # they haven't received messages/events).
            # We don't do anything here that blocks — the actual work happens in tasks.
            # What we mark here is that non-representative agents in large groups
            # should be skipped entirely at the top of _process_single_agent.
            self._group_followers: set[str] = set()
            for key, members in groups.items():
                if len(members) >= self.group_decision_min_size:
                    representative = members[0]
                    for follower in members[1:]:
                        # Mark follower to use representative's result.
                        self._group_followers.add(follower)
                        # Seed the follower's cache with representative's last decision
                        # so they short-circuit without needing a fresh LLM call.
                        cached = self.prompt_cache.get_cached_decision(representative)
                        if cached:
                            self.prompt_cache.cache_decision(follower, cached)
                    logger.debug(
                        f"Group decision: representative={representative}, "
                        f"followers={members[1:]} (zone={key[0]}, dest={key[1]})"
                    )
        else:
            self._group_followers = set()

        # Process all agents concurrently. Per-agent timeout is applied around
        # the LLM call itself in _process_single_agent, not while waiting for
        # semaphore queue slots.
        with self.perf_timer.measure("parallel_agent_processing"):
            await asyncio.gather(*tasks, return_exceptions=True)

        # No need to check for abandonment - it's handled naturally in action execution
        # Agents who choose move/wait over help automatically end relationships

    async def _process_single_agent(
        self,
        agent_id: str,
        agent,
        observations: dict,
        zones: list,
        current_sim_time: float,
        cycle_zone_cache: dict[str, str | None] | None = None,
    ):
        """Process a single agent's decision (async) with intelligent prompt caching."""
        try:
            # Group-decision follower short-circuit (Opt 7c).
            # If this agent has been designated a follower in the current cycle,
            # skip the full pipeline — the representative's decision is already
            # seeded into the prompt cache and the short-circuit in the
            # should_call_llm path will be triggered automatically.
            if getattr(self, "_group_followers", None) and agent_id in self._group_followers:
                async with self._state_lock:
                    self.llm_calls_skipped += 1
                logger.debug(f"{agent_id}: group follower — skipping this cycle")
                return

            # ── Position & zone (needed for goal status and exit filtering) ──
            position = self.state_queries.get_agent_position(agent_id)
            if position is None:
                logger.debug(f"{agent_id}: No position found, likely exited")
                return

            # While traversing escalator geometry, keep current movement and defer
            # new decisions until the next regular batch after leaving that area.
            if self._agent_is_on_escalator(agent_id):
                if self._should_defer_escalator_decision(agent_id, current_sim_time, position):
                    logger.debug(f"{agent_id}: in escalator departure context — deferring decision")
                    return
            else:
                self._escalator_deferral_state.pop(agent_id, None)

            # Use the pre-computed per-cycle zone cache when available to avoid
            # repeating O(n_zones) Shapely contains-checks per agent.
            if cycle_zone_cache is not None:
                zone_id = cycle_zone_cache.get(agent_id)
            else:
                zone_id = self._identify_zone(position)

            # ── Persistent goal ──────────────────────────────────────────────
            if agent_id not in self.agent_goals:
                cfg = self._agent_cfg.get(agent_id, {})
                self.agent_goals[agent_id] = cfg.get("initial_goal", "")
            agent_goal = self.agent_goals[agent_id]

            # Build goal-context prefix for the observation
            goal_lines: list[str] = []
            if agent_goal:
                goal_lines.append(f"Your current goal: {agent_goal}")
                if zone_id:
                    keywords = self._zone_goal_keywords.get(zone_id, [])
                    if any(kw.lower() in agent_goal.lower() for kw in keywords):
                        goal_lines.append(
                            "Goal status: You are currently at your goal destination."
                        )

            # Wait-duration nudge: inject a message that changes every minute an
            # agent has been continuously waiting.  The different text produces a
            # fresh prompt hash each minute, busting the cache and forcing the LLM
            # to reconsider rather than repeating the same wait decision forever.
            wait_start = self._agent_wait_since.get(agent_id)
            if self._wait_nudge_enabled and wait_start is not None:
                wait_secs = current_sim_time - wait_start
                if wait_secs >= 60:
                    wait_mins = int(wait_secs / 60)
                    goal_lines.append(
                        f"\u26a0\ufe0f You have been stationary for {wait_mins} minute(s) "
                        f"while the evacuation alarm is active. "
                        f"Reassess whether to continue waiting or move."
                    )

            # Get observation for this agent and prepend goal context
            observation = observations.get(agent_id, "")
            if goal_lines:
                observation = "\n".join(goal_lines) + "\n\n" + observation

            # Build profile-aware valid exit options block
            valid_exits_text = self._get_valid_exits_section(agent_id, observation, zone_id)

            # Build follower constraint block (injected when circular/following detected)
            following_constraint_text = self._get_following_constraint_text(observation)

            # Only expose zone-targeting when it is meaningful this turn.
            # In practice this is when exits are not currently available/visible,
            # so the agent needs an intermediate area-level move.
            include_zones = "Visible exits right now: none." in valid_exits_text
            zones_section = self._build_zones_section(zones, zone_id) if include_zones else ""
            has_zones = bool(zones_section.strip())

            # Build the action spec with prompt text
            # Per-role extra text injected verbatim before the JSON schema.
            # Configured via agents.roles.<role>.decision_prompt_extra in the scenario YAML.
            cfg = self._agent_cfg.get(agent_id, {})
            role_prompt_extra = cfg.get("decision_prompt_extra", "")
            situation_framing = (f"{role_prompt_extra}\n\n") if role_prompt_extra else ""

            age = str(cfg.get("age", "unknown"))
            gender = str(cfg.get("gender", "person"))
            personality_profile = str(
                cfg.get("personality_anchor")
                or cfg.get("personality_type", "unknown")
            )
            last_decision_summary = self._build_last_decision_summary(agent_id, current_sim_time)

            # Compute minimal inline suffixes for zone-related placeholders.
            # zone_option:    appends ', "zone"' to the target_type line when zones are available.
            # zone_name_hint: appends the zone_name description when zones are available.
            # zone_constraint: adds the zone constraint bullet when zones are available.
            zone_option = ', "zone"' if has_zones else ""
            zone_name_hint = (
                " (or zone name EXACTLY as shown in Available Zones)" if has_zones else ""
            )
            zone_constraint = "- If target_type='zone', zone_name is required.\n" if has_zones else ""

            prompt_text = self._render_decision_prompt(
                age=age,
                gender=gender,
                personality_profile=personality_profile,
                last_decision_summary=last_decision_summary,
                situation_framing=situation_framing,
                zone_option=zone_option,
                zone_name_hint=zone_name_hint,
                zone_constraint=zone_constraint,
                zones_section=zones_section,
                following_constraint_text=following_constraint_text,
                valid_exits_text=valid_exits_text,
            )

            action_spec = entity_lib.ActionSpec(
                call_to_action=prompt_text,
                output_type=entity_lib.OutputType.FREE,
            )

            # ===== INTELLIGENT PROMPT CACHING =====
            # Check if we should call LLM or reuse cached decision
            try:
                received_messages = self.message_system.get_received_messages(agent_id)
                messages_list = (
                    [
                        msg.get("message", str(msg)) if isinstance(msg, dict) else msg
                        for msg in received_messages
                    ]
                    if received_messages
                    else None
                )
            except Exception as e:
                logger.debug(f"{agent_id}: Could not get messages: {e}")
                messages_list = None

            # Check cache to decide whether to call LLM
            should_call_llm, cached_decision = self.prompt_cache.should_call_llm(
                agent_id=agent_id,
                observation=observation,
                action_spec_text=prompt_text,
                received_messages=messages_list,
            )

            llm_was_called = False  # Track whether LLM was called for decision record

            if not should_call_llm and cached_decision:
                # Reuse cached decision - no new information to act on
                action = cached_decision
                logger.info(
                    f"{agent_id}: ✓ Prompt unchanged, reusing cached decision (saved LLM call)"
                )
                async with self._state_lock:
                    self.llm_calls_skipped += 1

                # --- Short-circuit Opt 7b ---
                # If the agent already has a committed destination and is moving toward
                # it, there is no need to re-translate or re-execute — JuPedSim is
                # already routing them correctly.  We only run the full pipeline when
                # the agent is waiting (needs a fresh waypoint) or has no destination yet.
                existing_dest = self.agent_destinations.get(agent_id)
                agent_is_moving = (
                    self.action_executor.agent_action.get(agent_id) == "moving"
                    if hasattr(self.action_executor, "agent_action")
                    else False
                )
                if existing_dest and agent_is_moving:
                    logger.debug(
                        f"{agent_id}: ✓ Short-circuit — already moving to '{existing_dest}', "
                        f"skipping translation & execution"
                    )
                    return
            else:
                if should_call_llm and self._min_redecision_interval_secs > 0:
                    if "No significant new information." in observation:
                        previous_llm_time = self._last_llm_decision_time.get(agent_id)
                        if previous_llm_time is not None:
                            elapsed = current_sim_time - previous_llm_time
                            if elapsed < self._min_redecision_interval_secs:
                                throttled_cached = self.prompt_cache.get_cached_decision(agent_id)
                                if throttled_cached:
                                    action = throttled_cached
                                    logger.info(
                                        f"{agent_id}: ✓ Reusing cached decision "
                                        f"(redecision throttle {elapsed:.1f}s "
                                        f"< {self._min_redecision_interval_secs:.1f}s)"
                                    )
                                    async with self._state_lock:
                                        self.llm_calls_skipped += 1
                                    should_call_llm = False
                                    cached_decision = throttled_cached

                if not should_call_llm and cached_decision:
                    action = cached_decision
                else:
                # Call LLM - significant new information detected
                    with self.perf_timer.measure("agent_observe", is_parallel=True):
                        agent.observe(observation)

                    # Acquire semaphore before the LLM call to prevent burst overload.
                    async with self._llm_semaphore:
                        try:
                            with self.perf_timer.measure("agent_act_llm", is_parallel=True):
                                llm_current_agent_id.set(agent_id)
                                llm_current_sim_time.set(current_sim_time)
                                if self._per_agent_timeout_secs is not None:
                                    async with asyncio.timeout(self._per_agent_timeout_secs):
                                        action = await asyncio.to_thread(agent.act, action_spec)
                                else:
                                    action = await asyncio.to_thread(agent.act, action_spec)
                        except asyncio.TimeoutError:
                            timeout_secs = self._per_agent_timeout_secs
                            logger.warning(
                                f"{agent_id}: decision timed out after {timeout_secs:.0f}s — "
                                "keeping existing waypoint"
                            )
                            return

                    # Cache the decision for future reuse
                    self.prompt_cache.cache_decision(agent_id, action)
                    llm_was_called = True
                    self._last_llm_decision_time[agent_id] = current_sim_time

                    if should_call_llm and messages_list:
                        logger.info(
                            f"{agent_id}: ✓ Calling LLM (received {len(messages_list)} message(s))"
                        )
                    elif should_call_llm:
                        logger.info(f"{agent_id}: ✓ Calling LLM (observation changed significantly)")
                    else:
                        logger.info(f"{agent_id}: ✓ Calling LLM (first decision)")

                    async with self._state_lock:
                        self.llm_calls_made += 1

                    # Update observation/action cache
                    async with self._state_lock:
                        self.last_observations[agent_id] = observation
                        self.last_actions[agent_id] = action

            # Parse JSON response
            with self.perf_timer.measure("parse_json_response", is_parallel=True):
                reasoning = self._parse_json_response(action)

            # Persist goal update if the agent revised their objective
            if isinstance(reasoning, dict):
                goal_update = reasoning.get("goal_update", "")
                if goal_update and isinstance(goal_update, str) and goal_update.strip():
                    async with self._state_lock:
                        self.agent_goals[agent_id] = goal_update.strip()
                    logger.info(f"{agent_id}: Updated goal → '{goal_update.strip()}'")

            # Translate action to JuPedSim command
            if position is None:
                # Agent has likely exited - skip action execution
                logger.debug(f"{agent_id}: No position found, likely exited")
                return

            with self.perf_timer.measure("translate_action", is_parallel=True):
                translated = self.action_translator.translate(agent_id, action, position)

            # Update wait-since tracker: reset on any movement, extend on wait.
            if translated.get("action_type") == "wait":
                async with self._state_lock:
                    self._agent_wait_since.setdefault(agent_id, current_sim_time)
            else:
                async with self._state_lock:
                    self._agent_wait_since.pop(agent_id, None)

            # Inject the LLM's own reasoning text into the translated dict so that
            # the previous-decision memory shown to the agent next turn uses the
            # agent's actual words rather than the translator's description.
            llm_reasoning = self._reasoning_to_str(
                reasoning.get("reasoning", "") if isinstance(reasoning, dict) else ""
            )
            if llm_reasoning:
                translated["reasoning"] = llm_reasoning

            # Extract and deliver any message
            with self.perf_timer.measure("message_delivery", is_parallel=True):
                self.message_system.extract_and_deliver_message(
                    sender_id=agent_id,
                    action=action,
                    sender_position=position,
                    current_sim_time=current_sim_time,
                    state_queries=self.state_queries,
                    exited_agents=self.exited_agents,
                )

            # Detect route changes and store decision
            with self.perf_timer.measure("decision_storage", is_parallel=True):
                new_exit = extract_exit_name(translated, self.station_layout)

                # Async-safe read from agent_destinations
                async with self._state_lock:
                    old_exit = self.agent_destinations.get(agent_id)

                route_changed = False

                # Prepare decision record before acquiring lock
                decision_record = {
                    "time": current_sim_time,
                    "observation": observation,
                    "prompt": action_spec.call_to_action if llm_was_called else "cached",
                    "action": action,
                    "reasoning": reasoning,
                    "translated": translated,
                }

                # Async-safe update of shared state
                async with self._state_lock:
                    if new_exit:
                        if old_exit and old_exit != new_exit:
                            # Route change detected!
                            logger.info(f"🔄 {agent_id} changed route: {old_exit} → {new_exit}")
                            route_changed = True

                    # Add route change metadata if it occurred
                    if route_changed:
                        decision_record["route_change"] = {
                            "from_exit": old_exit,
                            "to_exit": new_exit,
                            "reason": self._reasoning_to_str(reasoning.get("reasoning", "")),
                        }

                    # Store decision — cap history to avoid unbounded RAM growth.
                    # Route-change records are always kept; others are a rolling window.
                    _MAX_DECISIONS_PER_AGENT = 200
                    if agent_id not in self.agent_decisions:
                        self.agent_decisions[agent_id] = {"decisions": []}

                    decisions_list = self.agent_decisions[agent_id]["decisions"]
                    decisions_list.append(decision_record)
                    if len(decisions_list) > _MAX_DECISIONS_PER_AGENT:
                        # Evict oldest entry; this keeps the list O(1) amortised.
                        decisions_list.pop(0)

            # Apply to JuPedSim
            with self.perf_timer.measure("apply_to_jupedsim", is_parallel=True):
                self.action_executor.execute_action(agent_id, translated, current_sim_time)

            logger.info(f"{agent_id} action: {action[:100]}...")

        except Exception as e:
            logger.error(f"Error processing {agent_id}: {e}", exc_info=True)

    def _parse_json_response(self, response: str) -> dict[str, str]:
        """Parse JSON response from agent, extracting reasoning components."""
        try:
            # Strip agent name prefix (e.g., "Agent 0 {" -> "{")
            json_start = response.find("{")
            if json_start > 0:
                response = response[json_start:]

            data = json.loads(response)
            return {
                "reasoning": data.get("reasoning", ""),
                "goal_update": data.get("goal_update", ""),
            }
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse JSON response: {response[:200]}")
            return {
                "reasoning": response[:200],
            }

    @staticmethod
    def _reasoning_to_str(reasoning_val) -> str:
        """Convert a reasoning value (PADM dict or plain string) to a readable string.

        For PADM responses, combines risk_identification and protective_action_assessment
        as the most decision-relevant fields.  Plain strings are returned unchanged.
        """
        if isinstance(reasoning_val, str):
            return reasoning_val
        if isinstance(reasoning_val, dict):
            parts = []
            for key in ("risk_identification", "protective_action_assessment"):
                val = reasoning_val.get(key, "")
                if val and isinstance(val, str):
                    parts.append(val.strip().rstrip("."))
            return ". ".join(parts) if parts else ""
        return str(reasoning_val) if reasoning_val else ""

    @staticmethod
    def _summarize_text(text: str, max_len: int = 220) -> str:
        """Return a compact single-line summary suitable for header context."""
        compact = " ".join(str(text).split())
        if len(compact) <= max_len:
            return compact
        return compact[: max_len - 3].rstrip() + "..."

    def _build_last_decision_summary(self, agent_id: str, current_sim_time: float) -> str:
        """Build a readable summary of the agent's previous decision."""
        decisions = self.agent_decisions.get(agent_id, {}).get("decisions", [])
        if not decisions:
            return "no previous decision recorded."

        last = decisions[-1]
        last_time = float(last.get("time", current_sim_time))
        seconds_ago = max(0.0, current_sim_time - last_time)

        translated = last.get("translated", {}) if isinstance(last, dict) else {}
        action_type = translated.get("action_type", "unknown")
        target_type = translated.get("target_type", "")
        target_exit = translated.get("target_exit", "")
        target_agent = translated.get("target_agent", "")
        zone_name = translated.get("zone_name", "")

        action_desc = action_type
        if action_type == "move":
            if target_type == "exit" and target_exit:
                action_desc = f"move to exit '{target_exit}'"
            elif target_type == "agent" and target_agent:
                action_desc = f"move toward {target_agent}"
            elif target_type == "zone" and zone_name:
                action_desc = f"move to zone '{zone_name}'"
            else:
                action_desc = "move"
        elif action_type == "wait":
            wait_reason = translated.get("wait_reason", "")
            action_desc = f"wait ({wait_reason})" if wait_reason else "wait"

        reasoning_payload = last.get("reasoning", "") if isinstance(last, dict) else ""
        if isinstance(reasoning_payload, dict):
            reason_text = self._reasoning_to_str(reasoning_payload.get("reasoning", ""))
        else:
            reason_text = self._reasoning_to_str(reasoning_payload)

        if not reason_text:
            reason_text = "no explicit reason was recorded"

        return f"{seconds_ago:.0f}s ago you chose to {action_desc} because {reason_text}."

    def on_agent_exit(self, agent_id: str) -> None:
        """
        Clean up cache when agent exits simulation.

        Args:
            agent_id: ID of exiting agent
        """
        self.prompt_cache.clear_agent(agent_id)
        self._agent_wait_since.pop(agent_id, None)
        logger.debug(f"{agent_id}: Cache cleared on exit")

    def get_cache_statistics(self) -> dict[str, Any]:
        """
        Get statistics about LLM call optimization.

        Returns:
            Dictionary with cache and call statistics
        """
        total_calls = self.llm_calls_made + self.llm_calls_skipped
        skip_rate = (self.llm_calls_skipped / total_calls * 100) if total_calls > 0 else 0

        return {
            "llm_calls_made": self.llm_calls_made,
            "llm_calls_skipped": self.llm_calls_skipped,
            "total_decision_cycles": total_calls,
            "skip_rate_percent": skip_rate,
            "cache_stats": self.prompt_cache.get_statistics(),
        }

    def log_cache_summary(self) -> None:
        """Log summary of cache statistics to console and logger."""
        stats = self.get_cache_statistics()
        logger.info("=" * 70)
        logger.info("LLM CALL OPTIMIZATION SUMMARY (Prompt Caching)")
        logger.info("=" * 70)
        logger.info(f"  Total decision cycles: {stats['total_decision_cycles']}")
        logger.info(f"  LLM calls made:        {stats['llm_calls_made']}")
        logger.info(f"  LLM calls skipped:     {stats['llm_calls_skipped']} ✓")
        logger.info(f"  Skip rate:             {stats['skip_rate_percent']:.1f}%")
        logger.info(f"  Cache size:            {stats['cache_stats']['cached_agents']} agents")
        logger.info("=" * 70)
