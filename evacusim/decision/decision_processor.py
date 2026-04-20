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
        self._llm_semaphore_limit: int = 20

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

        logger.debug("DecisionProcessor initialized for parallel async processing")

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
        """
        registry = self.action_translator.exit_registry
        valid_ids = set(registry.get_all_ids())
        cfg = self._agent_cfg.get(agent_id, {})
        profile = cfg.get("knowledge_profile", "novice")

        # Exits that arrive INTO this zone are one-way arrivals, not departure choices.
        arrival_exits = set(self._arrival_exits_by_zone.get(zone_id or "", []))

        def _is_valid_departure(eid: str) -> bool:
            return eid not in arrival_exits

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

        def _sort_display_exits(display_names: list[str]) -> list[str]:
            indexed = list(enumerate(display_names))

            def key_fn(item: tuple[int, str]) -> tuple[int, int, int, str]:
                idx, name = item
                is_visible = 0 if name in visible_distance_rank else 1
                dist_rank = visible_distance_rank.get(name, 99)
                return (is_visible, dist_rank, idx, name.lower())

            return [name for _, name in sorted(indexed, key=key_fn)]

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
                return "\nAllowed exits now:" f"{bullets}\n\n"
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
            return "\nVisible exits right now:" f"{bullets}\n\n"

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
            return "\nAllowed exits now:" f"{bullets}\n\n"

        return (
            "\nVisible exits right now: none.\n"
            "Do NOT use target_type='exit'. Follow someone or wait.\n\n"
        )

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
        """Return True if the agent is currently inside an escalator corridor polygon.

        Agents on escalators have no meaningful choices to make — direction and
        minimum speed are enforced physically.  Skipping decision-making prevents
        the LLM from issuing a 'wait' action mid-escalator, which would override
        the agent's journey and leave them stationary.
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
        corridors = getattr(getattr(sim, "geometry_manager", None), "escalator_corridors", {})
        if not corridors:
            return False
        p = Point(pos)
        return any(poly.contains(p) for poly in corridors.values())

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
            and not self._agent_is_on_escalator(agent_id)
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

        # Process all agents concurrently
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
            if wait_start is not None:
                wait_secs = current_sim_time - wait_start
                if wait_secs >= 60:
                    wait_mins = int(wait_secs / 60)
                    goal_lines.append(
                        f"\u26a0\ufe0f You have been stationary for {wait_mins} minute(s) "
                        f"while the evacuation alarm is active. "
                        f"Staying put any longer is dangerous \u2014 you must decide to move now."
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

            # The 'zone' target_type is only valid when AVAILABLE ZONES is non-empty.
            zone_target_type_line = (
                '  "target_type": ONE OF: "current_position", "exit", "agent", "zone",\n'
                if has_zones
                else '  "target_type": ONE OF: "current_position", "exit", "agent",\n'
            )
            zone_name_example = (
                '  "zone_name": null (or zone name EXACTLY as shown in AVAILABLE ZONES),\n'
                if has_zones
                else '  "zone_name": null,\n'
            )
            zone_constraint_line = (
                "- If target_type='zone', zone_name is required.\n" if has_zones else ""
            )

            # Build the action spec with prompt text
            # Per-role extra text injected verbatim before the JSON schema.
            # Configured via agents.roles.<role>.decision_prompt_extra in the scenario YAML.
            cfg = self._agent_cfg.get(agent_id, {})
            role_prompt_extra = cfg.get("decision_prompt_extra", "")
            role_prefix = (f"{role_prompt_extra}\n\n") if role_prompt_extra else ""

            action_spec = entity_lib.ActionSpec(
                call_to_action=(
                    f"{role_prefix}"
                    "Decision task: choose your next action now. Respond with ONLY this JSON:\n"
                    "{\n"
                    '  "reasoning": "1-2 sentences grounded in goal + current observations",\n'
                    '  "action_type": "wait" or "move",\n'
                    f"{zone_target_type_line}"
                    '  "target_agent": null (or agent_id like "agent_5"),\n'
                    '  "exit_name": null (or the exit name EXACTLY as listed in visible/valid exits),\n'
                    f"{zone_name_example}"
                    '  "wait_reason": null (or reason if waiting),\n'
                    '  "speed": null (or "slow_walk", "normal_walk", "brisk_walk", "jog", "run"),\n'
                    '  "message": null (or your spoken words),\n'
                    '  "message_type": null (or "directed", "shout"),\n'
                    '  "goal_update": null (or revised plain-text goal if your fundamental objective has changed),\n'
                    "}\n\n"
                    "Hard constraints:\n"
                    "- If action_type='wait', target_type MUST be 'current_position'.\n"
                    "- If target_type='exit', exit_name is required and must match an allowed exit name exactly.\n"
                    "- If target_type='agent', target_agent is required.\n"
                    f"{zone_constraint_line}"
                    "- Keep spoken message short and natural; no narration.\n"
                    "- goal_update: null unless your fundamental objective has genuinely changed.\n\n"
                    "Decision policy:\n"
                    "1. If helping someone, stay coordinated with that person.\n"
                    "2. To progress toward your goal, choose target_type='exit' or target_type='agent'.\n"
                    "3. If uncertain or waiting, choose target_type='current_position'.\n\n"
                    f"{zones_section}"
                    f"{following_constraint_text}"
                    f"{valid_exits_text}"
                ),
                output_type=entity_lib.OutputType.FREE,
            )

            prompt_text = action_spec.call_to_action

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
                # Call LLM - significant new information detected
                with self.perf_timer.measure("agent_observe", is_parallel=True):
                    agent.observe(observation)

                # Acquire semaphore before the LLM call to prevent burst overload.
                # At most 20 concurrent requests are sent to Azure at any one time.
                async with self._llm_semaphore:
                    with self.perf_timer.measure("agent_act_llm", is_parallel=True):
                        llm_current_agent_id.set(agent_id)
                        llm_current_sim_time.set(current_sim_time)
                        action = await asyncio.to_thread(agent.act, action_spec)

                # Cache the decision for future reuse
                self.prompt_cache.cache_decision(agent_id, action)
                llm_was_called = True

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
            llm_reasoning = reasoning.get("reasoning", "") if isinstance(reasoning, dict) else ""
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

                        # Update destination tracking
                        self.agent_destinations[agent_id] = new_exit

                    # Add route change metadata if it occurred
                    if route_changed:
                        decision_record["route_change"] = {
                            "from_exit": old_exit,
                            "to_exit": new_exit,
                            "reason": reasoning.get("reasoning", ""),
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
                "situation": data.get("situation", ""),
                "risk_level": data.get("risk_level", ""),
                "risk_assessment": data.get("risk_assessment", ""),
                "social_context": data.get("social_context", ""),
                "reasoning": data.get("reasoning", ""),
                "goal_update": data.get("goal_update", ""),
            }
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse JSON response: {response[:200]}")
            return {
                "situation": "Parse error",
                "reasoning": response[:200],
            }

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
