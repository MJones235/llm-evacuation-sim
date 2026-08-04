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
import hashlib
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
        event_manager=None,
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
        self.event_manager = event_manager
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

        # Per-agent persistent goal text used for journey framing.
        # Seeded from agent_cfg["initial_goal"] on first decision.
        self.agent_goals: dict[str, str] = {}

        # Tracks the sim-time when each agent last issued a non-wait action.
        # Used to inject an escalating wait-duration nudge into the observation so
        # that the prompt hash changes every minute, busting the cache and forcing
        # the LLM to re-evaluate agents stuck in a repetitive-wait deadlock.
        self._agent_wait_since: dict[str, float] = {}
        self._agent_reassess_modes: dict[str, str] = {}
        self._last_zone_by_agent: dict[str, str | None] = {}
        self._last_route_blocked_by_agent: dict[str, bool] = {}
        self._last_alarm_signature_by_agent: dict[str, str] = {}
        self._last_train_hint_by_agent: dict[str, str] = {}

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
        journey_block: str,
        new_since_last_decision: str,
        current_surroundings: str,
        available_actions_block: str,
        wait_action_rule_block: str,
        pace_field_block: str,
        pace_validation_block: str,
        situation_framing: str,
        following_constraint_text: str,
        valid_exits_text: str,
    ) -> str:
        """Render the decision prompt text from the configured template."""
        return self._decision_prompt_template.safe_substitute(
            age=age,
            gender=gender,
            personality_profile=personality_profile,
            journey_block=journey_block,
            new_since_last_decision=new_since_last_decision,
            current_surroundings=current_surroundings,
            available_actions_block=available_actions_block,
            wait_action_rule_block=wait_action_rule_block,
            pace_field_block=pace_field_block,
            pace_validation_block=pace_validation_block,
            situation_framing=situation_framing,
            following_constraint_text=following_constraint_text,
            valid_exits_text=valid_exits_text,
        )

    @staticmethod
    def _build_pace_prompt_blocks(offered_actions: list[str]) -> tuple[str, str, str]:
        """Build conditional wait/pace schema and validation text."""
        offered = set(offered_actions)
        if offered == {"wait"}:
            return "- if action = wait: set wait_reason.", "", ""

        wait_action_rule_block = "- if action = wait: set wait_reason and set pace to null."
        pace_field_block = '  "pace": null,\n'
        pace_validation_block = (
            "- pace definitions: normal_pace - moving as you normally would; "
            "hurrying - walking noticeably faster than usual, but not running; "
            "running - running.\n"
            '- if action = "wait": pace must be null.\n'
            '- if action is not "wait": pace must be one of "normal_pace", "hurrying", or "running".'
        )
        return wait_action_rule_block, pace_field_block, pace_validation_block

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

    def _get_valid_exits_section(
        self,
        agent_id: str,
        observation: str,
        zone_id: str | None,
    ) -> tuple[str, list[str], bool]:
        """Build exit guidance text and return offered evacuation exit IDs.

        Returns:
            tuple(valid_exits_text, offered_exit_ids, has_visible_or_known_exit)
        """
        registry = self.action_translator.exit_registry
        valid_ids = set(registry.get_all_ids())
        cfg = self._agent_cfg.get(agent_id, {})
        profile = cfg.get("knowledge_profile", "novice")

        goal_text = self.agent_goals.get(agent_id, "")
        goal_policy = self._get_goal_semantic_policy(zone_id, goal_text)

        arrival_exits = set(self._arrival_exits_by_zone.get(zone_id or "", []))

        def _is_valid_departure(eid: str) -> bool:
            return eid not in arrival_exits

        blocked_display: set[str] = set()
        for _m in _RE_BLOCKED_EXIT_LINE.finditer(observation):
            blocked_display.add(_m.group(1).strip())

        visible_names: set[str] = set()
        line_match = _RE_VISIBLE_EXITS_LINE.search(observation)
        if line_match:
            for m in _RE_EXITS_LINE_ENTRY.finditer(line_match.group(1)):
                visible_names.add(m.group(1).strip())

        # Candidate exit IDs by profile and visibility.
        candidate_ids: list[str] = []

        if profile == "commuter":
            mem_ids = self._zone_known_exits.get(zone_id or "", {}).get("commuter", [])
            for eid in mem_ids:
                if eid in valid_ids and _is_valid_departure(eid):
                    disp = registry.get_display_name(eid)
                    if disp not in blocked_display:
                        candidate_ids.append(eid)

        # Visible exits are available to all profiles.
        for eid in sorted(valid_ids):
            if not _is_valid_departure(eid):
                continue
            disp = registry.get_display_name(eid)
            if disp in blocked_display:
                continue
            if disp in visible_names:
                candidate_ids.append(eid)

        # Profile fallback when no visible exits were extracted.
        if not candidate_ids:
            fallback_ids = self._zone_known_exits.get(zone_id or "", {}).get(profile, [])
            for eid in fallback_ids:
                if eid in valid_ids and _is_valid_departure(eid):
                    disp = registry.get_display_name(eid)
                    if disp not in blocked_display:
                        candidate_ids.append(eid)

        # De-duplicate while preserving order.
        seen: set[str] = set()
        offered_exit_ids: list[str] = []
        for eid in candidate_ids:
            if eid in seen:
                continue
            seen.add(eid)
            offered_exit_ids.append(eid)

        semantics_text = self._build_exit_semantics_text(goal_policy, offered_exit_ids)
        if not offered_exit_ids:
            return (
                semantics_text + "No evacuation exits are currently available from where you are.\n",
                [],
                False,
            )

        bullets = []
        for eid in offered_exit_ids:
            disp = registry.get_display_name(eid)
            bullets.append(f"- {eid}: {disp}")

        return (
            "Evacuation exits you can choose now (use exit_id exactly):\n"
            + "\n".join(bullets)
            + "\n"
            + semantics_text,
            offered_exit_ids,
            True,
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
            return ""
        return "\n" + "\n".join(lines) + "\n"

    @staticmethod
    def _has_reachable_information_source(observation: str) -> bool:
        """Heuristic reachability check for staff/board information sources."""
        text = observation.lower()
        return any(
            key in text
            for key in (
                "staff",
                "rci",
                "responder",
                "indicator board",
                "board state",
                "station pa",
            )
        )

    def _has_train_service(self) -> bool:
        """Return True if the scenario exposes train exits in the registry."""
        try:
            for exit_id in self.action_translator.exit_registry.get_all_ids():
                if str(exit_id).startswith("train_platform_"):
                    return True
        except Exception:
            return False
        return False

    def _is_route_blocked(self, agent_id: str, observation: str) -> bool:
        """Return True if the agent's current routing target is flagged as blocked."""
        current_dest = self.agent_destinations.get(agent_id)
        if not current_dest:
            return False

        registry = getattr(self.action_translator, "exit_registry", None)
        current_display = (
            registry.get_display_name(current_dest) if registry is not None else current_dest
        )
        blocked_mentions = [_m.group(1).strip() for _m in _RE_BLOCKED_EXIT_LINE.finditer(observation)]
        return any(name == current_display for name in blocked_mentions)

    def _build_available_actions_block(
        self,
        offered_actions: list[str],
        offered_wait_reasons: list[str],
        offered_exit_ids: list[str],
    ) -> str:
        """Render offered-set actions with required fields."""
        lines: list[str] = ["Choose exactly one action from this offered set:"]
        for verb in offered_actions:
            if verb == "wait":
                reasons = ", ".join(offered_wait_reasons)
                lines.append(f"- wait (required: wait_reason in {{{reasons}}})")
            elif verb == "evacuate":
                exits = ", ".join(offered_exit_ids) if offered_exit_ids else "none"
                lines.append(f"- evacuate (required: exit_id from {{{exits}}})")
            else:
                lines.append(f"- {verb} (no extra field)")
        return "\n".join(lines)

    @staticmethod
    def _extract_new_information_text(observation: str) -> str:
        """Extract the cue-derived new-information line from an observation."""
        lines = [ln.strip() for ln in observation.splitlines() if ln.strip()]
        for idx, line in enumerate(lines):
            if line.startswith("What is NEW since your last decision:"):
                if idx + 1 < len(lines):
                    return lines[idx + 1]
                return "No significant new information."
        return "No significant new information."

    def _detect_cues(
        self,
        agent_id: str,
        observation: str,
        zone_id: str | None,
        route_blocked: bool,
    ) -> list[str]:
        """Detect cue types that should wake gated agents."""
        cues: list[str] = []
        obs_lower = observation.lower()

        last_zone = self._last_zone_by_agent.get(agent_id)
        if last_zone is None:
            self._last_zone_by_agent[agent_id] = zone_id
        elif zone_id is not None and zone_id != last_zone:
            cues.append("zone_entry")
            self._last_zone_by_agent[agent_id] = zone_id

        prev_blocked = self._last_route_blocked_by_agent.get(agent_id)
        if prev_blocked is None:
            self._last_route_blocked_by_agent[agent_id] = route_blocked
        elif prev_blocked != route_blocked:
            cues.append("route_blocked" if route_blocked else "route_unblocked")
            self._last_route_blocked_by_agent[agent_id] = route_blocked

        if "(directing you) said:" in obs_lower:
            if "pa" in obs_lower or "pa system" in obs_lower:
                cues.append("pa_message")
            else:
                cues.append("staff_instruction")

        train_hint = ""
        if "train" in obs_lower and ("arriv" in obs_lower or "waiting" in obs_lower):
            train_hint = "train_arrival"
        prev_train_hint = self._last_train_hint_by_agent.get(agent_id, "")
        if train_hint and train_hint != prev_train_hint:
            cues.append("train_arrival")
        self._last_train_hint_by_agent[agent_id] = train_hint

        alarm_signature = ""
        if "alarm" in obs_lower:
            if "all clear" in obs_lower or "all-clear" in obs_lower:
                alarm_signature = "all_clear"
            elif "sounding" in obs_lower or "activated" in obs_lower or "onset" in obs_lower:
                alarm_signature = "alarm_onset"
            else:
                alarm_signature = "alarm_state"
        prev_alarm = self._last_alarm_signature_by_agent.get(agent_id, "")
        if alarm_signature and alarm_signature != prev_alarm:
            cues.append("alarm_state_change")
        if alarm_signature:
            self._last_alarm_signature_by_agent[agent_id] = alarm_signature

        return cues

    @staticmethod
    def _extract_json_object(response: str) -> dict[str, Any] | None:
        """Extract and decode the first JSON object in a model response."""
        if not response:
            return None
        json_start = response.find("{")
        if json_start < 0:
            return None
        try:
            return json.loads(response[json_start:])
        except json.JSONDecodeError:
            return None

    def _validate_decision_payload(
        self,
        payload: dict[str, Any],
        offered_actions: set[str],
        offered_wait_reasons: set[str],
        offered_exit_ids: set[str],
    ) -> list[str]:
        """Return schema validation errors for a decision payload."""
        errors: list[str] = []
        action = payload.get("action")
        wait_reason = payload.get("wait_reason")
        exit_id = payload.get("exit_id")
        pace = payload.get("pace")
        reassess_when = payload.get("reassess_when")
        assessment = payload.get("assessment")

        if action not in offered_actions:
            errors.append(f"action must be one of {sorted(offered_actions)}")

        if not isinstance(assessment, dict):
            errors.append("assessment must be an object")
        else:
            required_assessment_fields = (
                "source_credibility",
                "situation_appraisal",
                "personal_relevance",
                "options_considered",
                "option_chosen_because",
                "information_gap",
            )
            for key in required_assessment_fields:
                if key not in assessment or not isinstance(assessment.get(key), str):
                    errors.append(f"assessment.{key} must be a string")

        if action == "wait":
            if wait_reason not in offered_wait_reasons:
                errors.append(
                    f"wait_reason must be one of {sorted(offered_wait_reasons)} when action=wait"
                )
            if pace is not None:
                errors.append("pace must be null when action=wait")
        else:
            if wait_reason is not None:
                errors.append("wait_reason must be null unless action=wait")
            if pace not in {"normal_pace", "hurrying", "running"}:
                errors.append(
                    'pace must be one of "normal_pace", "hurrying", or "running" when action is not wait'
                )

        if action == "evacuate":
            if exit_id not in offered_exit_ids:
                errors.append(f"exit_id must be one of {sorted(offered_exit_ids)} when action=evacuate")
        else:
            if exit_id is not None:
                errors.append("exit_id must be null unless action=evacuate")

        if reassess_when not in {"next_interval", "new_cue_only"}:
            errors.append("reassess_when must be 'next_interval' or 'new_cue_only'")

        return errors

    @staticmethod
    def _decision_payload_to_json(payload: dict[str, Any]) -> str:
        """Serialize payload into a stable compact JSON string."""
        return json.dumps(payload, ensure_ascii=True, separators=(",", ":"))

    def _build_fallback_decision(
        self,
        agent_id: str,
        offered_actions: set[str],
        offered_wait_reasons: set[str],
        offered_exit_ids: set[str],
    ) -> dict[str, Any]:
        """Fallback policy after repeated malformed outputs."""
        decisions = self.agent_decisions.get(agent_id, {}).get("decisions", [])
        if decisions:
            last_payload = decisions[-1].get("decision_payload")
            if isinstance(last_payload, dict):
                errors = self._validate_decision_payload(
                    last_payload,
                    offered_actions,
                    offered_wait_reasons,
                    offered_exit_ids,
                )
                if not errors:
                    return last_payload

        fallback_action = "continue_activity" if "continue_activity" in offered_actions else "wait"
        fallback_wait_reason = None
        fallback_exit = None
        fallback_pace: str | None = "normal_pace"
        if fallback_action == "wait":
            fallback_wait_reason = (
                "awaiting_information"
                if "awaiting_information" in offered_wait_reasons
                else sorted(offered_wait_reasons)[0]
            )
            fallback_pace = None

        return {
            "assessment": {
                "source_credibility": "Unable to parse a valid response this turn.",
                "situation_appraisal": "No updated appraisal was available.",
                "personal_relevance": "No updated relevance estimate was available.",
                "options_considered": "Fallback policy applied.",
                "option_chosen_because": "Reused a safe default after repeated schema failures.",
                "information_gap": "No reliable model output was available.",
            },
            "action": fallback_action,
            "wait_reason": fallback_wait_reason,
            "exit_id": fallback_exit,
            "pace": fallback_pace,
            "reassess_when": "next_interval",
        }

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
            valid_exits_text, offered_exit_ids, has_reachable_exit = self._get_valid_exits_section(
                agent_id,
                observation,
                zone_id,
            )

            route_blocked = self._is_route_blocked(agent_id, observation)
            cues = self._detect_cues(agent_id, observation, zone_id, route_blocked)

            # Cue-only gate: skip base-interval deliberation until a cue arrives.
            reassess_mode = self._agent_reassess_modes.get(agent_id, "next_interval")
            if reassess_mode == "new_cue_only" and not cues:
                logger.debug(f"{agent_id}: gated by reassess_when='new_cue_only' (no cue)")
                return

            offered_wait_reasons = ["awaiting_information", "awaiting_instruction"]
            if route_blocked:
                offered_wait_reasons.append("route_blocked")

            offered_actions = ["continue_activity", "wait"]
            if self.action_executor.has_reachable_information_source(agent_id, position, zone_id):
                offered_actions.append("seek_information")
            if has_reachable_exit and offered_exit_ids:
                offered_actions.append("evacuate")
            if self._has_train_service():
                offered_actions.append("leave_by_train")

            offered_actions_set = set(offered_actions)
            offered_wait_reasons_set = set(offered_wait_reasons)
            offered_exit_ids_set = set(offered_exit_ids)
            available_actions_block = self._build_available_actions_block(
                offered_actions,
                offered_wait_reasons,
                offered_exit_ids,
            )
            wait_action_rule_block, pace_field_block, pace_validation_block = (
                self._build_pace_prompt_blocks(offered_actions)
            )

            # Build follower constraint block (injected when circular/following detected)
            following_constraint_text = self._get_following_constraint_text(observation)

            # Build the action spec with prompt text
            cfg = self._agent_cfg.get(agent_id, {})
            role_prompt_extra = cfg.get("decision_prompt_extra", "")
            situation_framing = (f"{role_prompt_extra}\n\n") if role_prompt_extra else ""

            age = str(cfg.get("age", "unknown"))
            gender = str(cfg.get("gender", "person"))
            personality_profile = str(
                cfg.get("personality_anchor")
                or cfg.get("personality_type", "unknown")
            )
            journey_block = agent_goal if agent_goal else "Continue your assigned journey."
            previous_decision_summary = self._build_last_decision_summary(agent_id, current_sim_time)
            new_info_text = self._extract_new_information_text(observation)
            cue_text = ", ".join(cues) if cues else "none"
            new_since_last_decision = (
                f"{previous_decision_summary} New information: {new_info_text} "
                f"Detected cues: {cue_text}."
            )
            current_surroundings = observation

            prompt_text = self._render_decision_prompt(
                age=age,
                gender=gender,
                personality_profile=personality_profile,
                journey_block=journey_block,
                new_since_last_decision=new_since_last_decision,
                current_surroundings=current_surroundings,
                available_actions_block=available_actions_block,
                wait_action_rule_block=wait_action_rule_block,
                pace_field_block=pace_field_block,
                pace_validation_block=pace_validation_block,
                situation_framing=situation_framing,
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
            repair_status = "ok"
            decision_payload: dict[str, Any] | None = None

            if not should_call_llm and cached_decision:
                candidate_payload = self._extract_json_object(cached_decision)
                if candidate_payload is not None:
                    payload_errors = self._validate_decision_payload(
                        candidate_payload,
                        offered_actions_set,
                        offered_wait_reasons_set,
                        offered_exit_ids_set,
                    )
                    if not payload_errors:
                        decision_payload = candidate_payload
                        action = cached_decision
                        repair_status = "cached"
                        logger.info(
                            f"{agent_id}: ✓ Prompt unchanged, reusing cached decision (saved LLM call)"
                        )
                        async with self._state_lock:
                            self.llm_calls_skipped += 1

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

                if decision_payload is None:
                    should_call_llm = True
                    cached_decision = None
                    logger.info(
                        f"{agent_id}: cached decision invalid for current offered set — requesting fresh decision"
                    )

            if decision_payload is None:
                if should_call_llm and self._min_redecision_interval_secs > 0:
                    if "No significant new information." in observation:
                        previous_llm_time = self._last_llm_decision_time.get(agent_id)
                        if previous_llm_time is not None:
                            elapsed = current_sim_time - previous_llm_time
                            if elapsed < self._min_redecision_interval_secs:
                                throttled_cached = self.prompt_cache.get_cached_decision(agent_id)
                                if throttled_cached:
                                    candidate_payload = self._extract_json_object(throttled_cached)
                                    if candidate_payload is not None:
                                        payload_errors = self._validate_decision_payload(
                                            candidate_payload,
                                            offered_actions_set,
                                            offered_wait_reasons_set,
                                            offered_exit_ids_set,
                                        )
                                        if not payload_errors:
                                            decision_payload = candidate_payload
                                            action = throttled_cached
                                            repair_status = "cached"
                                            logger.info(
                                                f"{agent_id}: ✓ Reusing cached decision "
                                                f"(redecision throttle {elapsed:.1f}s "
                                                f"< {self._min_redecision_interval_secs:.1f}s)"
                                            )
                                            async with self._state_lock:
                                                self.llm_calls_skipped += 1
                                            should_call_llm = False

                if decision_payload is None:
                    with self.perf_timer.measure("agent_observe", is_parallel=True):
                        agent.observe(observation)

                    last_errors: list[str] = ["invalid response"]
                    raw_action = ""
                    attempts_used = 0

                    for attempt_idx in range(3):
                        attempts_used = attempt_idx + 1
                        attempt_prompt = prompt_text
                        if attempt_idx > 0:
                            attempt_prompt = (
                                prompt_text
                                + "\n\nSYSTEM NOTE: Your previous output violated the schema: "
                                + "; ".join(last_errors)
                                + ". Return only valid JSON that satisfies all rules."
                            )
                        attempt_action_spec = entity_lib.ActionSpec(
                            call_to_action=attempt_prompt,
                            output_type=entity_lib.OutputType.FREE,
                        )

                        async with self._llm_semaphore:
                            try:
                                with self.perf_timer.measure("agent_act_llm", is_parallel=True):
                                    llm_current_agent_id.set(agent_id)
                                    llm_current_sim_time.set(current_sim_time)
                                    if self._per_agent_timeout_secs is not None:
                                        async with asyncio.timeout(self._per_agent_timeout_secs):
                                            raw_action = await asyncio.to_thread(
                                                agent.act, attempt_action_spec
                                            )
                                    else:
                                        raw_action = await asyncio.to_thread(agent.act, attempt_action_spec)
                            except asyncio.TimeoutError:
                                timeout_secs = self._per_agent_timeout_secs
                                logger.warning(
                                    f"{agent_id}: decision timed out after {timeout_secs:.0f}s — "
                                    "keeping existing waypoint"
                                )
                                return

                        parsed = self._extract_json_object(raw_action)
                        if parsed is None:
                            last_errors = ["response was not valid JSON"]
                            continue

                        last_errors = self._validate_decision_payload(
                            parsed,
                            offered_actions_set,
                            offered_wait_reasons_set,
                            offered_exit_ids_set,
                        )
                        if not last_errors:
                            decision_payload = parsed
                            break

                    if decision_payload is None:
                        decision_payload = self._build_fallback_decision(
                            agent_id,
                            offered_actions_set,
                            offered_wait_reasons_set,
                            offered_exit_ids_set,
                        )
                        repair_status = "fallback"
                    elif attempts_used == 1:
                        repair_status = "ok"
                    elif attempts_used == 2:
                        repair_status = "repair_1"
                    else:
                        repair_status = "repair_2"

                    action = self._decision_payload_to_json(decision_payload)
                    self.prompt_cache.cache_decision(agent_id, action)
                    llm_was_called = True
                    self._last_llm_decision_time[agent_id] = current_sim_time

                    async with self._state_lock:
                        self.llm_calls_made += 1
                        self.last_observations[agent_id] = observation
                        self.last_actions[agent_id] = action

            if decision_payload is None:
                logger.warning(f"{agent_id}: no decision payload available, skipping")
                return

            self._agent_reassess_modes[agent_id] = str(
                decision_payload.get("reassess_when", "next_interval")
            )

            # Parse JSON response
            with self.perf_timer.measure("parse_json_response", is_parallel=True):
                reasoning = self._parse_json_response(action)

            # Translate action to JuPedSim command
            with self.perf_timer.measure("translate_action", is_parallel=True):
                translated = self.action_translator.translate(agent_id, action, position)

            if translated.get("action_type") == "wait":
                async with self._state_lock:
                    self._agent_wait_since.setdefault(agent_id, current_sim_time)
            else:
                async with self._state_lock:
                    self._agent_wait_since.pop(agent_id, None)

            llm_reasoning = self._reasoning_to_str(
                reasoning.get("assessment", {}) if isinstance(reasoning, dict) else ""
            )
            if llm_reasoning:
                translated["reasoning"] = llm_reasoning

            # Detect route changes and store decision
            with self.perf_timer.measure("decision_storage", is_parallel=True):
                new_exit = extract_exit_name(translated, self.station_layout)

                async with self._state_lock:
                    old_exit = self.agent_destinations.get(agent_id)

                route_changed = False

                prompt_hash = hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()
                run_meta = self.station_layout.get("run_metadata", {})
                decision_record = {
                    "time": current_sim_time,
                    "observation": observation,
                    "prompt": action_spec.call_to_action if llm_was_called else "cached",
                    "action": action,
                    "reasoning": reasoning,
                    "decision_payload": decision_payload,
                    "translated": translated,
                    "offered_set": sorted(offered_actions_set),
                    "repair_status": repair_status,
                    "run_id": run_meta.get("run_id"),
                    "condition": run_meta.get("condition"),
                    "agent_id": agent_id,
                    "t": current_sim_time,
                    "zone": zone_id,
                    "action_verb": decision_payload.get("action"),
                    "wait_reason": decision_payload.get("wait_reason"),
                    "exit_id": decision_payload.get("exit_id"),
                    "resolved_info_source": translated.get("resolved_info_source"),
                    "pace": decision_payload.get("pace"),
                    "reassess_when": decision_payload.get("reassess_when"),
                    "assessment": decision_payload.get("assessment"),
                    "prompt_hash": prompt_hash,
                    "model_id": run_meta.get("model_id"),
                    "seed": run_meta.get("seed"),
                    "cue_types": cues,
                }

                async with self._state_lock:
                    if new_exit and old_exit and old_exit != new_exit:
                        logger.info(f"🔄 {agent_id} changed route: {old_exit} → {new_exit}")
                        route_changed = True

                    if route_changed:
                        decision_record["route_change"] = {
                            "from_exit": old_exit,
                            "to_exit": new_exit,
                            "reason": self._reasoning_to_str(reasoning.get("assessment", {})),
                        }

                    _MAX_DECISIONS_PER_AGENT = 200
                    if agent_id not in self.agent_decisions:
                        self.agent_decisions[agent_id] = {"decisions": []}

                    decisions_list = self.agent_decisions[agent_id]["decisions"]
                    decisions_list.append(decision_record)
                    if len(decisions_list) > _MAX_DECISIONS_PER_AGENT:
                        decisions_list.pop(0)

            with self.perf_timer.measure("apply_to_jupedsim", is_parallel=True):
                self.action_executor.execute_action(agent_id, translated, current_sim_time)

            logger.info(f"{agent_id} action: {action[:100]}...")

        except Exception as e:
            logger.error(f"Error processing {agent_id}: {e}", exc_info=True)

    def _parse_json_response(self, response: str) -> dict[str, Any]:
        """Parse JSON response and expose assessment/action keys."""
        parsed = self._extract_json_object(response)
        if parsed is None:
            logger.warning(f"Failed to parse JSON response: {response[:200]}")
            return {"assessment": {}, "action": "", "raw": response[:200]}
        return {
            "assessment": parsed.get("assessment", {}),
            "action": parsed.get("action", ""),
            "wait_reason": parsed.get("wait_reason"),
            "exit_id": parsed.get("exit_id"),
            "pace": parsed.get("pace"),
            "reassess_when": parsed.get("reassess_when"),
        }

    @staticmethod
    def _reasoning_to_str(reasoning_val) -> str:
        """Convert assessment text to a compact single-line summary."""
        if isinstance(reasoning_val, str):
            return reasoning_val
        if isinstance(reasoning_val, dict):
            parts = []
            for key in ("situation_appraisal", "option_chosen_because"):
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
        """Build the Block 3 summary with previous verb/target only."""
        decisions = self.agent_decisions.get(agent_id, {}).get("decisions", [])
        if not decisions:
            return "First decision point for you."

        last = decisions[-1]
        last_time = float(last.get("time", current_sim_time))
        seconds_ago = max(0.0, current_sim_time - last_time)

        payload = last.get("decision_payload", {}) if isinstance(last, dict) else {}
        action = str(payload.get("action", "continue_activity"))
        wait_reason = payload.get("wait_reason")
        exit_id = payload.get("exit_id")

        target_text = ""
        if action == "wait" and wait_reason:
            target_text = f" (reason: {wait_reason})"
        elif action == "evacuate" and exit_id:
            target_text = f" (exit_id: {exit_id})"

        return f"{seconds_ago:.0f}s ago your previous choice was action='{action}'{target_text}."

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
