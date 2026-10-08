"""Perception: what an agent knows and may do at a decision point.

For each deciding agent the :class:`SituationAssembler` turns simulation state
and the agent's natural-language observation into a
:class:`~evacusim.core.decision_engine.DecisionContext`, in two steps::

    perceive(...)  ->  Perception        goal, observation text, exits the agent
                                          knows are usable, cues since last time
    frame(...)     ->  DecisionContext   the offered actions and exits, with the
                                          routing signals a rule engine weighs

The orchestrator may skip an agent between the two (an agent waiting for a new
cue that has none), so the more expensive framing is done only when needed.

Both decision engines receive the same DecisionContext. The LLM engine also
renders it into a prompt (:mod:`evacusim.decision.llm_prompt`).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from shapely.geometry import Point

from evacusim.conventions import is_train_exit, platform_zone
from evacusim.core.decision_engine import DecisionContext, ExitOption
from evacusim.utils.logger import get_logger

logger = get_logger(__name__)

# Phrases produced by the observation formatter that perception relies on.
_RE_VISIBLE_EXITS_LINE = re.compile(r"Exits visible right now:\s*([^\.]+)\.")
_RE_EXITS_LINE_ENTRY = re.compile(r"(.+?)\s+\((very close|nearby|visible in distance)\)")
_RE_BLOCKED_EXIT_LINE = re.compile(r"The (.+?) appears blocked or obstructed")

EVACUATION_GOAL = "Leave the station via a safe exit."
"""The goal of every agent who has decided to evacuate."""

WAIT_REASONS = ("awaiting_information", "awaiting_instruction")


def goal_is_train_oriented(goal: str) -> bool:
    """True when the goal is to reach a platform or board a train."""
    g = (goal or "").lower()
    return any(k in g for k in ("train", "platform", "board"))


def visible_exit_names(observation: str) -> set[str]:
    """Display names of exits the observation says are visible."""
    match = _RE_VISIBLE_EXITS_LINE.search(observation)
    if not match:
        return set()
    return {m.group(1).strip() for m in _RE_EXITS_LINE_ENTRY.finditer(match.group(1))}


def blocked_exit_names(observation: str) -> list[str]:
    """Display names of exits the observation says are blocked."""
    return [m.group(1).strip() for m in _RE_BLOCKED_EXIT_LINE.finditer(observation)]


def zone_containing(position: tuple[float, float], zones_polygons: dict) -> str | None:
    """The smallest named zone containing ``position``.

    Specific zones (``platform_1``) win over large background polygons
    (``level_-1``) that also contain the point.
    """
    point = Point(position)
    best_zone: str | None = None
    best_area = float("inf")
    for zone_id, polygon in zones_polygons.items():
        try:
            if (polygon.covers(point) or polygon.contains(point)) and polygon.area < best_area:
                best_area = polygon.area
                best_zone = zone_id
        except Exception:
            pass
    return best_zone


class GoalTracker:
    """Each agent's current goal, and whether they have committed to evacuating.

    An agent starts with the goal from its role (``initial_goal``). Choosing to
    evacuate replaces it with :data:`EVACUATION_GOAL` until an explicit
    all-clear. Descending to a platform to board a train is movement within
    the station, not evacuation, so it never replaces a train goal.
    """

    def __init__(self, agent_cfg: dict[str, dict]) -> None:
        self._agent_cfg = agent_cfg
        self.goals: dict[str, str] = {}
        self.committed: set[str] = set()

    def current(self, agent_id: str) -> str:
        """The agent's goal now, initialising it from the role on first use."""
        if agent_id in self.committed:
            self.goals[agent_id] = EVACUATION_GOAL
        elif agent_id not in self.goals:
            self.goals[agent_id] = self._agent_cfg.get(agent_id, {}).get("initial_goal", "")
        return self.goals[agent_id]

    def apply_decision(self, agent_id: str, payload: dict[str, Any], observation: str) -> None:
        """Update commitment after a decision (or release it on an all-clear)."""
        text = (observation or "").lower()
        if "all clear" in text or "all-clear" in text:
            if agent_id in self.committed:
                self.committed.discard(agent_id)
                self.goals.pop(agent_id, None)
                logger.info(f"{agent_id}: all-clear cue detected — released evacuation commitment")
            return

        if str(payload.get("action", "")) != "evacuate":
            return
        if goal_is_train_oriented(str(self.goals.get(agent_id, "") or "")):
            return
        if agent_id not in self.committed:
            logger.info(f"{agent_id}: evacuation decision committed as persistent goal")
        self.committed.add(agent_id)
        self.goals[agent_id] = EVACUATION_GOAL

    def clear_for_redecision(self, agent_id: str) -> None:
        """Forget a non-evacuation goal so it is re-derived (e.g. after changing level)."""
        if agent_id in self.committed:
            self.goals[agent_id] = EVACUATION_GOAL
        else:
            self.goals.pop(agent_id, None)


class CueDetector:
    """Detects changes since an agent's last decision that should wake it.

    Cues: ``zone_entry``, ``route_blocked`` / ``route_unblocked``,
    ``pa_message``, ``staff_instruction``, ``train_arrival`` and
    ``alarm_state_change``. Detection reads the observation text.
    """

    def __init__(self) -> None:
        self._last_zone: dict[str, str | None] = {}
        self._last_route_blocked: dict[str, bool] = {}
        self._last_alarm_signature: dict[str, str] = {}
        self._last_train_hint: dict[str, str] = {}

    def detect(
        self, agent_id: str, observation: str, zone_id: str | None, route_blocked: bool
    ) -> list[str]:
        cues: list[str] = []
        obs_lower = observation.lower()

        last_zone = self._last_zone.get(agent_id)
        if last_zone is None:
            self._last_zone[agent_id] = zone_id
        elif zone_id is not None and zone_id != last_zone:
            cues.append("zone_entry")
            self._last_zone[agent_id] = zone_id

        prev_blocked = self._last_route_blocked.get(agent_id)
        if prev_blocked is None:
            self._last_route_blocked[agent_id] = route_blocked
        elif prev_blocked != route_blocked:
            cues.append("route_blocked" if route_blocked else "route_unblocked")
            self._last_route_blocked[agent_id] = route_blocked

        if "(directing you) said:" in obs_lower:
            if "pa" in obs_lower or "pa system" in obs_lower:
                cues.append("pa_message")
            else:
                cues.append("staff_instruction")

        train_hint = ""
        if "train" in obs_lower and ("arriv" in obs_lower or "waiting" in obs_lower):
            train_hint = "train_arrival"
        if train_hint and train_hint != self._last_train_hint.get(agent_id, ""):
            cues.append("train_arrival")
        self._last_train_hint[agent_id] = train_hint

        alarm_signature = ""
        if "alarm" in obs_lower:
            if "all clear" in obs_lower or "all-clear" in obs_lower:
                alarm_signature = "all_clear"
            elif "sounding" in obs_lower or "activated" in obs_lower or "onset" in obs_lower:
                alarm_signature = "alarm_onset"
            else:
                alarm_signature = "alarm_state"
        if alarm_signature and alarm_signature != self._last_alarm_signature.get(agent_id, ""):
            cues.append("alarm_state_change")
        if alarm_signature:
            self._last_alarm_signature[agent_id] = alarm_signature

        return cues


@dataclass
class Perception:
    """What an agent perceives at a decision point (the first assembly step)."""

    agent_id: str
    position: tuple[float, float]
    zone_id: str | None
    goal: str
    observation: str
    """The observation text, prefixed with the agent's goal context."""
    offered_exit_ids: list[str]
    route_blocked: bool
    cues: list[str]


class SituationAssembler:
    """Builds each deciding agent's :class:`DecisionContext`.

    Args:
        agent_cfg: Per-agent records (role, profile, target...), keyed by id;
            shared with the orchestrator, which adds runtime-spawned agents.
        station_layout: Station knowledge from the ``station`` config section
            plus the geometry-derived layout.
        action_translator: Provides the exit registry, zones and exit coordinates.
        action_executor: Answers whether an information source is reachable.
        agent_destinations: Each agent's current target exit (shared, live).
        jps_sim: The pedestrian simulation (positions, levels, route distances).
        wait_nudge_enabled: Remind long-waiting agents to reassess.
        message_system: Source of each agent's perceived warning cues.
        state_queries: Nearby-agent lookup.
        neighbour_radius_m: Radius for ``nearby_agent_ids``; ``None`` skips the
            lookup (only the rule engine uses it).
    """

    def __init__(
        self,
        agent_cfg: dict[str, dict],
        station_layout: dict[str, Any],
        action_translator,
        action_executor,
        agent_destinations: dict[str, str],
        jps_sim=None,
        wait_nudge_enabled: bool = False,
        message_system=None,
        state_queries=None,
        neighbour_radius_m: float | None = None,
    ) -> None:
        self._agent_cfg = agent_cfg
        self._message_system = message_system
        self._state_queries = state_queries
        self._neighbour_radius_m = neighbour_radius_m
        self._translator = action_translator
        self._executor = action_executor
        self._destinations = agent_destinations
        self._jps_sim = jps_sim
        self._wait_nudge_enabled = wait_nudge_enabled

        self.goals = GoalTracker(agent_cfg)
        self.cues = CueDetector()
        self.wait_since: dict[str, float] = {}
        """When each currently waiting agent started waiting (for the wait nudge)."""

        self._arrival_exits_by_zone: dict[str, list[str]] = station_layout.get(
            "arrival_exits_by_zone", {}
        )
        self._zone_known_exits: dict[str, dict[str, list[str]]] = station_layout.get(
            "zone_known_exits_by_profile", {}
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
        self._platform_down_exits: dict[str, list[str]] = station_layout.get(
            "platform_down_exits", {}
        )

    # -- step 1: perceive ---------------------------------------------------

    def perceive(
        self,
        agent_id: str,
        position: tuple[float, float],
        zone_id: str | None,
        observation: str,
        current_sim_time: float,
    ) -> Perception:
        """Goal, goal-prefixed observation, usable exits and new cues."""
        goal = self.goals.current(agent_id)

        goal_lines: list[str] = []
        if goal:
            goal_lines.append(f"Your current goal: {goal}")
            if zone_id:
                keywords = self._zone_goal_keywords.get(zone_id, [])
                if any(kw.lower() in goal.lower() for kw in keywords):
                    goal_lines.append("Goal status: You are currently at your goal destination.")

        # A message that changes every minute an agent waits busts the LLM
        # prompt cache, forcing it to reconsider rather than wait forever.
        wait_start = self.wait_since.get(agent_id)
        if self._wait_nudge_enabled and wait_start is not None:
            wait_secs = current_sim_time - wait_start
            if wait_secs >= 60:
                goal_lines.append(
                    f"⚠️ You have been stationary for {int(wait_secs / 60)} minute(s) "
                    f"while the evacuation alarm is active. "
                    f"Reassess whether to continue waiting or move."
                )

        if goal_lines:
            observation = "\n".join(goal_lines) + "\n\n" + observation

        offered_exit_ids = self._usable_exit_ids(agent_id, observation, zone_id)
        route_blocked = self._is_route_blocked(agent_id, observation)
        cues = self.cues.detect(agent_id, observation, zone_id, route_blocked)
        return Perception(
            agent_id=agent_id,
            position=position,
            zone_id=zone_id,
            goal=goal,
            observation=observation,
            offered_exit_ids=offered_exit_ids,
            route_blocked=route_blocked,
            cues=cues,
        )

    def _usable_exit_ids(self, agent_id: str, observation: str, zone_id: str | None) -> list[str]:
        """Exits the agent can choose now, in offer order.

        Commuters recall the exits they know in this zone; everyone can choose
        exits they see. If neither yields any, the profile's known exits are a
        fallback. Exits that lead *into* this zone, and exits the agent has
        seen are blocked, are excluded.
        """
        registry = self._translator.exit_registry
        valid_ids = set(registry.get_all_ids())
        profile = self._agent_cfg.get(agent_id, {}).get("knowledge_profile", "novice")
        arrival_exits = set(self._arrival_exits_by_zone.get(zone_id or "", []))
        blocked_display = set(blocked_exit_names(observation))
        visible_names = visible_exit_names(observation)
        known = self._zone_known_exits.get(zone_id or "", {})

        def usable(eid: str) -> bool:
            # Blocked status is knowledge-driven: only exits the agent has
            # observed to be blocked count as blocked.
            return (
                eid in valid_ids
                and eid not in arrival_exits
                and registry.get_display_name(eid) not in blocked_display
            )

        candidates: list[str] = []
        if profile == "commuter":
            candidates += [eid for eid in known.get("commuter", []) if usable(eid)]
        candidates += [
            eid
            for eid in sorted(valid_ids)
            if usable(eid) and registry.get_display_name(eid) in visible_names
        ]
        if not candidates:
            candidates = [eid for eid in known.get(profile, []) if usable(eid)]
        return list(dict.fromkeys(candidates))

    def _is_route_blocked(self, agent_id: str, observation: str) -> bool:
        """True if the agent's current target exit is reported blocked."""
        current_dest = self._destinations.get(agent_id)
        if not current_dest:
            return False
        registry = getattr(self._translator, "exit_registry", None)
        current_display = (
            registry.get_display_name(current_dest) if registry is not None else current_dest
        )
        return current_display in blocked_exit_names(observation)

    # -- step 2: frame the choice ----------------------------------------------

    def frame(self, perception: Perception, current_sim_time: float) -> DecisionContext:
        """The offered actions and exits, with routing signals, for one agent."""
        agent_id = perception.agent_id
        cfg = self._agent_cfg.get(agent_id, {})
        offered_exit_ids = perception.offered_exit_ids

        offered_wait_reasons = list(WAIT_REASONS)
        if perception.route_blocked:
            offered_wait_reasons.append("route_blocked")

        offered_actions = ["continue_activity", "wait"]
        if self._executor.has_reachable_information_source(
            agent_id, perception.position, perception.zone_id
        ):
            offered_actions.append("seek_information")
        if offered_exit_ids:
            offered_actions.append("evacuate")
        if self._has_train_service():
            offered_actions.append("leave_by_train")

        exit_options = self._exit_options(
            agent_id,
            perception.position,
            perception.zone_id,
            offered_exit_ids,
            perception.observation,
        )
        goal_policy = self.goal_policy(perception.zone_id, perception.goal)
        prefer_exit_tags: tuple[str, ...] = ()
        avoid_exit_tags: tuple[str, ...] = ()
        if goal_policy:
            prefer_exit_tags = _clean_tags(goal_policy.get("prefer_exit_tags", []))
            avoid_exit_tags = _clean_tags(goal_policy.get("avoid_exit_tags", []))

        return DecisionContext(
            agent_id=agent_id,
            position=perception.position,
            zone_id=perception.zone_id,
            goal=perception.goal,
            observation=perception.observation,
            agent_cfg=cfg,
            offered_actions=offered_actions,
            offered_wait_reasons=offered_wait_reasons,
            offered_exit_ids=offered_exit_ids,
            exit_options=exit_options,
            route_blocked=perception.route_blocked,
            cues=perception.cues,
            current_sim_time=current_sim_time,
            offered_actions_set=set(offered_actions),
            offered_wait_reasons_set=set(offered_wait_reasons),
            offered_exit_ids_set=set(offered_exit_ids),
            prefer_exit_tags=prefer_exit_tags,
            avoid_exit_tags=avoid_exit_tags,
            preferred_exit_ids=self._preferred_connectors(cfg.get("target"), offered_exit_ids),
            committed_exit_id=self._destinations.get(agent_id),
            is_moving=self._executor.agent_action.get(agent_id) == "moving"
            if hasattr(self._executor, "agent_action")
            else False,
            goal_policy=goal_policy,
            warnings=self._warnings(agent_id),
            nearby_agent_ids=self._nearby(agent_id),
        )

    def _warnings(self, agent_id: str) -> tuple[dict[str, Any], ...]:
        if self._message_system is None or not hasattr(self._message_system, "cues_for"):
            return ()
        return tuple(self._message_system.cues_for(agent_id))

    def _nearby(self, agent_id: str) -> tuple[str, ...]:
        if self._neighbour_radius_m is None or self._state_queries is None:
            return ()
        nearby = self._state_queries.get_nearby_agents(agent_id, self._neighbour_radius_m)
        return tuple(a["id"] for a in nearby if a.get("id") != agent_id)

    def goal_policy(self, zone_id: str | None, goal: str) -> dict[str, Any] | None:
        """The first ``station.goal_semantic_policies`` entry matching this goal and zone."""
        if not goal or not self._goal_semantic_policies:
            return None
        goal_lower = goal.lower()
        zone_lower = (zone_id or "").lower()
        for policy in self._goal_semantic_policies:
            if not isinstance(policy, dict):
                continue
            keywords = [
                str(k).strip().lower()
                for k in policy.get("when_goal_contains_any", [])
                if str(k).strip()
            ]
            if not keywords or not any(keyword in goal_lower for keyword in keywords):
                continue
            zones = {
                str(z).strip().lower() for z in policy.get("applies_in_zones", []) if str(z).strip()
            }
            if zones and zone_lower not in zones:
                continue
            return policy
        return None

    def _has_train_service(self) -> bool:
        """True if the station has train exits (boarding is possible at all)."""
        try:
            return any(is_train_exit(e) for e in self._translator.exit_registry.get_all_ids())
        except Exception:
            return False

    def _preferred_connectors(self, target: Any, offered_exit_ids: list[str]) -> tuple[str, ...]:
        """The offered down-escalator(s) serving the agent's target platform.

        Maps ``train_platform_N`` / ``platform_N`` through
        ``station.platform_down_exits`` so a boarder descends via the bank
        serving its platform rather than the nearest one.
        """
        if not self._platform_down_exits:
            return ()
        zone = platform_zone(target)
        if not zone:
            return ()
        offered = set(offered_exit_ids)
        return tuple(c for c in self._platform_down_exits.get(zone, []) if c in offered)

    def _exit_options(
        self,
        agent_id: str,
        position: tuple[float, float],
        zone_id: str | None,
        offered_exit_ids: list[str],
        observation: str,
    ) -> dict[str, ExitOption]:
        """Routing signals per offered exit: distance, crowd, familiarity, visibility."""
        options: dict[str, ExitOption] = {}
        registry = getattr(self._translator, "exit_registry", None)
        cfg = self._agent_cfg.get(agent_id, {})
        profile = cfg.get("knowledge_profile", "novice")
        jps_sim = self._jps_sim
        live_level = None
        try:
            if jps_sim and hasattr(jps_sim, "get_agent_level"):
                live_level = jps_sim.get_agent_level(agent_id)
        except Exception:
            live_level = None
        agent_level = str(live_level if live_level is not None else cfg.get("level_id", "0"))

        zone_exits = self._zone_known_exits.get(zone_id or "", {})
        known_ids = set(zone_exits.get(profile, []))
        if profile == "commuter":
            known_ids |= set(zone_exits.get("commuter", []))
        visible_names = visible_exit_names(observation)

        try:
            all_positions = jps_sim.get_all_agent_positions() if jps_sim else {}
        except Exception:
            all_positions = {}
        crowd_radius_sq = 5.0**2

        for exit_id in offered_exit_ids:
            display = exit_id
            try:
                if registry is not None and hasattr(registry, "get_display_name"):
                    display = registry.get_display_name(exit_id) or exit_id
            except Exception:
                display = exit_id

            try:
                coords = self._translator._get_exit_coordinates(exit_id, agent_level)
            except Exception:
                coords = None

            distance_m = None
            route_distance_m = None
            crowd_count = 0
            if coords is not None:
                dx = position[0] - coords[0]
                dy = position[1] - coords[1]
                distance_m = (dx * dx + dy * dy) ** 0.5
                try:
                    if jps_sim and hasattr(jps_sim, "get_route_distance"):
                        route_distance_m = jps_sim.get_route_distance(agent_id, position, coords)
                except Exception:
                    route_distance_m = None
                escalators = getattr(jps_sim, "escalator_system", None)
                if escalators is not None and escalators.handles(exit_id):
                    # Everyone queueing for, stepping onto or riding it: the
                    # queue spreads over the landing and riders are off-floor.
                    crowd_count = escalators.load(exit_id)
                    if escalators.queued_exit(agent_id) == exit_id:
                        crowd_count -= 1
                else:
                    for other_id, p in all_positions.items():
                        if other_id == agent_id:
                            continue
                        if (p[0] - coords[0]) ** 2 + (p[1] - coords[1]) ** 2 <= crowd_radius_sq:
                            crowd_count += 1

            options[exit_id] = ExitOption(
                exit_id=exit_id,
                display_name=display,
                distance_m=distance_m,
                route_distance_m=route_distance_m,
                crowd_count=crowd_count,
                familiar=exit_id in known_ids,
                visible=display in visible_names,
                semantic_tags=tuple(self._exit_semantic_tags.get(exit_id, [])),
            )
        return options


def _clean_tags(tags: list) -> tuple[str, ...]:
    return tuple(str(t).strip() for t in tags if str(t).strip())
