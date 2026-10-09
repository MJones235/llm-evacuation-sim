"""Rule-based decision engine: a cue-driven response, then exit routing.

Each agent passes through three stages:

==============  ==============================================================
UNAWARE         No warning yet: the agent pursues its journey (catch a train,
                leave the station) using the routing policy below.
AWARE           A warning cue has been perceived: the agent stops its journey
                and investigates (seeks information where it can, otherwise
                waits).
EVACUATING      The agent acts: it follows the most recent instruction
                (``board_train``), or leaves the station by the best route.
==============  ==============================================================

Cues are the alarm, PA announcements and staff directives an agent has
perceived (annotated with a ``strength`` and an ``instruction`` in the scenario
config), plus a *social* cue when enough people nearby are already
evacuating. Each cue schedules evacuation at ``cue time + delay``, the delay
drawn from a lognormal whose median depends on the cue's strength; the
earliest scheduled time wins, so a stronger later cue can bring evacuation
forward but never delay it.

Routing (UNAWARE journeys and EVACUATING agents) scores the offered exits on
a weighted combination of signals carried on each
:class:`~evacusim.core.decision_engine.ExitOption`:

* **proximity** - shorter navigable routes score higher,
* **visibility** - exits in direct line of sight score higher,
* **busyness** - less crowded exits score higher,
* **familiarity** - exits the agent already knows score higher.

Delays are drawn from per-agent random streams derived from the run seed, so
runs are reproducible.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any

from evacusim.core.decision_engine import (
    DecisionContext,
    DecisionEngine,
    DecisionResult,
    ExitOption,
)
from evacusim.utils.seeding import derive_seed

_DEFAULT_PACE = "normal_pace"

# Goal keywords that indicate the agent is trying to board a train / reach a
# platform rather than leave the station.
_TRAIN_GOAL_KEYWORDS = ("train", "platform", "board")

UNAWARE, AWARE, EVACUATING = "unaware", "aware", "evacuating"


@dataclass
class _AgentState:
    """One agent's progress through the stages."""

    rng: random.Random
    stage: str = UNAWARE
    stage_since: float = 0.0
    cues_seen: int = 0
    evacuate_at: float = math.inf
    instruction: str = "none"
    social_cued: bool = False
    triggers: list[str] = field(default_factory=list)


class RuleBasedDecisionEngine(DecisionEngine):
    """Cue-driven stages (unaware, aware, evacuating) with weighted exit routing.

    Args:
        w_proximity, w_busyness, w_familiarity, w_visibility: Exit-scoring weights.
        crowd_radius_m: Radius for counting the crowd at an exit.
        pace: Pace for journey movement.
        response_median_s: Median delay (s) from a cue to evacuating, by
            strength (``weak``, ``medium``, ``strong``).
        response_sigma: Lognormal shape of those delays (0: always the median).
        social_enabled: Treat "enough neighbours evacuating" as a cue.
        social_radius_m: Who counts as a neighbour.
        social_threshold: Fraction of neighbours evacuating that is a cue.
        social_min_neighbours: Fewer neighbours than this never make a cue.
        social_strength: Strength of the social cue.
        evacuation_pace: Pace once evacuating.
        leave_prefer_tags / leave_avoid_tags: Exit tags (``station.exit_semantic_tags``)
            preferred and avoided when leaving the station.
        seed: Run seed (for the per-agent delay draws).
    """

    def __init__(
        self,
        w_proximity: float = 0.5,
        w_busyness: float = 0.3,
        w_familiarity: float = 0.2,
        w_visibility: float = 0.0,
        crowd_radius_m: float = 5.0,
        pace: str = _DEFAULT_PACE,
        response_median_s: dict[str, float] | None = None,
        response_sigma: float = 0.6,
        social_enabled: bool = True,
        social_radius_m: float = 5.0,
        social_threshold: float = 0.5,
        social_min_neighbours: int = 2,
        social_strength: str = "medium",
        evacuation_pace: str = _DEFAULT_PACE,
        leave_prefer_tags: tuple[str, ...] = ("outside_station", "to_street", "to_concourse"),
        leave_avoid_tags: tuple[str, ...] = ("to_platform", "to_train"),
        seed: int = 0,
    ) -> None:
        self.w_proximity = float(w_proximity)
        self.w_busyness = float(w_busyness)
        self.w_familiarity = float(w_familiarity)
        self.w_visibility = float(w_visibility)
        self.crowd_radius_m = float(crowd_radius_m)
        self.pace = pace
        self.response_median_s = dict(
            response_median_s or {"weak": 450.0, "medium": 75.0, "strong": 40.0}
        )
        self.response_sigma = float(response_sigma)
        self.social_enabled = social_enabled
        self.neighbour_radius_m = float(social_radius_m)
        self.social_threshold = float(social_threshold)
        self.social_min_neighbours = int(social_min_neighbours)
        self.social_strength = social_strength
        self.evacuation_pace = evacuation_pace
        self.leave_prefer_tags = set(leave_prefer_tags)
        self.leave_avoid_tags = set(leave_avoid_tags)
        self.seed = int(seed)
        self._agents: dict[str, _AgentState] = {}

    # ------------------------------------------------------------------ #
    # DecisionEngine interface
    # ------------------------------------------------------------------ #
    async def decide(self, ctx: DecisionContext) -> DecisionResult:
        state = self._update_stage(ctx)
        if state.stage == EVACUATING:
            payload = self._evacuation_payload(ctx, state)
        elif state.stage == AWARE:
            payload = self._investigation_payload(ctx, state)
        else:
            payload = self._decide_payload(ctx)
        return DecisionResult(
            payload=payload,
            llm_was_called=False,
            repair_status="rule_based",
            stage=state.stage,
        )

    def stage_of(self, agent_id: str) -> str:
        """The agent's current stage (``unaware`` if it has never decided)."""
        state = self._agents.get(agent_id)
        return state.stage if state else UNAWARE

    # ------------------------------------------------------------------ #
    # Stages
    # ------------------------------------------------------------------ #
    def _state(self, agent_id: str) -> _AgentState:
        if agent_id not in self._agents:
            rng = random.Random(derive_seed(self.seed, f"rule_engine:{agent_id}"))
            self._agents[agent_id] = _AgentState(rng=rng)
        return self._agents[agent_id]

    def _update_stage(self, ctx: DecisionContext) -> _AgentState:
        """Take in new cues and move the agent on through the stages."""
        state = self._state(ctx.agent_id)
        now = ctx.current_sim_time
        # Agents spawned during the run (calibration) record when they arrived;
        # everyone else has been present from the start.
        arrived = float(ctx.agent_cfg.get("spawn_time_s") or -math.inf)

        new_cues = list(ctx.warnings[state.cues_seen :])
        state.cues_seen = len(ctx.warnings)
        social = self._social_cue(ctx, state)
        if social is not None:
            new_cues.append(social)

        for cue in new_cues:
            # A cue heard before the agent arrived starts when the agent arrives.
            start = max(float(cue["time"]), arrived)
            delay = self._draw_delay(state.rng, cue["strength"])
            state.evacuate_at = min(state.evacuate_at, start + delay)
            if cue.get("instruction", "none") != "none":
                state.instruction = cue["instruction"]
            state.triggers.append(f"{cue['source']}:{cue['strength']}@{start:.0f}s")
            if state.stage == UNAWARE:
                state.stage, state.stage_since = AWARE, start

        if state.stage == AWARE and now >= state.evacuate_at:
            state.stage, state.stage_since = EVACUATING, now
        return state

    def _draw_delay(self, rng: random.Random, strength: str) -> float:
        median = self.response_median_s[strength]
        if self.response_sigma <= 0:
            return median
        return rng.lognormvariate(math.log(median), self.response_sigma)

    def _social_cue(self, ctx: DecisionContext, state: _AgentState) -> dict[str, Any] | None:
        """A cue when enough neighbours were already evacuating (once per agent)."""
        if not self.social_enabled or state.social_cued or state.stage == EVACUATING:
            return None
        neighbours = ctx.nearby_agent_ids
        if len(neighbours) < self.social_min_neighbours:
            return None
        now = ctx.current_sim_time
        # Only those who began evacuating before now: independent of the order
        # in which agents decide within a cycle.
        evacuating = sum(
            1
            for n in neighbours
            if n in self._agents
            and self._agents[n].stage == EVACUATING
            and self._agents[n].stage_since < now
        )
        if evacuating / len(neighbours) < self.social_threshold:
            return None
        state.social_cued = True
        return {"time": now, "source": "social", "strength": self.social_strength}

    def _investigation_payload(self, ctx: DecisionContext, state: _AgentState) -> dict[str, Any]:
        """AWARE: look for information, or wait for it."""
        reason = (
            f"Warning perceived ({', '.join(state.triggers)}); investigating before acting "
            f"(will act from t={state.evacuate_at:.0f}s unless a stronger cue arrives)."
        )
        if "seek_information" in ctx.offered_actions_set:
            return {
                "action": "seek_information",
                "wait_reason": None,
                "exit_id": None,
                "pace": self.pace,
                "reassess_when": "next_interval",
                "assessment": self._assessment(reason, ctx),
            }
        payload = self._wait_payload(ctx, prefer="awaiting_information")
        payload["assessment"] = self._assessment(reason, ctx)
        return payload

    def _evacuation_payload(self, ctx: DecisionContext, state: _AgentState) -> dict[str, Any]:
        """EVACUATING: follow the instruction, else leave the station by the best route."""
        actions = ctx.offered_actions_set
        candidates = [ctx.exit_options[e] for e in ctx.offered_exit_ids if e in ctx.exit_options]

        if state.instruction == "board_train":
            if "leave_by_train" in actions:
                return self._move_payload(
                    "leave_by_train",
                    None,
                    "Instructed to board the train.",
                    ctx,
                    pace=self.evacuation_pace,
                )
            boardable = self._filter_by_tags(candidates, {"to_train"})
            if "evacuate" in actions and boardable:
                best, why = self._pick_best_exit(boardable)
                return self._move_payload(
                    "evacuate",
                    best.exit_id,
                    "Instructed to board the train: " + why,
                    ctx,
                    pace=self.evacuation_pace,
                )

        if "evacuate" in actions and candidates:
            pool = self._apply_tag_preferences(
                candidates, self.leave_prefer_tags, self.leave_avoid_tags
            )
            committed = next((o for o in pool if o.exit_id == ctx.committed_exit_id), None)
            if committed is not None:
                return self._move_payload(
                    "evacuate",
                    committed.exit_id,
                    f"Evacuating; continuing toward {committed.display_name}.",
                    ctx,
                    pace=self.evacuation_pace,
                )
            best, why = self._pick_best_exit(pool)
            return self._move_payload(
                "evacuate", best.exit_id, "Evacuating: " + why, ctx, pace=self.evacuation_pace
            )

        if "wait" in actions:
            return self._wait_payload(
                ctx, prefer="route_blocked" if ctx.route_blocked else "awaiting_information"
            )
        return self._decide_payload(ctx)

    # ------------------------------------------------------------------ #
    # Journey routing (UNAWARE)
    # ------------------------------------------------------------------ #
    def _decide_payload(self, ctx: DecisionContext) -> dict[str, Any]:
        actions = ctx.offered_actions_set
        candidates = [ctx.exit_options[e] for e in ctx.offered_exit_ids if e in ctx.exit_options]
        prefer = {t for t in (ctx.prefer_exit_tags or ()) if t}
        avoid = {t for t in (ctx.avoid_exit_tags or ()) if t}

        # 1. Board a train when that is offered and the goal is train/platform-oriented.
        if "leave_by_train" in actions and self._goal_wants_train(ctx.goal):
            return self._move_payload(
                action="leave_by_train",
                exit_id=None,
                reason="Goal is to board a train and train service is available.",
                ctx=ctx,
            )

        # 2. Goal-directed movement toward a train/platform goal.
        #    Priority (a): if a train is actually boardable now — an active
        #    ``to_train`` exit is offered (the event layer only exposes a
        #    train-platform exit while a train dwells) — board it immediately.
        #    Priority (b): otherwise advance ONLY via a platform-ward connector
        #    (e.g. a down-escalator tagged ``to_platform``). When neither is
        #    reachable — e.g. already on the platform with no train dwelling —
        #    WAIT for a train rather than leaving via a street or up exit. This
        #    is what makes a boarder descend the concourse, hold on the platform,
        #    and board the next available train.
        if self._goal_wants_train(ctx.goal):
            boardable = self._filter_by_tags(candidates, {"to_train"})
            if "evacuate" in actions and boardable:
                best, why = self._pick_best_exit(boardable)
                return self._move_payload(
                    action="evacuate",
                    exit_id=best.exit_id,
                    reason="Boarding the waiting train: " + why,
                    ctx=ctx,
                )
            # Priority (a.5): head for the connector that serves THIS agent's
            # target platform (e.g. the down-escalator for platform 3), when the
            # processor resolved one and it is offered.  This routes a boarder to
            # the correct escalator bank rather than the merely-nearest platform
            # connector.
            preferred_ids = {e for e in (ctx.preferred_exit_ids or ()) if e}
            if "evacuate" in actions and preferred_ids:
                targeted = [o for o in candidates if o.exit_id in preferred_ids]
                if targeted:
                    best, why = self._pick_best_exit(targeted)
                    return self._move_payload(
                        action="evacuate",
                        exit_id=best.exit_id,
                        reason="Advancing toward the target platform: " + why,
                        ctx=ctx,
                    )
            include = (
                (prefer | self._DEFAULT_TRAIN_PREFER_TAGS)
                if prefer
                else self._DEFAULT_TRAIN_PREFER_TAGS
            )
            preferred = self._filter_by_tags(candidates, include)
            if "evacuate" in actions and preferred:
                best, why = self._pick_best_exit(preferred)
                return self._move_payload(
                    action="evacuate",
                    exit_id=best.exit_id,
                    reason="Advancing toward the platform: " + why,
                    ctx=ctx,
                )
            if "wait" in actions:
                return self._wait_payload(ctx, prefer="awaiting_information")
            # No wait offered (unusual) — fall through to the generic handling.

        # 3. Otherwise leave via the best-scoring exit, honouring avoid/prefer tags.
        if "evacuate" in actions and candidates:
            pool = self._apply_tag_preferences(candidates, prefer, avoid)
            committed = next(
                (option for option in pool if option.exit_id == ctx.committed_exit_id),
                None,
            )
            if committed is not None:
                return self._move_payload(
                    action="evacuate",
                    exit_id=committed.exit_id,
                    reason=f"Continuing toward the previously chosen {committed.display_name}.",
                    ctx=ctx,
                )
            best, why = self._pick_best_exit(pool)
            return self._move_payload(
                action="evacuate",
                exit_id=best.exit_id,
                reason=why,
                ctx=ctx,
            )

        # 3. No usable exit — wait when the route is blocked.
        if "wait" in actions and ctx.route_blocked:
            return self._wait_payload(ctx, prefer="route_blocked")

        # 4. Nothing to do toward the goal this cycle — continue current activity.
        if "continue_activity" in actions:
            return self._move_payload(
                action="continue_activity",
                exit_id=None,
                reason="No reachable exit advances the goal right now; continue current activity.",
                ctx=ctx,
            )

        # 5. Last resort — wait.
        if "wait" in actions:
            return self._wait_payload(ctx, prefer="awaiting_information")

        # Should be unreachable (assembly always offers continue_activity + wait),
        # but stay safe by picking any offered action with schema-correct fields.
        any_action = sorted(actions)[0]
        if any_action == "wait":
            return self._wait_payload(ctx, prefer="awaiting_information")
        return self._move_payload(
            action=any_action, exit_id=None, reason="Fallback action.", ctx=ctx
        )

    # ------------------------------------------------------------------ #
    # Scoring
    # ------------------------------------------------------------------ #
    # Semantic tags a train/platform goal routes toward when the config supplies
    # no explicit ``prefer_exit_tags`` policy. Deliberately excludes
    # ``vertical_connector`` (which also tags up-escalators) so a boarder on the
    # platform is not treated as "advancing" when it ascends to the concourse.
    _DEFAULT_TRAIN_PREFER_TAGS = frozenset({"to_platform", "to_train"})

    @staticmethod
    def _filter_by_tags(
        options: list[ExitOption], include: frozenset[str] | set[str]
    ) -> list[ExitOption]:
        """Options carrying at least one of the ``include`` semantic tags."""
        inc = set(include)
        return [o for o in options if set(o.semantic_tags) & inc]

    @staticmethod
    def _apply_tag_preferences(
        options: list[ExitOption], prefer: set[str], avoid: set[str]
    ) -> list[ExitOption]:
        """Narrow ``options`` by goal policy: drop avoided tags, then keep preferred.

        Each narrowing is applied only when it leaves at least one option, so a
        policy never strands an agent with an empty candidate set.
        """
        pool = options
        if avoid:
            non_avoid = [o for o in pool if not (set(o.semantic_tags) & avoid)]
            if non_avoid:
                pool = non_avoid
        if prefer:
            preferred = [o for o in pool if set(o.semantic_tags) & prefer]
            if preferred:
                pool = preferred
        return pool

    def _pick_best_exit(self, options: list[ExitOption]) -> tuple[ExitOption, str]:
        """Return the best exit by route distance, visibility, crowding, and familiarity."""
        prox_raw = [
            (1.0 / (1.0 + (o.route_distance_m or o.distance_m)))
            if (o.route_distance_m or o.distance_m) is not None
            else 0.0
            for o in options
        ]
        busy_raw = [float(-o.crowd_count) for o in options]
        prox_n = self._minmax(prox_raw)
        busy_n = self._minmax(busy_raw)

        best_idx = 0
        best_score = float("-inf")
        scored_options: list[tuple[ExitOption, float]] = []
        for i, o in enumerate(options):
            fam = 1.0 if o.familiar else 0.0
            visible = 1.0 if o.visible else 0.0
            score = (
                self.w_proximity * prox_n[i]
                + self.w_visibility * visible
                + self.w_busyness * busy_n[i]
                + self.w_familiarity * fam
            )
            scored_options.append((o, score))
            # Strictly-greater keeps the first exit on ties → deterministic.
            if score > best_score:
                best_score = score
                best_idx = i

        best = options[best_idx]
        effective_distance = best.route_distance_m or best.distance_m
        dist_txt = (
            f"{effective_distance:.1f}m route"
            if effective_distance is not None
            else "unknown route distance"
        )
        candidate_summary = "; ".join(
            f"{option.exit_id}: score={score:.3f}, "
            f"route={option.route_distance_m or option.distance_m or float('nan'):.1f}m, "
            f"visible={'yes' if option.visible else 'no'}, crowd={option.crowd_count}, "
            f"familiar={'yes' if option.familiar else 'no'}"
            for option, score in scored_options
        )
        why = (
            f"Chose {best.display_name} ({dist_txt}, {best.crowd_count} nearby, "
            f"{'visible' if best.visible else 'not visible'}, "
            f"{'familiar' if best.familiar else 'unfamiliar'}) as the best weighted "
            f"trade-off of route distance, visibility, busyness, and familiarity. "
            f"Candidates: {candidate_summary}."
        )
        return best, why

    @staticmethod
    def _minmax(xs: list[float]) -> list[float]:
        """Min-max normalise to [0, 1]; all-equal collapses to a neutral 0.5."""
        if not xs:
            return []
        lo = min(xs)
        hi = max(xs)
        if hi - lo < 1e-9:
            return [0.5] * len(xs)
        span = hi - lo
        return [(x - lo) / span for x in xs]

    # ------------------------------------------------------------------ #
    # Payload builders (schema-correct per _validate_decision_payload)
    # ------------------------------------------------------------------ #
    def _move_payload(
        self,
        action: str,
        exit_id: str | None,
        reason: str,
        ctx: DecisionContext,
        pace: str | None = None,
    ) -> dict[str, Any]:
        return {
            "action": action,
            "wait_reason": None,
            "exit_id": exit_id,
            "pace": pace or self.pace,
            "reassess_when": "next_interval",
            "assessment": self._assessment(reason, ctx),
        }

    def _wait_payload(self, ctx: DecisionContext, prefer: str) -> dict[str, Any]:
        offered = ctx.offered_wait_reasons_set
        if prefer in offered:
            wait_reason = prefer
        elif offered:
            wait_reason = sorted(offered)[0]
        else:
            wait_reason = prefer  # no reasons offered; keep a stable value
        reason = (
            "Route appears blocked; waiting for it to clear."
            if prefer == "route_blocked"
            else "No action advances the goal right now; waiting for information."
        )
        return {
            "action": "wait",
            "wait_reason": wait_reason,
            "exit_id": None,
            "pace": None,
            "reassess_when": "next_interval",
            "assessment": self._assessment(reason, ctx),
        }

    @staticmethod
    def _assessment(reason: str, ctx: DecisionContext) -> dict[str, str]:
        return {
            "source_credibility": "Deterministic rule-based policy (no model call).",
            "situation_appraisal": f"Goal: {ctx.goal or 'unspecified'}.",
            "personal_relevance": "Acting to progress the assigned goal.",
            "options_considered": (
                "Offered exits: "
                + (", ".join(ctx.offered_exit_ids) if ctx.offered_exit_ids else "none")
                + "."
            ),
            "option_chosen_because": reason,
            "information_gap": "Rule engine does not reason beyond the provided signals.",
        }

    @staticmethod
    def _goal_wants_train(goal: str) -> bool:
        g = (goal or "").lower()
        return any(kw in g for kw in _TRAIN_GOAL_KEYWORDS)
