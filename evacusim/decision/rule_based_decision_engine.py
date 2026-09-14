"""Deterministic, LLM-free decision engine.

Implements the :class:`~evacusim.core.decision_engine.DecisionEngine` protocol
without any model call.  Each agent has a goal — reach a train/platform or leave
the station — and picks a route by scoring the offered exits on a weighted
combination of three signals carried on each
:class:`~evacusim.core.decision_engine.ExitOption`:

* **proximity** — closer exits score higher (inverse straight-line distance),
* **busyness** — less crowded exits score higher,
* **familiarity** — exits the agent already knows score higher.

Blocked exits never reach the engine: the assembly phase filters them out of
``offered_exit_ids`` before building the context, and ``route_blocked`` is
surfaced so the engine can wait with the right reason when nothing is usable.

The engine is pure and deterministic (no randomness), so a run is fully
reproducible and requires no LLM, embedder, or network.
"""

from __future__ import annotations

from typing import Any

from evacusim.core.decision_engine import DecisionContext, DecisionResult, ExitOption

# Non-wait actions must declare a pace from this set (see
# DecisionProcessor._validate_decision_payload).
_DEFAULT_PACE = "normal_pace"

# Goal keywords that indicate the agent is trying to board a train / reach a
# platform rather than leave the station.
_TRAIN_GOAL_KEYWORDS = ("train", "platform", "board")


class RuleBasedDecisionEngine:
    """Route by weighting proximity, busyness, and familiarity."""

    def __init__(
        self,
        w_proximity: float = 0.5,
        w_busyness: float = 0.3,
        w_familiarity: float = 0.2,
        crowd_radius_m: float = 5.0,
        pace: str = _DEFAULT_PACE,
    ) -> None:
        self.w_proximity = float(w_proximity)
        self.w_busyness = float(w_busyness)
        self.w_familiarity = float(w_familiarity)
        self.crowd_radius_m = float(crowd_radius_m)
        self.pace = pace

    # ------------------------------------------------------------------ #
    # DecisionEngine protocol
    # ------------------------------------------------------------------ #
    async def decide(self, ctx: DecisionContext) -> DecisionResult:
        payload = self._decide_payload(ctx)
        return DecisionResult(
            payload=payload,
            action_json=None,  # orchestrator serialises via _decision_payload_to_json
            llm_was_called=False,
            repair_status="rule_based",
        )

    # ------------------------------------------------------------------ #
    # Core policy
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
            include = (prefer | self._DEFAULT_TRAIN_PREFER_TAGS) if prefer else self._DEFAULT_TRAIN_PREFER_TAGS
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
        options: list[ExitOption], include: "frozenset[str] | set[str]"
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
        """Return (best_exit, explanation) by weighted proximity/busyness/familiarity."""
        prox_raw = [
            (1.0 / (1.0 + o.distance_m)) if o.distance_m is not None else 0.0
            for o in options
        ]
        busy_raw = [float(-o.crowd_count) for o in options]
        prox_n = self._minmax(prox_raw)
        busy_n = self._minmax(busy_raw)

        best_idx = 0
        best_score = float("-inf")
        for i, o in enumerate(options):
            fam = 1.0 if o.familiar else 0.0
            score = (
                self.w_proximity * prox_n[i]
                + self.w_busyness * busy_n[i]
                + self.w_familiarity * fam
            )
            # Strictly-greater keeps the first exit on ties → deterministic.
            if score > best_score:
                best_score = score
                best_idx = i

        best = options[best_idx]
        dist_txt = f"{best.distance_m:.1f}m" if best.distance_m is not None else "unknown distance"
        why = (
            f"Chose {best.display_name} ({dist_txt}, {best.crowd_count} nearby, "
            f"{'familiar' if best.familiar else 'unfamiliar'}) as the best weighted "
            f"trade-off of proximity, busyness, and familiarity."
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
        self, action: str, exit_id: str | None, reason: str, ctx: DecisionContext
    ) -> dict[str, Any]:
        return {
            "action": action,
            "wait_reason": None,
            "exit_id": exit_id,
            "pace": self.pace,
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
