"""The decision engine interface: the seam between simulation and cognition.

Each decision cycle, for every deciding agent::

    SituationAssembler ──DecisionContext──▶ DecisionEngine ──DecisionResult──▶ execution
    (evacusim.decision.situation)            (pluggable)        (payload)       (translator,
                                                                                 executor)

Both engines see the same :class:`DecisionContext` and return the same kind of
:class:`DecisionResult`, whose ``payload`` follows
:mod:`evacusim.decision.payload`. Two engines are provided:

- :class:`~evacusim.decision.llm_decision_engine.LLMDecisionEngine` renders the
  context into a prompt and asks a language model (via Concordia agents).
- :class:`~evacusim.decision.rule_based_decision_engine.RuleBasedDecisionEngine`
  applies deterministic rules; no language model is used.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ExitOption:
    """A single evacuation/route exit offered to an agent this cycle.

    Carries the structured routing signals a rule-based engine needs.  The LLM
    engine ignores these numeric fields (it reasons over the rendered prompt),
    but they are cheap to compute during assembly and useful for telemetry.
    """

    exit_id: str
    display_name: str
    distance_m: float | None = None
    route_distance_m: float | None = None
    crowd_count: int = 0
    familiar: bool = False
    visible: bool = False
    semantic_tags: tuple[str, ...] = ()


@dataclass
class DecisionContext:
    """Engine-neutral snapshot of one agent's decision situation.

    Built by :class:`~evacusim.decision.situation.SituationAssembler` and
    handed to :meth:`DecisionEngine.decide`. Engines must choose from the
    ``offered_*`` lists.
    """

    agent_id: str
    position: tuple[float, float]
    zone_id: str | None
    goal: str
    observation: str
    agent_cfg: dict[str, Any]

    offered_actions: list[str]
    offered_wait_reasons: list[str]
    offered_exit_ids: list[str]
    exit_options: dict[str, ExitOption]

    route_blocked: bool
    cues: list[str]
    current_sim_time: float

    # The offered_* lists as sets, for fast membership tests.
    offered_actions_set: set[str] = field(default_factory=set)
    offered_wait_reasons_set: set[str] = field(default_factory=set)
    offered_exit_ids_set: set[str] = field(default_factory=set)

    # Goal-directed routing hints, resolved from ``goal_semantic_policies`` for
    # this agent's (goal, zone).  ``prefer_exit_tags`` are semantic tags the
    # agent should route toward (e.g. ``to_platform`` for a boarder); it should
    # advance only via a preferred exit and otherwise wait, rather than leave via
    # a non-preferred one.  ``avoid_exit_tags`` are tags to route away from
    # (e.g. ``to_platform`` for someone leaving the station).  Empty when no
    # policy matches.  Structured-engine only; the LLM engine ignores them (the
    # same guidance reaches the LLM through the prompt text).
    prefer_exit_tags: tuple[str, ...] = ()
    avoid_exit_tags: tuple[str, ...] = ()

    # Specific exit ids the agent should route toward first — ahead of the
    # generic ``prefer_exit_tags`` — because they lead to this agent's concrete
    # destination.  For a boarder this is the down-escalator serving its target
    # platform (e.g. a ``train_platform_3`` boarder gets ``escalator_a_down``),
    # resolved from ``platform_down_exits`` in config.  Only ids actually offered
    # this cycle are surfaced, so a blocked/unavailable connector falls back to
    # the tag-based choice.  Structured-engine only; the LLM engine routes to a
    # target platform through ``action_translator`` instead.
    preferred_exit_ids: tuple[str, ...] = ()

    # Exit selected on an earlier cycle. Engines may retain it while it remains
    # in the policy-filtered candidate set, avoiding route oscillation caused by
    # small crowd-count changes. A blocked or disallowed exit is not offered and
    # therefore cannot be retained.
    committed_exit_id: str | None = None

    # True while the agent is walking to ``committed_exit_id``.
    is_moving: bool = False

    # The ``station.goal_semantic_policies`` entry matching this goal and zone
    # (its tags are also in prefer/avoid_exit_tags); the LLM prompt quotes its
    # instruction.
    goal_policy: dict[str, Any] | None = None

    # Every warning cue the agent has perceived, oldest first:
    # ``{time, source, strength, instruction}``, source alarm / pa / staff.
    # Read by the rule-based engine; the LLM reads the observation text.
    warnings: tuple[dict[str, Any], ...] = ()

    # Agents within the rule engine's social-cue radius.
    nearby_agent_ids: tuple[str, ...] = ()


@dataclass
class DecisionResult:
    """What an engine returns for one agent.

    Attributes:
        payload: The validated decision payload (see
            :mod:`evacusim.decision.payload`), or ``None`` if the engine
            produced nothing usable (the agent keeps its current waypoint).
        action_json: The payload as the JSON text the engine produced, if any
            (the LLM's own response, possibly with surrounding text); ``None``
            means "serialize ``payload``".
        prompt: The prompt the engine sent or would have sent (LLM engine only).
        llm_was_called: True if a real LLM request was issued (telemetry).
        repair_status: One of ``ok`` / ``repair_1`` / ``repair_2`` /
            ``fallback`` / ``cached`` — mirrors existing decision telemetry.
        skip_downstream: When True the engine has determined there is nothing to
            translate/execute this cycle (e.g. an LLM cache hit for an agent that
            is already moving to its committed destination).
        stage: The agent's response stage, for engines that model one
            (rule-based: ``unaware``, ``aware``, ``evacuating``).
    """

    payload: dict[str, Any] | None
    action_json: str | None = None
    prompt: str | None = None
    llm_was_called: bool = False
    repair_status: str = "ok"
    skip_downstream: bool = False
    stage: str | None = None


class DecisionEngine:
    """Pluggable cognition: turns a :class:`DecisionContext` into a decision.

    Subclasses implement :meth:`decide`. The other methods are optional hooks
    with no-op defaults.
    """

    async def decide(self, ctx: DecisionContext) -> DecisionResult:
        """Choose an action for one agent from the context's offered set."""
        raise NotImplementedError

    def reset_agent(self, agent_id: str) -> None:
        """Forget anything that would make the agent's next decision a repeat.

        Called when an agent's situation changes discontinuously (it changed
        level, or its route must be re-planned).
        """

    def on_agent_exit(self, agent_id: str) -> None:
        """Release per-agent state when an agent leaves the simulation."""

    def statistics(self) -> dict[str, Any]:
        """Engine-specific counters for the end-of-run report."""
        return {}
