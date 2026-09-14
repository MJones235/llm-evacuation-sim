"""Engine-neutral decision interface.

This module defines the seam between the simulation's *situation assembly* /
*action execution* pipeline (owned by
:class:`evacusim.decision.decision_processor.DecisionProcessor`) and the
pluggable *cognition* that chooses an action for an agent.

Two implementations are provided:

- :class:`evacusim.decision.llm_decision_engine.LLMDecisionEngine` — the default
  Concordia/LLM-backed engine (prompt rendering, prompt cache, ``agent.act``
  retry/repair/fallback).
- :class:`evacusim.decision.rule_based_decision_engine.RuleBasedDecisionEngine`
  — a deterministic, LLM-free engine that routes by weighting proximity,
  busyness, and familiarity and handles blocked exits.

The contract between an engine and the rest of the pipeline is the
``decision_payload`` dict validated by
``DecisionProcessor._validate_decision_payload``::

    {action, wait_reason, exit_id, pace, reassess_when, assessment}

Everything downstream of the engine (action translation, JuPedSim execution,
telemetry) is engine-agnostic and consumes only that payload.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


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
    crowd_count: int = 0
    familiar: bool = False
    semantic_tags: tuple[str, ...] = ()


@dataclass
class DecisionContext:
    """Engine-neutral snapshot of one agent's decision situation.

    Built by the assembly phase of ``DecisionProcessor._process_single_agent``
    and handed to :meth:`DecisionEngine.decide`.  Structured fields drive the
    rule-based engine; the ``prompt_text`` / ``received_messages`` fields are
    LLM-only extras that the rule-based engine ignores.
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

    # Validation sets (mirror the offered_* lists; precomputed for the engine
    # and for _validate_decision_payload).
    offered_actions_set: set[str] = field(default_factory=set)
    offered_wait_reasons_set: set[str] = field(default_factory=set)
    offered_exit_ids_set: set[str] = field(default_factory=set)

    # LLM-only extras (ignored by non-LLM engines).
    prompt_text: str | None = None
    received_messages: list[str] | None = None


@dataclass
class DecisionResult:
    """What an engine returns for one agent.

    Attributes:
        payload: The validated ``decision_payload`` dict, or ``None`` if the
            engine produced nothing usable (the orchestrator then skips the
            agent, keeping their current waypoint).
        action_json: Canonical JSON string form of ``payload`` (the LLM engine
            already has this; other engines may leave it ``None`` and let the
            orchestrator serialize).
        llm_was_called: True if a real LLM request was issued (telemetry).
        repair_status: One of ``ok`` / ``repair_1`` / ``repair_2`` /
            ``fallback`` / ``cached`` — mirrors existing decision telemetry.
        skip_downstream: When True the engine has determined there is nothing to
            translate/execute this cycle (e.g. an LLM cache hit for an agent that
            is already moving to its committed destination).
    """

    payload: dict[str, Any] | None
    action_json: str | None = None
    llm_was_called: bool = False
    repair_status: str = "ok"
    skip_downstream: bool = False


@runtime_checkable
class DecisionEngine(Protocol):
    """Pluggable cognition that turns a :class:`DecisionContext` into a decision.

    Implementations must return a :class:`DecisionResult` whose ``payload``, when
    not ``None``, satisfies ``DecisionProcessor._validate_decision_payload``
    against the context's offered sets.
    """

    async def decide(self, ctx: DecisionContext) -> DecisionResult:
        ...
