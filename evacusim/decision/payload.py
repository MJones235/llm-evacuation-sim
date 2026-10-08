"""The decision payload: what a decision engine returns for one agent.

Every engine, LLM or rule-based, answers with a payload of this shape::

    {
      "action":        one of the offered actions,
      "wait_reason":   an offered wait reason if action == "wait", else null,
      "exit_id":       an offered exit if action == "evacuate", else null,
      "pace":          "normal_pace" | "hurrying" | "running", null if waiting,
      "reassess_when": "next_interval" | "new_cue_only",
      "assessment":    {six free-text fields, see ASSESSMENT_FIELDS},
    }

What is *offered* changes per agent and per cycle (an exit can be blocked,
a train may have left), so validity is always checked against an
:class:`OfferedSet`. This module has no state and no simulation dependencies.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

PACES = ("normal_pace", "hurrying", "running")
REASSESS_MODES = ("next_interval", "new_cue_only")

ASSESSMENT_FIELDS = (
    "source_credibility",
    "situation_appraisal",
    "personal_relevance",
    "options_considered",
    "option_chosen_because",
    "information_gap",
)
"""The six stages of the Protective Action Decision Model, as free text."""


@dataclass(frozen=True)
class OfferedSet:
    """The choices open to one agent at one decision point, in display order."""

    actions: tuple[str, ...]
    wait_reasons: tuple[str, ...]
    exit_ids: tuple[str, ...]
    action_set: frozenset[str] = field(init=False)
    wait_reason_set: frozenset[str] = field(init=False)
    exit_id_set: frozenset[str] = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "action_set", frozenset(self.actions))
        object.__setattr__(self, "wait_reason_set", frozenset(self.wait_reasons))
        object.__setattr__(self, "exit_id_set", frozenset(self.exit_ids))


def validate(payload: dict[str, Any], offered: OfferedSet) -> list[str]:
    """Problems that make ``payload`` invalid for this offered set (empty if valid)."""
    errors: list[str] = []
    action = payload.get("action")
    wait_reason = payload.get("wait_reason")
    exit_id = payload.get("exit_id")
    pace = payload.get("pace")
    reassess_when = payload.get("reassess_when")
    assessment = payload.get("assessment")

    if action not in offered.action_set:
        errors.append(f"action must be one of {sorted(offered.action_set)}")

    if not isinstance(assessment, dict):
        errors.append("assessment must be an object")
    else:
        for key in ASSESSMENT_FIELDS:
            if key not in assessment or not isinstance(assessment.get(key), str):
                errors.append(f"assessment.{key} must be a string")

    if action == "wait":
        if wait_reason not in offered.wait_reason_set:
            errors.append(
                f"wait_reason must be one of {sorted(offered.wait_reason_set)} when action=wait"
            )
        if pace is not None:
            errors.append("pace must be null when action=wait")
    else:
        if wait_reason is not None:
            errors.append("wait_reason must be null unless action=wait")
        if pace not in PACES:
            errors.append(
                'pace must be one of "normal_pace", "hurrying", or "running" when action is not wait'
            )

    if action == "evacuate":
        if exit_id not in offered.exit_id_set:
            errors.append(
                f"exit_id must be one of {sorted(offered.exit_id_set)} when action=evacuate"
            )
    elif exit_id is not None:
        errors.append("exit_id must be null unless action=evacuate")

    if reassess_when not in REASSESS_MODES:
        errors.append("reassess_when must be 'next_interval' or 'new_cue_only'")

    return errors


def fallback(previous: dict[str, Any] | None, offered: OfferedSet) -> dict[str, Any]:
    """A safe payload when an engine produced nothing valid.

    Repeats the previous decision if it is still valid; otherwise continues
    the current activity (or waits, if that is all that is offered).
    """
    if isinstance(previous, dict) and not validate(previous, offered):
        return previous

    action = "continue_activity" if "continue_activity" in offered.action_set else "wait"
    wait_reason = None
    pace: str | None = "normal_pace"
    if action == "wait":
        wait_reason = (
            "awaiting_information"
            if "awaiting_information" in offered.wait_reason_set
            else sorted(offered.wait_reason_set)[0]
        )
        pace = None

    return {
        "assessment": {
            "source_credibility": "Unable to parse a valid response this turn.",
            "situation_appraisal": "No updated appraisal was available.",
            "personal_relevance": "No updated relevance estimate was available.",
            "options_considered": "Fallback policy applied.",
            "option_chosen_because": "Reused a safe default after repeated schema failures.",
            "information_gap": "No reliable model output was available.",
        },
        "action": action,
        "wait_reason": wait_reason,
        "exit_id": None,
        "pace": pace,
        "reassess_when": "next_interval",
    }


def extract_json_object(text: str) -> dict[str, Any] | None:
    """Decode the JSON object starting at the first ``{`` in ``text``, if any."""
    if not text:
        return None
    start = text.find("{")
    if start < 0:
        return None
    try:
        return json.loads(text[start:])
    except json.JSONDecodeError:
        return None


def to_json(payload: dict[str, Any]) -> str:
    """Compact, stable JSON for a payload."""
    return json.dumps(payload, ensure_ascii=True, separators=(",", ":"))


def summarize_assessment(assessment: Any) -> str:
    """One line from an assessment: the appraisal and the reason for the choice."""
    if isinstance(assessment, str):
        return assessment
    if isinstance(assessment, dict):
        parts = []
        for key in ("situation_appraisal", "option_chosen_because"):
            value = assessment.get(key, "")
            if value and isinstance(value, str):
                parts.append(value.strip().rstrip("."))
        return ". ".join(parts) if parts else ""
    return str(assessment) if assessment else ""
