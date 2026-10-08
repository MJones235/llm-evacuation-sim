"""Render a :class:`DecisionContext` into the LLM decision prompt.

The prompt is a ``string.Template`` (default:
``templates/decision_prompt.template.txt``; override with
``prompts.decision_prompt_template_path``) with six blocks: role, journey,
what is new since the last decision, current surroundings, the offered
actions, and the JSON response format. This module fills its placeholders.
Only the LLM engine uses it.
"""

from __future__ import annotations

import re
from pathlib import Path
from string import Template
from typing import Any

from evacusim.core.decision_engine import DecisionContext
from evacusim.utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_TEMPLATE = Path(__file__).with_name("templates") / "decision_prompt.template.txt"

_RE_CIRCULAR_FOLLOW = re.compile(
    r"You are following (Person (\w+)), and Person \2 is following YOU"
)
_RE_FOLLOWER_SINGULAR = re.compile(r"⚠️ (Person (\w+)) is trying to follow YOU")
_RE_FOLLOWER_PLURAL = re.compile(r"⚠️ (.+?) are trying to follow YOU")


def load_template(template_path: str | None) -> Template:
    """Read the decision prompt template.

    Raises:
        FileNotFoundError: The configured template file does not exist.
        RuntimeError: The template file cannot be read.
    """
    chosen_path = Path(template_path).expanduser() if template_path else DEFAULT_TEMPLATE
    if not chosen_path.exists():
        raise FileNotFoundError(f"Decision prompt template file not found: {chosen_path}")
    try:
        text = chosen_path.read_text(encoding="utf-8")
    except Exception as e:
        raise RuntimeError(
            f"Failed to read decision prompt template from {chosen_path}: {e}"
        ) from e
    logger.info(f"Loaded decision prompt template from {chosen_path}")
    return Template(text)


class DecisionPromptBuilder:
    """Builds one agent's decision prompt.

    Args:
        exit_registry: Display names of exits.
        exit_semantic_tags: ``station.exit_semantic_tags`` (where each exit leads).
        agent_decisions: Every agent's decision history (read only), for the
            "previous choice" summary.
        template_path: Prompt template override; ``None`` for the default.
    """

    def __init__(
        self,
        exit_registry,
        exit_semantic_tags: dict[str, list[str]],
        agent_decisions: dict[str, dict[str, Any]],
        template_path: str | None = None,
    ) -> None:
        self._registry = exit_registry
        self._exit_semantic_tags = exit_semantic_tags
        self._history = agent_decisions
        self._template = load_template(template_path)

    def render(self, ctx: DecisionContext) -> str:
        """The full prompt for ``ctx``."""
        cfg = ctx.agent_cfg
        offered_actions = ctx.offered_actions
        wait_action_rule, pace_field, pace_validation = pace_blocks(offered_actions)
        role_extra = cfg.get("decision_prompt_extra", "")
        cues = ", ".join(ctx.cues) if ctx.cues else "none"
        return self._template.safe_substitute(
            age=str(cfg.get("age", "unknown")),
            gender=str(cfg.get("gender", "person")),
            personality_profile=str(
                cfg.get("personality_anchor") or cfg.get("personality_type", "unknown")
            ),
            journey_block=ctx.goal if ctx.goal else "Continue your assigned journey.",
            new_since_last_decision=(
                f"{self._previous_decision_summary(ctx.agent_id, ctx.current_sim_time)} "
                f"New information: {new_information(ctx.observation)} "
                f"Detected cues: {cues}."
            ),
            current_surroundings=ctx.observation,
            available_actions_block=available_actions_block(
                offered_actions, ctx.offered_wait_reasons, ctx.offered_exit_ids
            ),
            action_enum_block=" | ".join(offered_actions) if offered_actions else "wait",
            wait_action_rule_block=wait_action_rule,
            pace_field_block=pace_field,
            pace_validation_block=pace_validation,
            situation_framing=f"{role_extra}\n\n" if role_extra else "",
            following_constraint_text=following_constraints(ctx.observation),
            valid_exits_text=self._exits_block(ctx.offered_exit_ids, ctx.goal_policy),
        )

    def _exits_block(self, offered_exit_ids: list[str], goal_policy: dict | None) -> str:
        """The usable exits, by id and display name, plus goal-related exit guidance."""
        semantics = self._exit_semantics(goal_policy, offered_exit_ids)
        if not offered_exit_ids:
            return (
                "Route availability from your current position: no usable evacuation exits right now.\n"
                + semantics
                + "No evacuation exits are currently available from where you are.\n"
            )
        display = self._registry.get_display_name
        bullets = "\n".join(f"- {eid}: {display(eid)}" for eid in offered_exit_ids)
        route_names = ", ".join(display(eid) for eid in offered_exit_ids)
        return (
            f"Route availability from your current position: usable exits now are {route_names}.\n"
            + "Evacuation exits you can choose now (use exit_id exactly):\n"
            + bullets
            + "\n"
            + semantics
        )

    def _exit_semantics(self, goal_policy: dict | None, offered_exit_ids: list[str]) -> str:
        """The goal policy's instruction and which offered exits suit the goal."""
        if not goal_policy:
            return "\n"
        offered = set(offered_exit_ids)
        prefer = {
            str(t).strip().lower()
            for t in goal_policy.get("prefer_exit_tags", [])
            if str(t).strip()
        }
        avoid = {
            str(t).strip().lower() for t in goal_policy.get("avoid_exit_tags", []) if str(t).strip()
        }

        preferred_names: list[str] = []
        avoided_names: list[str] = []
        for eid, tags in self._exit_semantic_tags.items():
            if eid not in offered:
                continue
            tag_set = {str(tag).strip().lower() for tag in (tags or []) if str(tag).strip()}
            if prefer and (tag_set & prefer):
                preferred_names.append(self._registry.get_display_name(eid))
            if avoid and (tag_set & avoid):
                avoided_names.append(self._registry.get_display_name(eid))

        lines = ["\nExit context:"]
        instruction = str(goal_policy.get("instruction", "")).strip()
        if instruction:
            lines.append(f"- {instruction}")
        if preferred_names:
            shown = ", ".join(sorted(set(preferred_names))[:8])
            lines.append(f"- Exits associated with this goal context: {shown}")
        if avoided_names:
            shown = ", ".join(sorted(set(avoided_names))[:8])
            lines.append(f"- Other available exits in a different context: {shown}")
        if len(lines) == 1:
            return ""
        return "\n" + "\n".join(lines) + "\n"

    def _previous_decision_summary(self, agent_id: str, current_sim_time: float) -> str:
        """When the agent last decided, and what (action and target only)."""
        decisions = self._history.get(agent_id, {}).get("decisions", [])
        if not decisions:
            return "First decision point for you."
        last = decisions[-1]
        seconds_ago = max(0.0, current_sim_time - float(last.get("time", current_sim_time)))
        payload = last.get("decision_payload", {}) if isinstance(last, dict) else {}
        action = str(payload.get("action", "continue_activity"))
        target = ""
        if action == "wait" and payload.get("wait_reason"):
            target = f" (reason: {payload.get('wait_reason')})"
        elif action == "evacuate" and payload.get("exit_id"):
            target = f" (exit_id: {payload.get('exit_id')})"
        return f"{seconds_ago:.0f}s ago your previous choice was action='{action}'{target}."


def available_actions_block(
    offered_actions: list[str], offered_wait_reasons: list[str], offered_exit_ids: list[str]
) -> str:
    """The offered actions, each with the field it requires."""
    lines = ["Choose exactly one action from this offered set:"]
    for verb in offered_actions:
        if verb == "wait":
            lines.append(f"- wait (required: wait_reason in {{{', '.join(offered_wait_reasons)}}})")
        elif verb == "evacuate":
            exits = ", ".join(offered_exit_ids) if offered_exit_ids else "none"
            lines.append(f"- evacuate (required: exit_id from {{{exits}}})")
        else:
            lines.append(f"- {verb} (no extra field)")
    return "\n".join(lines)


def pace_blocks(offered_actions: list[str]) -> tuple[str, str, str]:
    """The wait rule, the pace field and the pace rules (omitted if only waiting is offered)."""
    if set(offered_actions) == {"wait"}:
        return "- if action = wait: set wait_reason.", "", ""
    return (
        "- if action = wait: set wait_reason and set pace to null.",
        '  "pace": null,\n',
        "- pace definitions: normal_pace - moving as you normally would; "
        "hurrying - walking noticeably faster than usual, but not running; "
        "running - running.\n"
        '- if action = "wait": pace must be null.\n'
        '- if action is not "wait": pace must be one of "normal_pace", "hurrying", or "running".',
    )


def new_information(observation: str) -> str:
    """The line following "What is NEW since your last decision:" in the observation."""
    lines = [ln.strip() for ln in observation.splitlines() if ln.strip()]
    for idx, line in enumerate(lines):
        if line.startswith("What is NEW since your last decision:"):
            return lines[idx + 1] if idx + 1 < len(lines) else "No significant new information."
    return "No significant new information."


def following_constraints(observation: str) -> str:
    """A movement constraint when the agent is in a following loop or being followed."""
    constraints = []
    circular = _RE_CIRCULAR_FOLLOW.search(observation)
    if circular:
        person_label = circular.group(1)
        agent_id_str = f"agent_{circular.group(2)}"
        constraints.append(
            f"\n⛔ CIRCULAR FOLLOWING — BREAK THE LOOP: You and {person_label} are "
            f"following each other. Neither of you is heading toward an exit. "
            f"You MUST do something different this turn. "
            f"Do NOT set target_agent='{agent_id_str}'. "
            f"Choose one of: (a) target_type='exit' if an exit is listed in VALID EXIT OPTIONS, "
            f"(b) target_type='agent' toward a DIFFERENT person who is heading toward an exit, "
            f"or (c) target_type='current_position' to stop and reorient."
        )
    else:
        followers = _RE_FOLLOWER_SINGULAR.findall(observation)
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
                f"\n⚠️ FOLLOWER RULE: {names} is following YOU — you are their leader. "
                f"Do NOT set target_agent={avoid}. "
                f"You must lead: choose target_type='exit' using your own knowledge, "
                f"or follow a DIFFERENT person who is actually heading toward an exit."
            )
    if constraints:
        return "\n═══ MOVEMENT CONSTRAINTS (READ FIRST) ═══" + "".join(constraints) + "\n\n"
    return ""
