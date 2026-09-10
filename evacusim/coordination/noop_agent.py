"""Lightweight stand-in agent for LLM-free runs.

When a non-LLM :class:`~evacusim.core.decision_engine.DecisionEngine` is
selected, the simulation does not build Concordia entities (which require a
language model and a sentence embedder).  The rest of the pipeline still keeps a
registry of ``agent_id -> agent`` and occasionally pushes broadcast text at each
agent via ``observe`` (PA announcements, event broadcasts).  :class:`NoOpAgent`
satisfies exactly that surface — ``observe`` is a no-op — while making it a hard
error if anything tries to invoke cognition (``act``) on it, which would mean an
LLM-free run had accidentally reached the model path.
"""

from __future__ import annotations

from typing import Any


class NoOpAgent:
    """A registry placeholder with a no-op ``observe`` and a guarded ``act``."""

    __slots__ = ("agent_id", "name")

    def __init__(self, agent_id: str, name: str | None = None) -> None:
        self.agent_id = agent_id
        self.name = name or agent_id

    def observe(self, observation: Any) -> None:
        """Discard the observation — no memory bank exists in LLM-free mode."""
        return None

    def act(self, action_spec: Any = None) -> str:
        raise RuntimeError(
            f"NoOpAgent.act called for '{self.agent_id}': a non-LLM engine must "
            "never route decisions through Concordia. This indicates the LLM "
            "decision path was reached during an LLM-free run."
        )
