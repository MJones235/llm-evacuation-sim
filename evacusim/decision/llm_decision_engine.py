"""Concordia/LLM-backed decision engine.

This is the default :class:`~evacusim.core.decision_engine.DecisionEngine`.  It
delegates to :meth:`DecisionProcessor._llm_produce_decision`, which owns the
prompt cache, the per-cycle asyncio concurrency primitives, the telemetry
counters, and the ``agent.act`` retry/repair/fallback loop.  Keeping that state
on the processor (rather than migrating it here) lets the extraction preserve
the existing behaviour exactly while still establishing a clean, swappable seam:
selecting a different engine simply bypasses this class — and, with it, every
LLM call — entirely.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from evacusim.core.decision_engine import DecisionContext, DecisionResult

if TYPE_CHECKING:  # pragma: no cover - typing only
    from evacusim.decision.decision_processor import DecisionProcessor


class LLMDecisionEngine:
    """Default engine: produce a decision via the Concordia/LLM pipeline."""

    def __init__(self, processor: "DecisionProcessor") -> None:
        self._processor = processor

    async def decide(self, ctx: DecisionContext) -> DecisionResult:
        return await self._processor._llm_produce_decision(ctx)
