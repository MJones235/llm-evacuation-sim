"""LLM decision engine: agents decide by prompting a language model.

For each agent, :meth:`LLMDecisionEngine.decide`:

1. renders the context into a prompt (:class:`~evacusim.decision.llm_prompt.DecisionPromptBuilder`);
2. reuses the agent's previous decision instead of calling the model when
   nothing significant has changed (prompt cache), or when it decided very
   recently and the observation reports nothing new (re-decision throttle);
3. otherwise passes the observation to the agent's Concordia entity and asks
   it to act, up to three times: a response that is not valid JSON or does
   not fit the offered set is re-asked with the errors appended;
4. falls back to a safe payload (:func:`evacusim.decision.payload.fallback`)
   if all three attempts fail.

Calls run concurrently, at most ``max_parallel`` at a time, each limited to
``timeout_s`` seconds of wall-clock time.
"""

from __future__ import annotations

import asyncio
from typing import Any

from concordia.typing import entity as entity_lib

from evacusim.concordia.azure_llm_concordia import llm_current_agent_id, llm_current_sim_time
from evacusim.core.decision_engine import DecisionContext, DecisionEngine, DecisionResult
from evacusim.decision import payload as payload_lib
from evacusim.decision.llm_prompt import DecisionPromptBuilder
from evacusim.decision.payload import OfferedSet
from evacusim.decision.prompt_cache import PromptCache
from evacusim.utils.logger import get_logger

logger = get_logger(__name__)

MAX_ATTEMPTS = 3


class LLMDecisionEngine(DecisionEngine):
    """Chooses actions by prompting each agent's Concordia entity.

    Args:
        agents: Concordia entities, keyed by agent id (live: runtime-spawned
            agents are added by the runner).
        prompt_builder: Renders contexts into prompts.
        message_system: Source of messages each agent has received (an input
            to the prompt cache).
        agent_decisions: Every agent's decision history (read only), for the
            fallback's "repeat the previous decision".
        perf_timer: Records time spent observing and calling the model.
        max_parallel: Concurrent model calls.
        timeout_s: Wall-clock limit per call (``None``: no limit). On timeout
            the agent keeps its current waypoint.
        min_redecision_interval_s: Reuse a decision made less than this many
            simulated seconds ago when nothing new is observed.
    """

    def __init__(
        self,
        agents: dict[str, entity_lib.Entity],
        prompt_builder: DecisionPromptBuilder,
        message_system,
        agent_decisions: dict[str, dict[str, Any]],
        perf_timer,
        max_parallel: int = 10,
        timeout_s: float | None = 30.0,
        min_redecision_interval_s: float = 0.0,
    ) -> None:
        self._agents = agents
        self._prompts = prompt_builder
        self._messages = message_system
        self._history = agent_decisions
        self._perf_timer = perf_timer
        self._max_parallel = max(1, int(max_parallel))
        self._timeout_s = timeout_s
        self._min_redecision_interval_s = max(0.0, float(min_redecision_interval_s))

        self.prompt_cache = PromptCache(enable_detailed_logging=True)
        self._last_call_time: dict[str, float] = {}
        self.llm_calls_made = 0
        self.llm_calls_skipped = 0

        # asyncio primitives bind to the event loop that creates them, and
        # each decision cycle runs in a fresh loop, so this is created per loop.
        self._semaphore: asyncio.Semaphore | None = None
        self._semaphore_loop: asyncio.AbstractEventLoop | None = None

    # -- DecisionEngine hooks -------------------------------------------------

    def reset_agent(self, agent_id: str) -> None:
        self.prompt_cache.clear_agent(agent_id)

    def on_agent_exit(self, agent_id: str) -> None:
        self.prompt_cache.clear_agent(agent_id)

    def statistics(self) -> dict[str, Any]:
        total = self.llm_calls_made + self.llm_calls_skipped
        return {
            "llm_calls_made": self.llm_calls_made,
            "llm_calls_skipped": self.llm_calls_skipped,
            "total_decision_cycles": total,
            "skip_rate_percent": (self.llm_calls_skipped / total * 100) if total else 0,
            "cache_stats": self.prompt_cache.get_statistics(),
        }

    # -- deciding ---------------------------------------------------------------

    async def decide(self, ctx: DecisionContext) -> DecisionResult:
        agent_id = ctx.agent_id
        offered = OfferedSet(
            tuple(ctx.offered_actions), tuple(ctx.offered_wait_reasons), tuple(ctx.offered_exit_ids)
        )
        prompt = self._prompts.render(ctx)

        reused = self._reuse_previous(ctx, prompt, offered)
        if reused is not None:
            return reused

        payload, repair_status = await self._ask_model(ctx, prompt, offered)
        if payload is None:  # timed out: keep the current waypoint
            return DecisionResult(payload=None, skip_downstream=True)

        action_json = payload_lib.to_json(payload)
        self.prompt_cache.cache_decision(agent_id, action_json)
        self._last_call_time[agent_id] = ctx.current_sim_time
        self.llm_calls_made += 1
        return DecisionResult(
            payload=payload,
            action_json=action_json,
            prompt=prompt,
            llm_was_called=True,
            repair_status=repair_status,
        )

    def _reuse_previous(
        self, ctx: DecisionContext, prompt: str, offered: OfferedSet
    ) -> DecisionResult | None:
        """The agent's previous decision, if it should be reused rather than re-asked."""
        agent_id = ctx.agent_id
        try:
            received = self._messages.get_received_messages(agent_id)
            messages = (
                [m.get("message", str(m)) if isinstance(m, dict) else m for m in received]
                if received
                else None
            )
        except Exception as e:
            logger.debug(f"{agent_id}: Could not get messages: {e}")
            messages = None

        should_call, cached = self.prompt_cache.should_call_llm(
            agent_id=agent_id,
            observation=ctx.observation,
            action_spec_text=prompt,
            received_messages=messages,
        )

        if not should_call and cached:
            candidate = payload_lib.extract_json_object(cached)
            if candidate is not None and not payload_lib.validate(candidate, offered):
                logger.info(
                    f"{agent_id}: ✓ Prompt unchanged, reusing cached decision (saved LLM call)"
                )
                self.llm_calls_skipped += 1
                if ctx.committed_exit_id and ctx.is_moving:
                    logger.debug(
                        f"{agent_id}: ✓ Short-circuit — already moving to "
                        f"'{ctx.committed_exit_id}', skipping translation & execution"
                    )
                    return DecisionResult(payload=None, skip_downstream=True)
                return DecisionResult(
                    payload=candidate, action_json=cached, prompt=prompt, repair_status="cached"
                )
            logger.info(
                f"{agent_id}: cached decision invalid for current offered set — requesting fresh decision"
            )

        # Throttle: nothing new was observed and the agent decided recently.
        if (
            self._min_redecision_interval_s > 0
            and "No significant new information." in ctx.observation
        ):
            previous = self._last_call_time.get(agent_id)
            if previous is not None:
                elapsed = ctx.current_sim_time - previous
                if elapsed < self._min_redecision_interval_s:
                    throttled = self.prompt_cache.get_cached_decision(agent_id)
                    candidate = payload_lib.extract_json_object(throttled) if throttled else None
                    if candidate is not None and not payload_lib.validate(candidate, offered):
                        logger.info(
                            f"{agent_id}: ✓ Reusing cached decision (redecision throttle "
                            f"{elapsed:.1f}s < {self._min_redecision_interval_s:.1f}s)"
                        )
                        self.llm_calls_skipped += 1
                        return DecisionResult(
                            payload=candidate,
                            action_json=throttled,
                            prompt=prompt,
                            repair_status="cached",
                        )
        return None

    async def _ask_model(
        self, ctx: DecisionContext, prompt: str, offered: OfferedSet
    ) -> tuple[dict[str, Any] | None, str]:
        """Ask the agent's entity to act, repairing invalid answers.

        Returns ``(payload, repair_status)``; ``(None, "timeout")`` on timeout.
        """
        agent_id = ctx.agent_id
        agent = self._agents[agent_id]
        with self._perf_timer.measure("agent_observe", is_parallel=True):
            agent.observe(ctx.observation)

        errors = ["invalid response"]
        for attempt in range(MAX_ATTEMPTS):
            attempt_prompt = prompt
            if attempt > 0:
                attempt_prompt = (
                    prompt
                    + "\n\nSYSTEM NOTE: Your previous output violated the schema: "
                    + "; ".join(errors)
                    + ". Return only valid JSON that satisfies all rules."
                )
            spec = entity_lib.ActionSpec(
                call_to_action=attempt_prompt, output_type=entity_lib.OutputType.FREE
            )
            async with self._loop_semaphore():
                try:
                    with self._perf_timer.measure("agent_act_llm", is_parallel=True):
                        llm_current_agent_id.set(agent_id)
                        llm_current_sim_time.set(ctx.current_sim_time)
                        if self._timeout_s is not None:
                            async with asyncio.timeout(self._timeout_s):
                                response = await asyncio.to_thread(agent.act, spec)
                        else:
                            response = await asyncio.to_thread(agent.act, spec)
                except TimeoutError:
                    logger.warning(
                        f"{agent_id}: decision timed out after {self._timeout_s:.0f}s — "
                        "keeping existing waypoint"
                    )
                    return None, "timeout"

            candidate = payload_lib.extract_json_object(response)
            if candidate is None:
                errors = ["response was not valid JSON"]
                continue
            errors = payload_lib.validate(candidate, offered)
            if not errors:
                return candidate, ("ok", "repair_1", "repair_2")[attempt]

        previous = self._history.get(agent_id, {}).get("decisions", [])
        last_payload = previous[-1].get("decision_payload") if previous else None
        return payload_lib.fallback(last_payload, offered), "fallback"

    def _loop_semaphore(self) -> asyncio.Semaphore:
        loop = asyncio.get_running_loop()
        if self._semaphore is None or self._semaphore_loop is not loop:
            self._semaphore = asyncio.Semaphore(self._max_parallel)
            self._semaphore_loop = loop
        return self._semaphore
