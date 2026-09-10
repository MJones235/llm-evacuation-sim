"""Characterization tests for the pluggable decision-engine seam (Feature B, B1-B3).

These lock two things after the DecisionProcessor refactor:
  1. The extracted LLM production path (``_llm_produce_decision``) still turns a
     valid model response into a validated decision_payload with the expected
     telemetry (llm_was_called / repair_status), i.e. behaviour is unchanged.
  2. A custom engine can be injected and is what the processor delegates to,
     and the default engine is the LLM engine — so a run can be made LLM-free
     purely by injecting a different engine.
"""

import asyncio
import contextlib
import json
import unittest

from evacusim.core.decision_engine import (
    DecisionContext,
    DecisionEngine,
    DecisionResult,
    ExitOption,
)
from evacusim.decision.decision_processor import DecisionProcessor
from evacusim.decision.llm_decision_engine import LLMDecisionEngine


class _PerfTimer:
    @contextlib.contextmanager
    def measure(self, *args, **kwargs):
        yield


class _MessageSystem:
    def get_received_messages(self, agent_id):
        return []


class _ActionExecutor:
    def __init__(self):
        self.agent_action = {}


class _FakeAgent:
    """Concordia-entity stand-in: observe() is a no-op, act() returns canned JSON."""

    def __init__(self, response):
        self._response = response
        self.observed = []

    def observe(self, obs):
        self.observed.append(obs)

    def act(self, action_spec):
        return self._response


def _make_processor(concordia_agents, decision_engine=None):
    return DecisionProcessor(
        concordia_agents=concordia_agents,
        exited_agents=set(),
        action_translator=object(),
        action_executor=_ActionExecutor(),
        message_system=_MessageSystem(),
        state_queries=object(),
        station_layout={},
        agent_decisions={},
        agent_destinations={},
        last_observations={},
        last_actions={},
        perf_timer=_PerfTimer(),
        decision_engine=decision_engine,
    )


def _make_ctx(dp, agent_id, offered_actions, offered_exit_ids, prompt_text="PROMPT"):
    oa = set(offered_actions)
    oe = set(offered_exit_ids)
    ow = set()
    return DecisionContext(
        agent_id=agent_id,
        position=(0.0, 0.0),
        zone_id="concourse",
        goal="Leave the station.",
        observation="No significant new information.",
        agent_cfg={},
        offered_actions=list(offered_actions),
        offered_wait_reasons=[],
        offered_exit_ids=list(offered_exit_ids),
        exit_options={
            e: ExitOption(exit_id=e, display_name=e) for e in offered_exit_ids
        },
        route_blocked=False,
        cues=[],
        current_sim_time=0.0,
        offered_actions_set=oa,
        offered_wait_reasons_set=ow,
        offered_exit_ids_set=oe,
        prompt_text=prompt_text,
    )


class DecisionEngineSeamTests(unittest.TestCase):
    def test_default_engine_is_llm_engine(self):
        dp = _make_processor({})
        self.assertIsInstance(dp._engine, LLMDecisionEngine)
        self.assertIsInstance(dp._engine, DecisionEngine)

    def test_llm_path_produces_validated_payload(self):
        offered_actions = ["continue_activity", "wait"]
        offered_exit_ids = []
        # Build the processor first so we can ask it for a guaranteed-valid
        # payload (avoids hardcoding the decision schema in the test).
        dp = _make_processor({})
        fallback = dp._build_fallback_decision(
            "agent_0", set(offered_actions), set(), set(offered_exit_ids)
        )
        valid_json = dp._decision_payload_to_json(fallback)

        dp = _make_processor({"agent_0": _FakeAgent(valid_json)})
        ctx = _make_ctx(dp, "agent_0", offered_actions, offered_exit_ids)

        async def run():
            dp._state_lock = asyncio.Lock()
            dp._llm_semaphore = asyncio.Semaphore(1)
            return await dp._llm_produce_decision(ctx)

        result = asyncio.run(run())
        self.assertIsInstance(result, DecisionResult)
        self.assertIsNotNone(result.payload)
        self.assertTrue(result.llm_was_called)
        self.assertEqual(result.repair_status, "ok")
        self.assertFalse(result.skip_downstream)
        # action_json round-trips to the same validated payload.
        self.assertEqual(json.loads(result.action_json), result.payload)
        errors = dp._validate_decision_payload(
            result.payload, set(offered_actions), set(), set(offered_exit_ids)
        )
        self.assertEqual(errors, [])

    def test_llm_path_falls_back_on_garbage_response(self):
        offered_actions = ["continue_activity", "wait"]
        dp = _make_processor({"agent_0": _FakeAgent("not json at all")})
        ctx = _make_ctx(dp, "agent_0", offered_actions, [])

        async def run():
            dp._state_lock = asyncio.Lock()
            dp._llm_semaphore = asyncio.Semaphore(1)
            return await dp._llm_produce_decision(ctx)

        result = asyncio.run(run())
        self.assertIsNotNone(result.payload)
        self.assertEqual(result.repair_status, "fallback")
        self.assertTrue(result.llm_was_called)

    def test_injected_engine_is_used(self):
        calls = []

        class StubEngine:
            async def decide(self, ctx):
                calls.append(ctx.agent_id)
                return DecisionResult(payload={"action": "wait"}, action_json='{"action": "wait"}')

        dp = _make_processor({}, decision_engine=StubEngine())
        self.assertIsInstance(dp._engine, DecisionEngine)
        ctx = _make_ctx(dp, "agent_9", ["wait"], [])
        result = asyncio.run(dp._engine.decide(ctx))
        self.assertEqual(calls, ["agent_9"])
        self.assertEqual(result.payload, {"action": "wait"})


if __name__ == "__main__":
    unittest.main()
