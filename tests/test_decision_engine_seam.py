"""The decision-engine seam: the LLM engine's contract, and engine injection.

1. The LLM engine turns a model response into a validated payload, repairs an
   invalid one, falls back when every attempt fails, and reuses a decision
   when the prompt is unchanged.
2. The processor uses the LLM engine by default and an injected engine when
   given one, so a run is made LLM-free by injecting a different engine.
"""

import asyncio
import contextlib
import json
import unittest

from evacusim.core.decision_engine import DecisionContext, DecisionEngine, DecisionResult
from evacusim.decision import payload
from evacusim.decision.decision_processor import DecisionProcessor
from evacusim.decision.llm_decision_engine import LLMDecisionEngine
from evacusim.decision.llm_prompt import DecisionPromptBuilder

OFFERED = payload.OfferedSet(("continue_activity", "wait"), ("awaiting_information",), ())
VALID = payload.fallback(None, OFFERED)


class _PerfTimer:
    @contextlib.contextmanager
    def measure(self, *args, **kwargs):
        yield


class _MessageSystem:
    def get_received_messages(self, agent_id):
        return []


class _Registry:
    def get_display_name(self, exit_id):
        return exit_id

    def get_all_ids(self):
        return []


class _Translator:
    exit_registry = _Registry()
    zones_polygons = {}


class _FakeAgent:
    """Concordia-entity stand-in: act() returns the queued responses in turn."""

    def __init__(self, *responses):
        self._responses = list(responses)
        self.prompts = []

    def observe(self, observation):
        pass

    def act(self, spec):
        self.prompts.append(spec.call_to_action)
        return self._responses.pop(0) if len(self._responses) > 1 else self._responses[0]


def _engine(agent):
    history = {}
    return LLMDecisionEngine(
        agents={"agent_0": agent},
        prompt_builder=DecisionPromptBuilder(_Registry(), {}, history),
        message_system=_MessageSystem(),
        agent_decisions=history,
        perf_timer=_PerfTimer(),
        max_parallel=1,
    )


def _ctx(observation="No significant new information."):
    return DecisionContext(
        agent_id="agent_0",
        position=(0.0, 0.0),
        zone_id="concourse",
        goal="Leave the station.",
        observation=observation,
        agent_cfg={},
        offered_actions=list(OFFERED.actions),
        offered_wait_reasons=list(OFFERED.wait_reasons),
        offered_exit_ids=[],
        exit_options={},
        route_blocked=False,
        cues=[],
        current_sim_time=0.0,
        offered_actions_set=set(OFFERED.actions),
        offered_wait_reasons_set=set(OFFERED.wait_reasons),
        offered_exit_ids_set=set(),
    )


def _decide(engine, ctx):
    return asyncio.run(engine.decide(ctx))


class LLMEngineTests(unittest.TestCase):
    def test_valid_response_is_returned(self):
        result = _decide(_engine(_FakeAgent(json.dumps(VALID))), _ctx())
        self.assertEqual(result.payload, VALID)
        self.assertTrue(result.llm_was_called)
        self.assertEqual(result.repair_status, "ok")
        self.assertEqual(json.loads(result.action_json), result.payload)
        self.assertIn("Choose exactly one action from this offered set:", result.prompt)

    def test_invalid_response_is_repaired(self):
        agent = _FakeAgent("not json", json.dumps(VALID))
        result = _decide(_engine(agent), _ctx())
        self.assertEqual(result.repair_status, "repair_1")
        self.assertIn("SYSTEM NOTE", agent.prompts[1])

    def test_falls_back_when_every_attempt_fails(self):
        result = _decide(_engine(_FakeAgent("not json at all")), _ctx())
        self.assertEqual(result.repair_status, "fallback")
        self.assertEqual(payload.validate(result.payload, OFFERED), [])

    def test_unchanged_prompt_reuses_the_decision(self):
        agent = _FakeAgent(json.dumps(VALID))
        engine = _engine(agent)
        _decide(engine, _ctx())
        second = _decide(engine, _ctx())
        self.assertEqual(second.repair_status, "cached")
        self.assertFalse(second.llm_was_called)
        self.assertEqual(len(agent.prompts), 1)

    def test_reset_agent_forces_a_fresh_decision(self):
        agent = _FakeAgent(json.dumps(VALID))
        engine = _engine(agent)
        _decide(engine, _ctx())
        engine.reset_agent("agent_0")
        self.assertTrue(_decide(engine, _ctx()).llm_was_called)


class EngineInjectionTests(unittest.TestCase):
    def _processor(self, engine=None):
        return DecisionProcessor(
            agents={},
            exited_agents=set(),
            action_translator=_Translator(),
            action_executor=object(),
            message_system=_MessageSystem(),
            state_queries=object(),
            station_layout={},
            agent_decisions={},
            agent_destinations={},
            perf_timer=_PerfTimer(),
            decision_engine=engine,
        )

    def test_default_engine_is_llm_engine(self):
        engine = self._processor().engine
        self.assertIsInstance(engine, LLMDecisionEngine)
        self.assertIsInstance(engine, DecisionEngine)

    def test_injected_engine_is_used(self):
        class StubEngine(DecisionEngine):
            async def decide(self, ctx):
                return DecisionResult(payload={"action": "wait"})

        stub = StubEngine()
        self.assertIs(self._processor(stub).engine, stub)


if __name__ == "__main__":
    unittest.main()
