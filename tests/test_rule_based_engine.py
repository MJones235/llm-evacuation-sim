"""Unit tests for the LLM-free RuleBasedDecisionEngine (Feature B, B4/B6).

Each proximity/busyness/familiarity signal is isolated via the weights, and
every produced payload is checked against the *real* decision schema validator
(DecisionProcessor._validate_decision_payload) so the rule engine's output is
guaranteed interchangeable with the LLM engine's downstream.
"""

import asyncio
import contextlib
import unittest

from evacusim.core.decision_engine import DecisionContext, ExitOption
from evacusim.decision.decision_processor import DecisionProcessor
from evacusim.decision.rule_based_decision_engine import RuleBasedDecisionEngine


# --- minimal processor purely to reuse the schema validator -----------------
class _PerfTimer:
    @contextlib.contextmanager
    def measure(self, *a, **k):
        yield


class _MsgSys:
    def get_received_messages(self, agent_id):
        return []


def _validator():
    return DecisionProcessor(
        concordia_agents={},
        exited_agents=set(),
        action_translator=object(),
        action_executor=object(),
        message_system=_MsgSys(),
        state_queries=object(),
        station_layout={},
        agent_decisions={},
        agent_destinations={},
        last_observations={},
        last_actions={},
        perf_timer=_PerfTimer(),
    )


def _ctx(exit_options, offered_actions, offered_exit_ids, *, goal="Leave the station.",
         route_blocked=False, offered_wait_reasons=("awaiting_information",)):
    return DecisionContext(
        agent_id="agent_0",
        position=(0.0, 0.0),
        zone_id="concourse",
        goal=goal,
        observation="obs",
        agent_cfg={},
        offered_actions=list(offered_actions),
        offered_wait_reasons=list(offered_wait_reasons),
        offered_exit_ids=list(offered_exit_ids),
        exit_options={o.exit_id: o for o in exit_options},
        route_blocked=route_blocked,
        cues=[],
        current_sim_time=0.0,
        offered_actions_set=set(offered_actions),
        offered_wait_reasons_set=set(offered_wait_reasons),
        offered_exit_ids_set=set(offered_exit_ids),
    )


def _decide(engine, ctx):
    return asyncio.run(engine.decide(ctx))


class RuleBasedEngineTests(unittest.TestCase):
    def setUp(self):
        self.dp = _validator()

    def _assert_valid(self, payload, ctx):
        errors = self.dp._validate_decision_payload(
            payload,
            ctx.offered_actions_set,
            ctx.offered_wait_reasons_set,
            ctx.offered_exit_ids_set,
        )
        self.assertEqual(errors, [], f"payload failed schema: {errors}\n{payload}")

    def test_proximity_dominates(self):
        eng = RuleBasedDecisionEngine(w_proximity=1.0, w_busyness=0.0, w_familiarity=0.0)
        opts = [
            ExitOption("far", "Far", distance_m=50.0, crowd_count=0, familiar=False),
            ExitOption("near", "Near", distance_m=3.0, crowd_count=0, familiar=False),
        ]
        ctx = _ctx(opts, ["evacuate", "wait", "continue_activity"], ["far", "near"])
        res = _decide(eng, ctx)
        self.assertEqual(res.payload["action"], "evacuate")
        self.assertEqual(res.payload["exit_id"], "near")
        self.assertFalse(res.llm_was_called)
        self.assertEqual(res.repair_status, "rule_based")
        self._assert_valid(res.payload, ctx)

    def test_busyness_dominates_when_distance_equal(self):
        eng = RuleBasedDecisionEngine(w_proximity=0.0, w_busyness=1.0, w_familiarity=0.0)
        opts = [
            ExitOption("busy", "Busy", distance_m=10.0, crowd_count=8, familiar=False),
            ExitOption("quiet", "Quiet", distance_m=10.0, crowd_count=0, familiar=False),
        ]
        ctx = _ctx(opts, ["evacuate", "wait", "continue_activity"], ["busy", "quiet"])
        res = _decide(eng, ctx)
        self.assertEqual(res.payload["exit_id"], "quiet")
        self._assert_valid(res.payload, ctx)

    def test_familiarity_dominates(self):
        eng = RuleBasedDecisionEngine(w_proximity=0.0, w_busyness=0.0, w_familiarity=1.0)
        opts = [
            ExitOption("unknown", "Unknown", distance_m=5.0, crowd_count=0, familiar=False),
            ExitOption("known", "Known", distance_m=5.0, crowd_count=0, familiar=True),
        ]
        ctx = _ctx(opts, ["evacuate", "wait", "continue_activity"], ["unknown", "known"])
        res = _decide(eng, ctx)
        self.assertEqual(res.payload["exit_id"], "known")
        self._assert_valid(res.payload, ctx)

    def test_weights_change_outcome(self):
        opts = [
            ExitOption("near_busy", "NearBusy", distance_m=2.0, crowd_count=20, familiar=False),
            ExitOption("far_quiet", "FarQuiet", distance_m=40.0, crowd_count=0, familiar=False),
        ]
        offered = ["evacuate", "wait", "continue_activity"]
        ids = ["near_busy", "far_quiet"]
        prox_eng = RuleBasedDecisionEngine(w_proximity=1.0, w_busyness=0.0, w_familiarity=0.0)
        busy_eng = RuleBasedDecisionEngine(w_proximity=0.0, w_busyness=1.0, w_familiarity=0.0)
        self.assertEqual(_decide(prox_eng, _ctx(opts, offered, ids)).payload["exit_id"], "near_busy")
        self.assertEqual(_decide(busy_eng, _ctx(opts, offered, ids)).payload["exit_id"], "far_quiet")

    def test_ties_are_deterministic_first(self):
        eng = RuleBasedDecisionEngine()
        opts = [
            ExitOption("a", "A", distance_m=5.0, crowd_count=1, familiar=True),
            ExitOption("b", "B", distance_m=5.0, crowd_count=1, familiar=True),
        ]
        ctx = _ctx(opts, ["evacuate", "wait", "continue_activity"], ["a", "b"])
        first = _decide(eng, ctx).payload["exit_id"]
        self.assertEqual(first, "a")
        # Repeated calls are identical (no randomness).
        self.assertEqual(_decide(eng, ctx).payload["exit_id"], "a")

    def test_blocked_route_waits(self):
        eng = RuleBasedDecisionEngine()
        ctx = _ctx(
            [], ["wait", "continue_activity"], [],
            route_blocked=True,
            offered_wait_reasons=("awaiting_information", "route_blocked"),
        )
        res = _decide(eng, ctx)
        self.assertEqual(res.payload["action"], "wait")
        self.assertEqual(res.payload["wait_reason"], "route_blocked")
        self.assertIsNone(res.payload["pace"])
        self._assert_valid(res.payload, ctx)

    def test_no_exits_continues_activity(self):
        eng = RuleBasedDecisionEngine()
        ctx = _ctx([], ["wait", "continue_activity"], [])
        res = _decide(eng, ctx)
        self.assertEqual(res.payload["action"], "continue_activity")
        self.assertIsNone(res.payload["exit_id"])
        self.assertEqual(res.payload["pace"], "normal_pace")
        self._assert_valid(res.payload, ctx)

    def test_leave_by_train_when_goal_is_platform(self):
        eng = RuleBasedDecisionEngine()
        opts = [ExitOption("grey_street", "Grey Street", distance_m=4.0)]
        ctx = _ctx(
            opts,
            ["evacuate", "wait", "continue_activity", "leave_by_train"],
            ["grey_street"],
            goal="Board your train at platform 3.",
        )
        res = _decide(eng, ctx)
        self.assertEqual(res.payload["action"], "leave_by_train")
        self.assertIsNone(res.payload["exit_id"])
        self.assertEqual(res.payload["pace"], "normal_pace")
        self._assert_valid(res.payload, ctx)

    def test_leave_goal_ignores_train_action(self):
        # Same offered actions but a leave-the-station goal → should evacuate, not board.
        eng = RuleBasedDecisionEngine()
        opts = [ExitOption("grey_street", "Grey Street", distance_m=4.0)]
        ctx = _ctx(
            opts,
            ["evacuate", "wait", "continue_activity", "leave_by_train"],
            ["grey_street"],
            goal="Leave the station by the nearest exit.",
        )
        res = _decide(eng, ctx)
        self.assertEqual(res.payload["action"], "evacuate")
        self.assertEqual(res.payload["exit_id"], "grey_street")
        self._assert_valid(res.payload, ctx)


if __name__ == "__main__":
    unittest.main()
