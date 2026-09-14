"""B7 end-to-end gate: a full decision cycle runs with ZERO LLM calls.

This is the acceptance test for Feature B's goal — "run an entire scenario
without requiring any LLM calls".  It drives the *real*
``DecisionProcessor.process_all_agents`` decision cycle (assembly → engine →
translate → execute → telemetry) with a :class:`RuleBasedDecisionEngine` and
:class:`NoOpAgent` registry, using lightweight fakes only for the physics
boundary (JuPedSim / action execution).  It asserts:

  * every agent produces a *validated* decision payload,
  * the rule engine (not the LLM) produced it (``repair_status == "rule_based"``),
  * no LLM call was made and none was even attempted (NoOpAgent.act raises),
  * the financial report shows no LLM usage (the zero-LLM signal).

Full-geometry scenarios are driven by the external study repo; this test
locks the LLM-free contract inside evacusim without needing station geometry.
"""

import contextlib
import unittest

from shapely.geometry import box

from evacusim.decision.decision_processor import DecisionProcessor
from evacusim.decision.rule_based_decision_engine import RuleBasedDecisionEngine
from evacusim.coordination.noop_agent import NoOpAgent
from evacusim.translation.action_translator import ActionTranslator
from evacusim.metrics.llm_cost_reporter import FinancialReporter


# --- physics-boundary fakes ------------------------------------------------
class _PerfTimer:
    @contextlib.contextmanager
    def measure(self, *args, **kwargs):
        yield


class _MessageSystem:
    def get_received_messages(self, agent_id):
        return []


class _FakeJps:
    """Single-level JuPedSim stand-in (no ``simulations`` attr)."""

    def __init__(self, positions):
        self._positions = dict(positions)
        self.escalator_controller = None

    def get_agent_position(self, agent_id):
        return self._positions.get(agent_id)

    def get_all_agent_positions(self):
        return dict(self._positions)


class _FakeStateQueries:
    def __init__(self, jps):
        self._jps = jps

    def get_agent_position(self, agent_id):
        return self._jps.get_agent_position(agent_id)


class _RecordingExecutor:
    """Absorbs execute_action; never touches an LLM. Records executed actions."""

    def __init__(self):
        self.agent_action = {}
        self.executed = []

    def has_reachable_information_source(self, agent_id, position, zone_id):
        return False

    def execute_action(self, agent_id, translated, current_sim_time):
        self.executed.append((agent_id, translated))


def _station_layout():
    """Minimal single-level layout: one concourse zone, one street exit."""
    concourse = box(-10.0, -10.0, 60.0, 10.0)
    return {
        "exits": {"main_exit": (50.0, 0.0)},
        "zones": {"concourse": {}},
        "zones_polygons": {"concourse": concourse},
        # Novices in the concourse know the main exit (drives exit-offering).
        "zone_known_exits_by_profile": {"concourse": {"novice": ["main_exit"]}},
        "zone_goal_keywords": {},
        "run_metadata": {"run_id": "b7-zero-llm", "condition": "rule_based"},
    }


def _make_processor(agents_config, positions):
    jps = _FakeJps(positions)
    layout = _station_layout()
    translator = ActionTranslator(layout, None, jps)
    executor = _RecordingExecutor()
    dp = DecisionProcessor(
        concordia_agents={c["id"]: NoOpAgent(c["id"], c.get("name")) for c in agents_config},
        exited_agents=set(),
        action_translator=translator,
        action_executor=executor,
        message_system=_MessageSystem(),
        state_queries=_FakeStateQueries(jps),
        station_layout=layout,
        agent_decisions={},
        agent_destinations={},
        last_observations={},
        last_actions={},
        perf_timer=_PerfTimer(),
        jps_sim=jps,
        agent_configs=agents_config,
        decision_engine=RuleBasedDecisionEngine(),
    )
    return dp, executor


class ZeroLLMFullCycleTests(unittest.TestCase):
    def test_full_decision_cycle_makes_no_llm_calls(self):
        agents_config = [
            {
                "id": f"agent_{i}",
                "name": f"Agent {i}",
                "knowledge_profile": "novice",
                "level_id": "0",
                "initial_goal": "Leave the station.",
            }
            for i in range(3)
        ]
        positions = {
            "agent_0": (0.0, 0.0),
            "agent_1": (10.0, 2.0),
            "agent_2": (20.0, -3.0),
        }
        dp, executor = _make_processor(agents_config, positions)

        observations = {a["id"]: "No significant new information." for a in agents_config}
        dp.process_all_agents(observations, current_sim_time=0.0)

        # (1) Every agent produced a decision — a swallowed per-agent exception
        # would leave agent_decisions empty and fail here.
        self.assertEqual(set(dp.agent_decisions), {a["id"] for a in agents_config})

        for agent_id, record in dp.agent_decisions.items():
            decisions = record["decisions"]
            self.assertTrue(decisions, f"{agent_id} has no decisions")
            last = decisions[-1]
            payload = last["decision_payload"]

            # (2) Payload is schema-valid against the offered sets.
            errors = dp._validate_decision_payload(
                payload,
                set(last["offered_set"]),
                {"awaiting_information", "awaiting_instruction", "route_blocked"},
                {"main_exit"},
            )
            self.assertEqual(errors, [], f"{agent_id} payload invalid: {errors}")

            # (3) The rule engine produced it — not the LLM.
            self.assertEqual(last["repair_status"], "rule_based")
            # (4) No LLM prompt was issued for this decision.
            self.assertEqual(last["prompt"], "cached")

        # (5) The processor recorded zero LLM calls for the whole cycle.
        self.assertEqual(dp.llm_calls_made, 0)

        # (6) With a leave-the-station goal and a known exit, the rule engine
        # routes to evacuate — proving the translate/execute path ran LLM-free.
        actions = [d["decisions"][-1]["decision_payload"]["action"] for d in dp.agent_decisions.values()]
        self.assertIn("evacuate", actions)
        self.assertTrue(executor.executed, "no actions were executed")

    def test_noop_agent_act_would_raise_if_llm_path_reached(self):
        # Guard: the LLM path must be unreachable in a rule-based run.
        with self.assertRaises(RuntimeError):
            NoOpAgent("agent_0", "Agent 0").act(object())

    def test_financial_report_shows_no_llm_usage_without_provider(self):
        # A rule-based run passes language_model=None, so llm_provider is None.
        report = FinancialReporter.generate_report(None, num_agents=3)
        self.assertIn("FINANCIAL REPORT", report)
        self.assertIn("usage stats not available", report)


if __name__ == "__main__":
    unittest.main()
