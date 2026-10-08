"""Factory-selection tests for the ``decision`` config section.

Schema validation of the section is tested in test_config_schema.py.
"""

import unittest

from evacusim.decision.rule_based_decision_engine import RuleBasedDecisionEngine
from evacusim.setup.simulation_runner_factory import SimulationRunnerFactory


class DecisionEngineFactoryTests(unittest.TestCase):
    """`SimulationRunnerFactory._build_decision_engine` maps config to an engine (or None)."""

    def test_default_is_none(self):
        self.assertIsNone(SimulationRunnerFactory._build_decision_engine({}))

    def test_llm_is_none(self):
        self.assertIsNone(
            SimulationRunnerFactory._build_decision_engine({"decision": {"engine": "llm"}})
        )

    def test_rule_based_builds_engine_with_weights(self):
        engine = SimulationRunnerFactory._build_decision_engine(
            {
                "decision": {
                    "engine": "rule_based",
                    "crowd_radius_m": 7.5,
                    "rule_weights": {
                        "proximity": 0.35,
                        "visibility": 0.5,
                        "busyness": 0.05,
                        "familiarity": 0.1,
                    },
                }
            }
        )
        self.assertIsInstance(engine, RuleBasedDecisionEngine)
        self.assertEqual(engine.w_proximity, 0.35)
        self.assertEqual(engine.w_visibility, 0.5)
        self.assertEqual(engine.w_busyness, 0.05)
        self.assertEqual(engine.w_familiarity, 0.1)
        self.assertEqual(engine.crowd_radius_m, 7.5)

    def test_rule_based_defaults_when_weights_absent(self):
        engine = SimulationRunnerFactory._build_decision_engine(
            {"decision": {"engine": "rule_based"}}
        )
        self.assertIsInstance(engine, RuleBasedDecisionEngine)
        self.assertEqual(engine.w_proximity, 0.5)
        self.assertEqual(engine.w_visibility, 0.0)
        self.assertEqual(engine.w_busyness, 0.3)
        self.assertEqual(engine.w_familiarity, 0.2)

    def test_unknown_engine_falls_back_to_none(self):
        self.assertIsNone(
            SimulationRunnerFactory._build_decision_engine({"decision": {"engine": "wat"}})
        )


if __name__ == "__main__":
    unittest.main()
