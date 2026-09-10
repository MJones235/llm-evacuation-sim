"""Validation and factory-selection tests for the optional ``decision`` config section."""

import unittest

from evacusim.config.config_loader import ConfigLoader
from evacusim.setup.simulation_runner_factory import SimulationRunnerFactory
from evacusim.decision.rule_based_decision_engine import RuleBasedDecisionEngine


class DecisionSectionValidationTests(unittest.TestCase):
    """`ConfigLoader._validate_decision_section` accepts valid shapes, rejects bad ones."""

    def test_absent_section_is_ok(self):
        ConfigLoader._validate_decision_section({})

    def test_llm_engine_ok(self):
        ConfigLoader._validate_decision_section({"decision": {"engine": "llm"}})

    def test_rule_based_with_weights_ok(self):
        ConfigLoader._validate_decision_section(
            {
                "decision": {
                    "engine": "rule_based",
                    "crowd_radius_m": 4.0,
                    "rule_weights": {"proximity": 0.5, "busyness": 0.3, "familiarity": 0.2},
                }
            }
        )

    def test_non_dict_section_rejected(self):
        with self.assertRaises(ValueError):
            ConfigLoader._validate_decision_section({"decision": "rule_based"})

    def test_unknown_engine_rejected(self):
        with self.assertRaises(ValueError):
            ConfigLoader._validate_decision_section({"decision": {"engine": "telepathy"}})

    def test_negative_weight_rejected(self):
        with self.assertRaises(ValueError):
            ConfigLoader._validate_decision_section(
                {"decision": {"engine": "rule_based", "rule_weights": {"proximity": -1}}}
            )

    def test_nonpositive_crowd_radius_rejected(self):
        with self.assertRaises(ValueError):
            ConfigLoader._validate_decision_section(
                {"decision": {"engine": "rule_based", "crowd_radius_m": 0}}
            )


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
                    "rule_weights": {"proximity": 0.6, "busyness": 0.25, "familiarity": 0.15},
                }
            }
        )
        self.assertIsInstance(engine, RuleBasedDecisionEngine)
        self.assertEqual(engine.w_proximity, 0.6)
        self.assertEqual(engine.w_busyness, 0.25)
        self.assertEqual(engine.w_familiarity, 0.15)
        self.assertEqual(engine.crowd_radius_m, 7.5)

    def test_rule_based_defaults_when_weights_absent(self):
        engine = SimulationRunnerFactory._build_decision_engine(
            {"decision": {"engine": "rule_based"}}
        )
        self.assertIsInstance(engine, RuleBasedDecisionEngine)
        self.assertEqual(engine.w_proximity, 0.5)
        self.assertEqual(engine.w_busyness, 0.3)
        self.assertEqual(engine.w_familiarity, 0.2)

    def test_unknown_engine_falls_back_to_none(self):
        self.assertIsNone(
            SimulationRunnerFactory._build_decision_engine({"decision": {"engine": "wat"}})
        )


if __name__ == "__main__":
    unittest.main()
