"""The parameter schema accepts valid configurations and rejects invalid ones."""

import copy
import unittest

from evacusim.config.config_loader import ConfigError, ConfigLoader
from evacusim.config.schema import (
    BlockExitEvent,
    MessageEvent,
    PAAnnouncementEvent,
    RuleBasedDecisionConfig,
    TrainArrivalEvent,
)

MINIMAL = {
    "simulation": {"network_path": "network"},
    "agents": {"count": 0, "knowledge_profiles": {"test": 1}},
    "station": {
        "knowledge": {"base_memories": ["Test station."], "profiles": {"test": ["Known."]}}
    },
}

CALIBRATION = {
    "enabled": True,
    "entrance_usage_csv": "u.csv",
    "timetable_csv": "t.csv",
    "entrance_dest_exits": ["train_platform_1"],
    "platform_exit": "street_exit_a",
    "spawn_points": {
        "entrance_a": {"level": "0", "xy": [10.0, 5.0]},
        "1": {"level": "-1", "xy": [20.0, -3.0]},
    },
}


def config(**sections):
    """MINIMAL with the given top-level sections replaced or added."""
    cfg = copy.deepcopy(MINIMAL)
    cfg.update(copy.deepcopy(sections))
    return cfg


class SchemaTests(unittest.TestCase):
    def assertInvalid(self, cfg, mentions):
        with self.assertRaises(ConfigError) as ctx:
            ConfigLoader.validate_config(cfg)
        self.assertIn(mentions, str(ctx.exception))

    def test_minimal_config_fills_defaults(self):
        run = ConfigLoader.validate_config(config())
        self.assertEqual(run.simulation.dt, 0.05)
        self.assertEqual(run.decision.engine, "llm")
        self.assertIsNone(run.calibration)

    def test_unknown_key_is_rejected_with_its_path(self):
        cfg = config()
        cfg["simulation"]["spawn_interval"] = 2.0
        self.assertInvalid(cfg, "simulation.spawn_interval")

    def test_values_are_not_coerced(self):
        for invalid in (-1, 86400, "07:30", True):
            cfg = config()
            cfg["simulation"]["start_time_s"] = invalid
            self.assertInvalid(cfg, "simulation.start_time_s")

    def test_int_is_accepted_for_float(self):
        cfg = config()
        cfg["simulation"]["start_time_s"] = 27000
        self.assertEqual(ConfigLoader.validate_config(cfg).simulation.start_time_s, 27000)

    def test_knowledge_profiles_must_exist_in_station(self):
        cfg = config()
        cfg["agents"]["knowledge_profiles"] = {"expert": 1}
        self.assertInvalid(cfg, "expert")

    def test_knowledge_profile_weights_must_be_positive(self):
        cfg = config()
        cfg["agents"]["knowledge_profiles"] = {"test": 0}
        self.assertInvalid(cfg, "knowledge_profiles.test")


class DecisionSectionTests(SchemaTests):
    def test_llm_engine(self):
        run = ConfigLoader.validate_config(config(decision={"engine": "llm"}))
        self.assertEqual(run.decision.engine, "llm")

    def test_rule_based_with_weights(self):
        run = ConfigLoader.validate_config(
            config(
                decision={
                    "engine": "rule_based",
                    "crowd_radius_m": 4.0,
                    "rule_weights": {
                        "proximity": 0.35,
                        "visibility": 0.5,
                        "busyness": 0.05,
                        "familiarity": 0.1,
                    },
                }
            )
        )
        self.assertIsInstance(run.decision, RuleBasedDecisionConfig)
        self.assertEqual(run.decision.rule_weights.visibility, 0.5)

    def test_rule_based_defaults(self):
        run = ConfigLoader.validate_config(config(decision={"engine": "rule_based"}))
        weights = run.decision.rule_weights
        self.assertEqual(
            (weights.proximity, weights.busyness, weights.familiarity, weights.visibility),
            (0.5, 0.3, 0.2, 0.0),
        )

    def test_rule_settings_are_rejected_for_llm(self):
        self.assertInvalid(config(decision={"engine": "llm", "crowd_radius_m": 4.0}), "crowd")

    def test_non_mapping_rejected(self):
        self.assertInvalid(config(decision="rule_based"), "decision")

    def test_unknown_engine_rejected(self):
        self.assertInvalid(config(decision={"engine": "telepathy"}), "decision")

    def test_negative_weight_rejected(self):
        self.assertInvalid(
            config(decision={"engine": "rule_based", "rule_weights": {"proximity": -1}}),
            "proximity",
        )

    def test_nonpositive_crowd_radius_rejected(self):
        self.assertInvalid(
            config(decision={"engine": "rule_based", "crowd_radius_m": 0}), "crowd_radius_m"
        )


class CalibrationSectionTests(SchemaTests):
    def rule_based(self, **calibration):
        return config(decision={"engine": "rule_based"}, calibration={**CALIBRATION, **calibration})

    def test_valid(self):
        run = ConfigLoader.validate_config(self.rule_based())
        self.assertEqual(run.calibration.spawn_points["1"].xy, [20.0, -3.0])

    def test_requires_rule_based_engine(self):
        self.assertInvalid(config(calibration=CALIBRATION), "rule_based")

    def test_disabled_does_not_require_rule_based_engine(self):
        ConfigLoader.validate_config(config(calibration={**CALIBRATION, "enabled": False}))

    def test_missing_usage_csv(self):
        cfg = self.rule_based()
        del cfg["calibration"]["entrance_usage_csv"]
        self.assertInvalid(cfg, "entrance_usage_csv")

    def test_bad_spawn_points(self):
        self.assertInvalid(self.rule_based(spawn_points={}), "spawn_points")
        self.assertInvalid(
            self.rule_based(spawn_points={"e": {"level": "0", "xy": [1.0]}}), "spawn_points.e.xy"
        )

    def test_bad_dest_exits(self):
        self.assertInvalid(self.rule_based(entrance_dest_exits="x"), "entrance_dest_exits")

    def test_seed_is_top_level(self):
        self.assertInvalid(self.rule_based(seed=7), "calibration.seed")
        self.assertInvalid(config(seed="seven"), "seed")
        self.assertEqual(ConfigLoader.validate_config(config(seed=7)).seed, 7)


class EventTests(SchemaTests):
    def test_each_event_type_maps_to_its_model(self):
        run = ConfigLoader.validate_config(
            config(
                events=[
                    {"time": 15.0, "message": "The fire alarm is ringing."},
                    {"time": 20.0, "type": "pa_announcement", "message": "Please leave."},
                    {"time": 10.0, "type": "train_arrival", "platforms": [1, 2]},
                    {"time": 15.0, "type": "block_exit", "exits": ["escalator_d_down"]},
                ]
            )
        )
        self.assertEqual(
            [type(e) for e in run.events],
            [MessageEvent, PAAnnouncementEvent, TrainArrivalEvent, BlockExitEvent],
        )
        self.assertEqual(run.events[2].dwell_seconds, 30.0)

    def test_field_of_another_event_type_is_rejected(self):
        self.assertInvalid(
            config(events=[{"time": 15.0, "message": "Alarm.", "platforms": [1]}]), "platforms"
        )

    def test_staff_phase_uses_after_seconds(self):
        staff = {
            "enabled": True,
            "phases": [{"trigger": "after_seconds", "trigger_after_seconds": 480}],
        }
        self.assertInvalid(config(systems={"brigade": staff}), "trigger_after_seconds")


if __name__ == "__main__":
    unittest.main()
