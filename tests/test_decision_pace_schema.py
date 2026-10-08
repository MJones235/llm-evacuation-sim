"""Pace rules in the decision payload and in the LLM prompt."""

import unittest

from evacusim.decision import payload as payload_lib
from evacusim.decision.llm_prompt import pace_blocks


class DecisionPaceSchemaTests(unittest.TestCase):
    def setUp(self):
        self.base_assessment = {
            "source_credibility": "credible",
            "situation_appraisal": "busy",
            "personal_relevance": "high",
            "options_considered": "wait or evacuate",
            "option_chosen_because": "best available",
            "information_gap": "none",
        }

    def _validate(self, payload, offered_actions=None):
        offered = payload_lib.OfferedSet(
            tuple(offered_actions or ("wait", "evacuate", "continue_activity")),
            ("awaiting_information", "grouping"),
            ("Exit_A",),
        )
        return payload_lib.validate(payload, offered)

    def test_wait_requires_null_pace(self):
        payload = {
            "assessment": self.base_assessment,
            "action": "wait",
            "wait_reason": "awaiting_information",
            "exit_id": None,
            "pace": "normal_pace",
            "reassess_when": "next_interval",
        }
        errors = self._validate(payload)
        self.assertTrue(any("pace must be null when action=wait" in e for e in errors))

    def test_wait_with_null_pace_is_valid(self):
        payload = {
            "assessment": self.base_assessment,
            "action": "wait",
            "wait_reason": "awaiting_information",
            "exit_id": None,
            "pace": None,
            "reassess_when": "next_interval",
        }
        errors = self._validate(payload)
        self.assertEqual(errors, [])

    def test_non_wait_requires_labeled_pace(self):
        payload = {
            "assessment": self.base_assessment,
            "action": "continue_activity",
            "wait_reason": None,
            "exit_id": None,
            "pace": None,
            "reassess_when": "next_interval",
        }
        errors = self._validate(payload)
        self.assertTrue(any("pace must be one of" in e for e in errors))

    def test_evacuate_with_valid_pace_is_valid(self):
        payload = {
            "assessment": self.base_assessment,
            "action": "evacuate",
            "wait_reason": None,
            "exit_id": "Exit_A",
            "pace": "hurrying",
            "reassess_when": "new_cue_only",
        }
        errors = self._validate(payload)
        self.assertEqual(errors, [])

    def test_wait_only_prompt_blocks_omit_pace(self):
        wait_rule, pace_field, pace_validation = pace_blocks(["wait"])
        self.assertIn("set wait_reason", wait_rule)
        self.assertEqual(pace_field, "")
        self.assertEqual(pace_validation, "")

    def test_mixed_prompt_blocks_include_pace(self):
        wait_rule, pace_field, pace_validation = pace_blocks(["wait", "evacuate"])
        self.assertIn("set pace to null", wait_rule)
        self.assertIn('"pace": null', pace_field)
        self.assertIn("pace must be one of", pace_validation)


if __name__ == "__main__":
    unittest.main()
