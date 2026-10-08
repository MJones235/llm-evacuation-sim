"""The fake language model answers decision prompts validly and deterministically."""

import json
import unittest

from evacusim.testing.fake_llm import FakeLanguageModel, fake_embedder

PROMPT = """Block 5 - Available actions
Choose exactly one action from this offered set:
- continue_activity (no extra field)
- wait (required: wait_reason in {awaiting_information, awaiting_instruction})
- evacuate (required: exit_id from {grey_street, eldon_square})

Validation rules:
- if action = wait: set wait_reason and set pace to null.
"""


class FakeLanguageModelTests(unittest.TestCase):
    def test_answers_only_with_offered_options(self):
        model = FakeLanguageModel()
        for i in range(40):
            prompt = f"{PROMPT}\nvariant {i}\nSYSTEM NOTE: repair"
            payload = json.loads(model.sample_text(prompt))
            self.assertIn(payload["action"], {"continue_activity", "wait", "evacuate"})
            if payload["action"] == "evacuate":
                self.assertIn(payload["exit_id"], {"grey_street", "eldon_square"})
            if payload["action"] == "wait":
                self.assertIn(
                    payload["wait_reason"], {"awaiting_information", "awaiting_instruction"}
                )
                self.assertIsNone(payload["pace"])

    def test_same_prompt_same_answer_across_instances(self):
        self.assertEqual(
            FakeLanguageModel().sample_text(PROMPT), FakeLanguageModel().sample_text(PROMPT)
        )

    def test_some_first_attempts_are_malformed(self):
        model = FakeLanguageModel()
        answers = [model.sample_text(f"{PROMPT}\nvariant {i}") for i in range(50)]
        self.assertTrue(any(not a.startswith("{") for a in answers))

    def test_embedder_is_deterministic(self):
        self.assertEqual(fake_embedder("hello").tolist(), fake_embedder("hello").tolist())


if __name__ == "__main__":
    unittest.main()
