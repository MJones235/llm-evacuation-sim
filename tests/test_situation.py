"""Perception: usable exits, goal commitment and cue detection."""

import unittest

from evacusim.decision import payload
from evacusim.decision.situation import (
    EVACUATION_GOAL,
    CueDetector,
    GoalTracker,
    SituationAssembler,
)


class _Registry:
    NAMES = {"grey_street": "Grey Street", "escalator_a_down": "Escalator A", "esc_up": "Up"}

    def get_all_ids(self):
        return list(self.NAMES)

    def get_display_name(self, exit_id):
        return self.NAMES[exit_id]


class _Translator:
    exit_registry = _Registry()


def _assembler(profile="novice"):
    layout = {
        "arrival_exits_by_zone": {"concourse": ["esc_up"]},
        "zone_known_exits_by_profile": {
            "concourse": {"commuter": ["escalator_a_down"], "novice": ["grey_street"]}
        },
    }
    return SituationAssembler(
        agent_cfg={"a": {"knowledge_profile": profile}},
        station_layout=layout,
        action_translator=_Translator(),
        action_executor=object(),
        agent_destinations={},
    )


class UsableExitTests(unittest.TestCase):
    def test_novice_falls_back_to_known_exits(self):
        self.assertEqual(_assembler()._usable_exit_ids("a", "", "concourse"), ["grey_street"])

    def test_visible_exits_are_offered_but_not_arrival_exits(self):
        obs = "Exits visible right now: Grey Street (nearby); Up (very close)."
        self.assertEqual(_assembler()._usable_exit_ids("a", obs, "concourse"), ["grey_street"])

    def test_commuters_recall_known_exits_first(self):
        obs = "Exits visible right now: Grey Street (nearby)."
        self.assertEqual(
            _assembler("commuter")._usable_exit_ids("a", obs, "concourse"),
            ["escalator_a_down", "grey_street"],
        )

    def test_exits_seen_blocked_are_excluded(self):
        obs = "Exits visible right now: Grey Street (nearby). The Grey Street appears blocked or obstructed"
        self.assertEqual(_assembler()._usable_exit_ids("a", obs, "concourse"), [])


class GoalTrackerTests(unittest.TestCase):
    def test_evacuating_replaces_the_goal_until_all_clear(self):
        goals = GoalTracker({"a": {"initial_goal": "Catch your train."}})
        goals.goals["a"] = "Meet a friend."
        goals.apply_decision("a", {"action": "evacuate"}, "")
        self.assertEqual(goals.current("a"), EVACUATION_GOAL)
        goals.apply_decision("a", {"action": "wait"}, "The all clear has been given.")
        self.assertEqual(goals.current("a"), "Catch your train.")

    def test_heading_for_a_train_is_not_evacuating(self):
        goals = GoalTracker({"a": {"initial_goal": "Catch your train from Platform 1."}})
        goals.current("a")
        goals.apply_decision("a", {"action": "evacuate"}, "")
        self.assertEqual(goals.current("a"), "Catch your train from Platform 1.")


class CueDetectorTests(unittest.TestCase):
    def test_first_sighting_sets_baselines_without_cues(self):
        self.assertEqual(CueDetector().detect("a", "", "concourse", False), [])

    def test_changes_are_cues(self):
        cues = CueDetector()
        cues.detect("a", "", "concourse", False)
        found = cues.detect("a", "The fire alarm is sounding.", "platform_1", True)
        self.assertEqual(found, ["zone_entry", "route_blocked", "alarm_state_change"])
        self.assertEqual(cues.detect("a", "The fire alarm is sounding.", "platform_1", True), [])


class PayloadTests(unittest.TestCase):
    OFFERED = payload.OfferedSet(("wait", "evacuate"), ("awaiting_information",), ("grey_street",))

    def test_fallback_repeats_a_still_valid_previous_decision(self):
        previous = payload.fallback(None, self.OFFERED)
        self.assertIs(payload.fallback(previous, self.OFFERED), previous)

    def test_fallback_waits_when_continuing_is_not_offered(self):
        decision = payload.fallback({"action": "teleport"}, self.OFFERED)
        self.assertEqual(decision["action"], "wait")
        self.assertEqual(payload.validate(decision, self.OFFERED), [])

    def test_evacuate_requires_an_offered_exit(self):
        decision = {**payload.fallback(None, self.OFFERED), "action": "evacuate"}
        decision.update(wait_reason=None, pace="running", exit_id="blackett_street")
        self.assertIn("exit_id", payload.validate(decision, self.OFFERED)[0])

    def test_extract_json_object_ignores_leading_text(self):
        self.assertEqual(payload.extract_json_object('Answer: {"a": 1}'), {"a": 1})
        self.assertIsNone(payload.extract_json_object("no json"))


if __name__ == "__main__":
    unittest.main()
