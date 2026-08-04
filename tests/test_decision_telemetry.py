import unittest

from evacusim.metrics.decision_telemetry import build_decision_telemetry


class DecisionTelemetryTests(unittest.TestCase):
    def test_rollup_counts_and_rates(self):
        agent_decisions = {
            "a1": {
                "decisions": [
                    {
                        "prompt": "cached",
                        "repair_status": "no_repair_needed",
                        "cue_types": ["event", "message"],
                        "decision_payload": {
                            "action": "wait",
                            "pace": None,
                            "reassess_when": "next_interval",
                        },
                    },
                    {
                        "prompt": "llm_call",
                        "repair_status": "fallback_after_repair_failures",
                        "cue_types": ["event"],
                        "decision_payload": {
                            "action": "evacuate",
                            "pace": "running",
                            "reassess_when": "new_cue_only",
                        },
                    },
                ]
            }
        }

        telemetry = build_decision_telemetry(agent_decisions, wait_events=[{"agent": "a1"}])

        self.assertEqual(telemetry["decision_records_total"], 2)
        self.assertEqual(telemetry["wait_decisions"], 1)
        self.assertAlmostEqual(telemetry["wait_decision_rate"], 0.5)
        self.assertEqual(telemetry["fallback_count"], 1)
        self.assertAlmostEqual(telemetry["fallback_rate"], 0.5)
        self.assertEqual(telemetry["action_counts"]["evacuate"], 1)
        self.assertEqual(telemetry["pace_counts"]["running"], 1)
        self.assertEqual(telemetry["cue_type_counts"]["event"], 2)
        self.assertEqual(telemetry["wait_events_recorded"], 1)


if __name__ == "__main__":
    unittest.main()
