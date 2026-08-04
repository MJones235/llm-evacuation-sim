import json
import tempfile
import unittest
from pathlib import Path

from evacusim.metrics.regression_harness import compare_runs, summarize_run


def _decision(action, pace, reassess_when="next_interval"):
    return {
        "decision_payload": {
            "action": action,
            "pace": pace,
            "reassess_when": reassess_when,
        }
    }


class RegressionHarnessTests(unittest.TestCase):
    def test_summarize_and_compare_runs(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            baseline_path = root / "baseline.json"
            candidate_path = root / "candidate.json"

            baseline = {
                "final_time": 120.0,
                "route_changes": [{"agent": "a1"}],
                "decision_telemetry": {
                    "fallback_rate": 0.0,
                    "repair_status_counts": {"no_repair_needed": 3},
                },
                "agent_decisions": {
                    "a1": {"decisions": [_decision("wait", None), _decision("wait", None)]},
                    "a2": {"decisions": [_decision("evacuate", "hurrying")]},
                },
            }
            candidate = {
                "final_time": 110.0,
                "route_changes": [{"agent": "a1"}, {"agent": "a2"}],
                "decision_telemetry": {
                    "fallback_rate": 0.25,
                    "repair_status_counts": {
                        "no_repair_needed": 2,
                        "fallback_after_repair_failures": 1,
                    },
                },
                "agent_decisions": {
                    "a1": {
                        "decisions": [
                            _decision("wait", None),
                            _decision("continue_activity", "normal_pace"),
                        ]
                    },
                    "a2": {
                        "decisions": [
                            _decision("evacuate", "running", reassess_when="new_cue_only")
                        ]
                    },
                },
            }

            with open(baseline_path, "w") as f:
                json.dump(baseline, f)
            with open(candidate_path, "w") as f:
                json.dump(candidate, f)

            # Sidecar population file should be auto-discovered.
            pop_sidecar = root / "population_timeseries.json"
            with open(pop_sidecar, "w") as f:
                json.dump(
                    {
                        "snapshots": [
                            {"sim_time_s": 0.0, "sim_time_min": 0.0, "concourse": 10, "queue": 2},
                            {"sim_time_s": 60.0, "sim_time_min": 1.0, "concourse": 4, "queue": 1},
                        ]
                    },
                    f,
                )

            baseline_summary = summarize_run(baseline_path)
            self.assertEqual(baseline_summary.decisions_total, 3)
            self.assertEqual(baseline_summary.wait_decisions, 2)

            report = compare_runs(baseline_path, candidate_path)
            self.assertEqual(report["delta"]["final_time_s"], -10.0)
            self.assertEqual(report["delta"]["route_change_count"], 1)
            self.assertAlmostEqual(report["delta"]["fallback_rate"], 0.25)
            self.assertIn("evacuate", report["delta"]["action_mix"])
            self.assertIn("running", report["delta"]["pace_mix"])
            self.assertIn("fallback_after_repair_failures", report["delta"]["repair_status_counts"])
            self.assertIn("concourse", report["delta"]["max_zone_occupancy"])


if __name__ == "__main__":
    unittest.main()
