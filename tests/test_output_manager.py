"""Run directories: named by time, engine and seed, and never shared."""

import os
import tempfile
import unittest
from pathlib import Path

from evacusim.config.schema import OutputConfig
from evacusim.setup.output_manager import OutputManager


class OutputManagerTests(unittest.TestCase):
    def setUp(self):
        self._log_path = os.environ.get("CONCORDIA_LLM_LOG_PATH")

    def tearDown(self):
        if self._log_path is None:
            os.environ.pop("CONCORDIA_LLM_LOG_PATH", None)
        else:
            os.environ["CONCORDIA_LLM_LOG_PATH"] = self._log_path

    def test_runs_started_together_get_separate_directories(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = OutputConfig(directory=tmp)
            dirs = {
                OutputManager.setup_output_directory(output, "rule_based", 7)[1] for _ in range(3)
            }
            self.assertEqual(len(dirs), 3)
            for d in dirs:
                self.assertTrue(Path(d).is_dir())
                self.assertIn("_rule_based_s7", d.name)
