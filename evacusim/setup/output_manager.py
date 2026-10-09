"""
Output directory management for Station Concordia simulations.

This module is responsible for:
- Creating unique run directories
- Setting up output file paths
- Configuring environment variables for logging
"""

import os
from datetime import datetime
from pathlib import Path

from evacusim.config.schema import OutputConfig
from evacusim.utils.logger import get_logger

logger = get_logger(__name__)


class OutputManager:
    """Handles output directory and file management for simulation runs."""

    @staticmethod
    def setup_output_directory(
        output: OutputConfig, engine: str = "", seed: int | None = None
    ) -> tuple[str, Path, Path]:
        """
        Create a new, uniquely named directory for this run.

        The name is ``run_<YYYYmmdd_HHMMSS>[_<engine>][_s<seed>]``, with a
        ``_2``, ``_3``... suffix if a run started in the same second already
        took it. Also points the LLM prompt log at the directory.

        Args:
            output: Output settings (the ``output`` section)
            engine: Decision engine name, included in the directory name.
            seed: Run seed, included in the directory name.

        Returns:
            Tuple of (run_id, output_dir, decisions_file)
            - run_id: The directory name, e.g. "run_20261009_143022_rule_based_s0"
            - output_dir: Path to the run's output directory
            - decisions_file: Path to the agent decisions JSON file
        """
        base_id = datetime.now().strftime("run_%Y%m%d_%H%M%S")
        if engine:
            base_id += f"_{engine}"
        if seed is not None:
            base_id += f"_s{seed}"
        root = Path(output.directory)
        root.mkdir(parents=True, exist_ok=True)
        run_id, n = base_id, 1
        while True:
            try:
                (root / run_id).mkdir()
                break
            except FileExistsError:
                n += 1
                run_id = f"{base_id}_{n}"
        output_dir = root / run_id
        decisions_file = output_dir / "agent_decisions.json"

        # Configure LLM logging environment variable
        os.environ["CONCORDIA_LLM_LOG_PATH"] = str(output_dir / "llm_prompt_log.jsonl")

        logger.info(f"Run ID: {run_id}")
        logger.info(f"Output directory: {output_dir}")

        return run_id, output_dir, decisions_file
