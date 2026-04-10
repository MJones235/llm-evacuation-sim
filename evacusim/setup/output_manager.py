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

from evacusim.utils.logger import get_logger

logger = get_logger(__name__)


class OutputManager:
    """Handles output directory and file management for simulation runs."""

    @staticmethod
    def setup_output_directory(config: dict) -> tuple[str, Path, Path]:
        """
        Setup output directory structure for a simulation run.

        Creates a unique run directory with timestamp and sets up file paths.
        Also configures environment variables for LLM logging.

        Args:
            config: Configuration dictionary containing output settings

        Returns:
            Tuple of (run_id, output_dir, decisions_file)
            - run_id: Unique identifier string (e.g., "run_20260209_143022")
            - output_dir: Path to the run's output directory
            - decisions_file: Path to the agent decisions JSON file
        """
        # Generate unique run ID
        run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S")

        # Setup directory structure
        output_config = config.get("output", {})
        base_output_dir = Path(output_config.get("directory", "scenarios/station_concordia/output"))
        output_dir = base_output_dir / run_id
        output_dir.mkdir(parents=True, exist_ok=True)

        # Setup file paths
        decisions_file = output_dir / "agent_decisions.json"

        # Configure LLM logging environment variable
        os.environ["CONCORDIA_LLM_LOG_PATH"] = str(output_dir / "llm_prompt_log.jsonl")

        logger.info(f"Run ID: {run_id}")
        logger.info(f"Output directory: {output_dir}")

        return run_id, output_dir, decisions_file
