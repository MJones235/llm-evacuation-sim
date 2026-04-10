"""
Viewer launching utilities for Station Concordia simulations.

This module is responsible for:
- Launching GUI viewers for live monitoring
- Launching spatial matplotlib viewers
- Managing viewer subprocess lifecycle
"""

import subprocess
import sys
from pathlib import Path

from evacusim.utils.logger import get_logger

logger = get_logger(__name__)


class ViewerLauncher:
    """Handles launching visualization viewers for simulations."""

    @staticmethod
    def launch_viewers(
        decisions_file: Path,
        run_id: str,
        network_path: Path,
        launch_gui: bool = True,
        launch_spatial: bool = True,
    ) -> tuple[subprocess.Popen | None, subprocess.Popen | None]:
        """
        Launch visualization viewers for the simulation.

        Args:
            decisions_file: Path to the agent decisions JSON file
            run_id: Unique identifier for this simulation run
            network_path: Path to the station network directory
            launch_gui: Whether to launch the GUI viewer
            launch_spatial: Whether to launch the spatial viewer

        Returns:
            Tuple of (gui_process, spatial_process)
        """
        gui_process = None
        if launch_gui:
            gui_process = ViewerLauncher.launch_gui_viewer(decisions_file, run_id)

        spatial_process = None
        if launch_spatial:
            spatial_process = ViewerLauncher.launch_spatial_viewer(decisions_file, network_path)

        return gui_process, spatial_process

    @staticmethod
    def launch_gui_viewer(
        decisions_file: Path,
        run_id: str,
    ) -> subprocess.Popen | None:
        """
        Launch the GUI viewer for live monitoring.

        Args:
            decisions_file: Path to the agent decisions JSON file
            run_id: Unique identifier for this simulation run

        Returns:
            Popen process object if successful, None if failed
        """
        logger.info("Launching GUI viewer for live monitoring...")
        try:
            viewer_path = (
                Path(__file__).parent / "view_concordia_gui.py"
            )
            process = subprocess.Popen(
                [
                    sys.executable,
                    str(viewer_path),
                    "--output-file",
                    str(decisions_file.absolute()),
                    "--run-id",
                    run_id,
                ]
            )
            logger.info("GUI viewer launched - it will update as simulation runs")
            return process
        except Exception as e:
            logger.warning(f"Failed to launch GUI viewer: {e}")
            return None

    @staticmethod
    def launch_spatial_viewer(
        decisions_file: Path,
        network_path: Path,
    ) -> subprocess.Popen | None:
        """
        Launch the spatial matplotlib viewer showing agent positions on map.

        Args:
            decisions_file: Path to the agent decisions JSON file
            network_path: Path to the station network directory

        Returns:
            Popen process object if successful, None if failed
        """
        logger.info("Launching spatial matplotlib viewer...")
        try:
            spatial_viewer_path = (
                Path(__file__).parent / "view_concordia_spatial.py"
            )
            process = subprocess.Popen(
                [
                    sys.executable,
                    str(spatial_viewer_path),
                    "--output-file",
                    str(decisions_file.absolute()),
                    "--network-path",
                    str(network_path),
                ]
            )
            logger.info("Spatial viewer launched - shows agent positions on map")
            return process
        except Exception as e:
            logger.warning(f"Failed to launch spatial viewer: {e}")
            return None
