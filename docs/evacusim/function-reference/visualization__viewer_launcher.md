# Function Reference: evacusim/visualization/viewer_launcher.py

- Source file: [evacusim/visualization/viewer_launcher.py](../../../evacusim/visualization/viewer_launcher.py)
- Top-level classes: 1
- Top-level functions: 0

## Module Summary

Viewer launching utilities for Station Concordia simulations.

## Classes and Methods

### Class ViewerLauncher

- Location: [evacusim/visualization/viewer_launcher.py#L19](../../../evacusim/visualization/viewer_launcher.py#L19)
- Summary: Handles launching visualization viewers for simulations.
- Method count: 3

#### launch_viewers

- Signature: def launch_viewers(decisions_file, run_id, network_path, launch_gui=True, launch_spatial=True) -> tuple[subprocess.Popen | None, subprocess.Popen | None]
- Location: [evacusim/visualization/viewer_launcher.py#L23](../../../evacusim/visualization/viewer_launcher.py#L23)
- Summary: Launch visualization viewers for the simulation.
- Key calls: ViewerLauncher.launch_gui_viewer, ViewerLauncher.launch_spatial_viewer

#### launch_gui_viewer

- Signature: def launch_gui_viewer(decisions_file, run_id) -> subprocess.Popen | None
- Location: [evacusim/visualization/viewer_launcher.py#L54](../../../evacusim/visualization/viewer_launcher.py#L54)
- Summary: Launch the GUI viewer for live monitoring.
- Key calls: logger.info, subprocess.Popen, logger.warning, Path, str, decisions_file.absolute

#### launch_spatial_viewer

- Signature: def launch_spatial_viewer(decisions_file, network_path) -> subprocess.Popen | None
- Location: [evacusim/visualization/viewer_launcher.py#L90](../../../evacusim/visualization/viewer_launcher.py#L90)
- Summary: Launch the spatial matplotlib viewer showing agent positions on map.
- Key calls: logger.info, subprocess.Popen, logger.warning, Path, str, decisions_file.absolute

## Coverage Notes

- Nested helper functions documented in this file: 0
