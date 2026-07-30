# Function Reference: evacusim/visualization/video_generation_helper.py

- Source file: [evacusim/visualization/video_generation_helper.py](../../../evacusim/visualization/video_generation_helper.py)
- Top-level classes: 2
- Top-level functions: 0

## Module Summary

Video generation utilities for Station Concordia simulations.

## Classes and Methods

### Class RoleColourMap

- Location: [evacusim/visualization/video_generation_helper.py#L47](../../../evacusim/visualization/video_generation_helper.py#L47)
- Summary: Assigns stable (face, edge) colour pairs to agent roles from a palette.
- Method count: 4

#### __init__

- Signature: def __init__(self) -> None
- Location: [evacusim/visualization/video_generation_helper.py#L62](../../../evacusim/visualization/video_generation_helper.py#L62)
- Summary: No docstring summary available.

#### from_roles

- Signature: def from_roles(cls, agent_roles) -> 'RoleColourMap'
- Location: [evacusim/visualization/video_generation_helper.py#L66](../../../evacusim/visualization/video_generation_helper.py#L66)
- Summary: Pre-populate the map with all known roles (sorted for determinism).
- Key calls: cls, sorted, set, agent_roles.values, instance._assign

#### _assign

- Signature: def _assign(self, role) -> int
- Location: [evacusim/visualization/video_generation_helper.py#L74](../../../evacusim/visualization/video_generation_helper.py#L74)
- Summary: Lazily assign the next palette slot to *role* and return its index.
- Key calls: len

#### get

- Signature: def get(self, agent_id, agent_roles) -> tuple[str, str]
- Location: [evacusim/visualization/video_generation_helper.py#L80](../../../evacusim/visualization/video_generation_helper.py#L80)
- Summary: Return ``(face_colour, edge_colour)`` for *agent_id*.
- Key calls: agent_roles.get, self._assign, len

### Class VideoGenerationHelper

- Location: [evacusim/visualization/video_generation_helper.py#L92](../../../evacusim/visualization/video_generation_helper.py#L92)
- Summary: Helper class for generating videos from simulation output.
- Method count: 3

#### load_geometry_from_network

- Signature: def load_geometry_from_network(network_path) -> dict | None
- Location: [evacusim/visualization/video_generation_helper.py#L96](../../../evacusim/visualization/video_generation_helper.py#L96)
- Summary: Load station geometry from SUMO network files.
- Key calls: list, load_walkable_areas, load_entrance_areas, load_platform_areas, load_obstacles, load_escalator_corridors, load_train_entrance_areas, load_train_track_sides, logger.info, level0_file.exists

#### merge_position_history

- Signature: def merge_position_history(decisions_file, history_file) -> Path | None
- Location: [evacusim/visualization/video_generation_helper.py#L184](../../../evacusim/visualization/video_generation_helper.py#L184)
- Summary: Merge position history with decisions data.
- Key calls: logger.info, history_data.get, open, json.load, decisions_data.get, positions_sidecar.exists, json.dump, logger.error, sidecar.get, line.strip

#### generate_simulation_video

- Signature: def generate_simulation_video(decisions_file, run_id, network_path, fps=20, speedup=1.0) -> bool
- Location: [evacusim/visualization/video_generation_helper.py#L252](../../../evacusim/visualization/video_generation_helper.py#L252)
- Summary: Generate video from simulation output.
- Key calls: logger.info, VideoGenerationHelper.load_geometry_from_network, VideoGenerationHelper.merge_position_history, history_file.exists, logger.warning, generate_video_from_output, logger.error, merged_file.exists, merged_file.unlink

## Coverage Notes

- Nested helper functions documented in this file: 0
