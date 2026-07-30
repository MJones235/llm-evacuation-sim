# Function Reference: evacusim/visualization/video_generator.py

- Source file: [evacusim/visualization/video_generator.py](../../../evacusim/visualization/video_generator.py)
- Top-level classes: 1
- Top-level functions: 1

## Module Summary

Video generator for Station Concordia simulations.

## Top-Level Functions

### generate_video_from_output

- Signature: def generate_video_from_output(output_file, video_path=None, geometry=None, fps=20, speedup=1.0, dpi=100) -> bool
- Location: [evacusim/visualization/video_generator.py#L745](../../../evacusim/visualization/video_generator.py#L745)
- Summary: Generate video from simulation output file.
- Key calls: VideoGenerator, generator.generate

## Classes and Methods

### Class VideoGenerator

- Location: [evacusim/visualization/video_generator.py#L25](../../../evacusim/visualization/video_generator.py#L25)
- Summary: Generates MP4 videos from simulation output data.
- Method count: 12

#### __init__

- Signature: def __init__(self, output_file, geometry=None, fps=20, speedup=1.0)
- Location: [evacusim/visualization/video_generator.py#L28](../../../evacusim/visualization/video_generator.py#L28)
- Summary: Initialize video generator.
- Key calls: self._load_data, self.data.get, logger.info, RoleColourMap.from_roles, self._build_level_bounds, self._extract_time_series, set, self._compute_train_rects, ValueError, keys

#### _load_data

- Signature: def _load_data(self) -> dict
- Location: [evacusim/visualization/video_generator.py#L97](../../../evacusim/visualization/video_generator.py#L97)
- Summary: Load simulation output data.
- Key calls: open, json.load, logger.error

#### _build_level_bounds

- Signature: def _build_level_bounds(self) -> dict
- Location: [evacusim/visualization/video_generator.py#L106](../../../evacusim/visualization/video_generator.py#L106)
- Summary: Build bounding box for each level from geometry.
- Key calls: items, geom.get, isinstance, logger.info, areas.values, min, max, coords_list.extend

#### _determine_agent_level

- Signature: def _determine_agent_level(self, agent_id, position) -> str
- Location: [evacusim/visualization/video_generator.py#L141](../../../evacusim/visualization/video_generator.py#L141)
- Summary: Determine which level an agent is on based on position and metadata.
- Key calls: self.level_bounds.items, str, level.startswith, len

#### _extract_time_series

- Signature: def _extract_time_series(self) -> list[dict]
- Location: [evacusim/visualization/video_generator.py#L173](../../../evacusim/visualization/video_generator.py#L173)
- Summary: Extract time series of agent positions and decisions.
- Key calls: logger.info, logger.warning, self.data.get, time_series.append, len, frame.get

#### _setup_figure

- Signature: def _setup_figure(self) -> tuple
- Location: [evacusim/visualization/video_generator.py#L220](../../../evacusim/visualization/video_generator.py#L220)
- Summary: Setup matplotlib figure and axes for multi-level visualization.
- Key calls: plt.subplots, fig.suptitle, ax_level_0.set_title, ax_level_0.set_xlabel, ax_level_0.set_ylabel, ax_level_0.grid, ax_level_0.set_aspect, ax_level_m1.set_title, ax_level_m1.set_xlabel, ax_level_m1.set_ylabel

#### _draw_geometry

- Signature: def _draw_geometry(self, ax, level_name=None)
- Location: [evacusim/visualization/video_generator.py#L259](../../../evacusim/visualization/video_generator.py#L259)
- Summary: Draw station geometry on axes for a specific level.
- Key calls: items, logger.warning, MPLPolygon, ax.add_patch, ax.text, level_name.startswith, name.rsplit, name.startswith, sum, len

#### _set_limits_from_geometry

- Signature: def _set_limits_from_geometry(self, ax, level_name=None)
- Location: [evacusim/visualization/video_generator.py#L366](../../../evacusim/visualization/video_generator.py#L366)
- Summary: Set axis limits from geometry for a specific level.
- Key calls: geom.get, isinstance, ax.set_xlim, ax.set_ylim, logger.warning, areas.values, min, max, level_name.startswith, coords_list.extend

#### _get_train_entrance_areas

- Signature: def _get_train_entrance_areas(self) -> dict[str, list]
- Location: [evacusim/visualization/video_generator.py#L411](../../../evacusim/visualization/video_generator.py#L411)
- Summary: Return train_entrance_areas coord dict from geometry (level -1 only).
- Key calls: geom.get, get

#### _compute_train_rects

- Signature: def _compute_train_rects(self) -> dict[str, dict]
- Location: [evacusim/visualization/video_generator.py#L421](../../../evacusim/visualization/video_generator.py#L421)
- Summary: Pre-compute the on-screen bounding box for each train from geometry.
- Key calls: get, geom_m1.get, train_entrances.items, walkable.get, track_sides.get, logger.debug, exit_name.rsplit, min, max, sum

#### _draw_frame

- Signature: def _draw_frame(self, axes_dict, frame_data, title_text)
- Location: [evacusim/visualization/video_generator.py#L517](../../../evacusim/visualization/video_generator.py#L517)
- Summary: Draw a single frame of the video.
- Key calls: axes_dict.values, title_text.set_text, frame_data.get, positions.items, agent_states.items, set, len, ax.get_children, axes_dict.items, positions.keys

#### generate

- Signature: def generate(self, output_path, dpi=100) -> bool
- Location: [evacusim/visualization/video_generator.py#L708](../../../evacusim/visualization/video_generator.py#L708)
- Summary: Generate video file.
- Key calls: logger.info, self._setup_figure, FFMpegWriter, plt.close, writer.saving, logger.error, str, self._draw_frame, writer.grab_frame

## Coverage Notes

- Nested helper functions documented in this file: 0
