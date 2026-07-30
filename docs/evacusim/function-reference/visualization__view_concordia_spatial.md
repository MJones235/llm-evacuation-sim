# Function Reference: evacusim/visualization/view_concordia_spatial.py

- Source file: [evacusim/visualization/view_concordia_spatial.py](../../../evacusim/visualization/view_concordia_spatial.py)
- Top-level classes: 1
- Top-level functions: 1

## Module Summary

Spatial viewer for Station Concordia simulation.

## Top-Level Functions

### main

- Signature: def main()
- Location: [evacusim/visualization/view_concordia_spatial.py#L566](../../../evacusim/visualization/view_concordia_spatial.py#L566)
- Summary: Main entry point.
- Key calls: argparse.ArgumentParser, parser.add_argument, parser.parse_args, Path, SpatialConcordiaViewer, viewer.run

## Classes and Methods

### Class SpatialConcordiaViewer

- Location: [evacusim/visualization/view_concordia_spatial.py#L37](../../../evacusim/visualization/view_concordia_spatial.py#L37)
- Summary: Real-time spatial viewer for Concordia simulation.
- Method count: 13

#### __init__

- Signature: def __init__(self, output_file, geometry_file=None, network_path=None)
- Location: [evacusim/visualization/view_concordia_spatial.py#L40](../../../evacusim/visualization/view_concordia_spatial.py#L40)
- Summary: Initialize spatial viewer.
- Key calls: RoleColourMap, set, plt.subplots, self.fig.suptitle, self._setup_map_axes, self.network_path.exists, self._load_geometry_from_network

#### _load_geometry

- Signature: def _load_geometry(self, geometry_file) -> dict
- Location: [evacusim/visualization/view_concordia_spatial.py#L90](../../../evacusim/visualization/view_concordia_spatial.py#L90)
- Summary: Load station geometry from file.
- Key calls: open, json.load, print

#### _load_geometry_from_network

- Signature: def _load_geometry_from_network(self, network_path, level_id='0') -> dict | None
- Location: [evacusim/visualization/view_concordia_spatial.py#L99](../../../evacusim/visualization/view_concordia_spatial.py#L99)
- Summary: Load station geometry from SUMO network files for a specific level.
- Key calls: level_file.exists, load_walkable_areas, load_entrance_areas, load_platform_areas, load_obstacles, load_escalator_corridors, load_train_entrance_areas, print, str, list

#### _setup_map_axes

- Signature: def _setup_map_axes(self)
- Location: [evacusim/visualization/view_concordia_spatial.py#L169](../../../evacusim/visualization/view_concordia_spatial.py#L169)
- Summary: Setup the map visualization axes for both levels.
- Key calls: self.ax_level_0.set_title, self.ax_level_0.set_xlabel, self.ax_level_0.set_ylabel, self.ax_level_0.grid, self.ax_level_0.set_aspect, self.ax_level_m1.set_title, self.ax_level_m1.set_xlabel, self.ax_level_m1.set_ylabel, self.ax_level_m1.grid, self.ax_level_m1.set_aspect

#### _draw_geometry

- Signature: def _draw_geometry(self, ax, geometry)
- Location: [evacusim/visualization/view_concordia_spatial.py#L194](../../../evacusim/visualization/view_concordia_spatial.py#L194)
- Summary: Draw station geometry on a specific axes.
- Key calls: items, MPLPolygon, ax.add_patch, ax.text, name.rsplit, name.startswith, sum, len

#### _set_fixed_limits

- Signature: def _set_fixed_limits(self, ax, x_min, x_max, y_min, y_max)
- Location: [evacusim/visualization/view_concordia_spatial.py#L294](../../../evacusim/visualization/view_concordia_spatial.py#L294)
- Summary: Set fixed axis limits with small padding.
- Key calls: ax.set_xlim, ax.set_ylim

#### _set_fixed_limits_from_geometry

- Signature: def _set_fixed_limits_from_geometry(self, ax, geometry)
- Location: [evacusim/visualization/view_concordia_spatial.py#L301](../../../evacusim/visualization/view_concordia_spatial.py#L301)
- Summary: Compute geometry bounds and lock axis limits.
- Key calls: isinstance, self._set_fixed_limits, geometry.get, areas.values, min, max, coords_list.extend

#### _update_data

- Signature: def _update_data(self)
- Location: [evacusim/visualization/view_concordia_spatial.py#L320](../../../evacusim/visualization/view_concordia_spatial.py#L320)
- Summary: Load latest data from output file.
- Key calls: self.output_file.exists, self.output_file.with_name, sidecar.exists, self.title_text.set_text, time.time, open, json.load, RoleColourMap.from_roles

#### _update_visualization

- Signature: def _update_visualization(self, frame)
- Location: [evacusim/visualization/view_concordia_spatial.py#L397](../../../evacusim/visualization/view_concordia_spatial.py#L397)
- Summary: Update visualization with latest data.
- Key calls: self._update_agent_positions, self._update_blocked_exits, self._update_data

#### _update_blocked_exits

- Signature: def _update_blocked_exits(self)
- Location: [evacusim/visualization/view_concordia_spatial.py#L409](../../../evacusim/visualization/view_concordia_spatial.py#L409)
- Summary: Draw visual markers for blocked exits (Phase 4.
- Key calls: self.geometry_level_0.get, marker.remove, self.ax_level_0.text, self.blocked_exit_markers.extend, sum, len, self.ax_level_0.plot

#### _agent_colour

- Signature: def _agent_colour(self, agent_id) -> tuple[str, str]
- Location: [evacusim/visualization/view_concordia_spatial.py#L459](../../../evacusim/visualization/view_concordia_spatial.py#L459)
- Summary: Return (face_colour, edge_colour) for *agent_id* via the shared palette map.
- Key calls: self._colour_map.get

#### _update_agent_positions

- Signature: def _update_agent_positions(self)
- Location: [evacusim/visualization/view_concordia_spatial.py#L463](../../../evacusim/visualization/view_concordia_spatial.py#L463)
- Summary: Update agent position markers on both levels.
- Key calls: self.agent_dots.values, self.agent_labels.values, set, self.agent_positions.items, len, dot.remove, label.remove, self.agent_positions.keys, join, self._status_text.set_text

#### run

- Signature: def run(self)
- Location: [evacusim/visualization/view_concordia_spatial.py#L548](../../../evacusim/visualization/view_concordia_spatial.py#L548)
- Summary: Run the viewer with animation.
- Key calls: print, FuncAnimation, plt.tight_layout, plt.show

## Coverage Notes

- Nested helper functions documented in this file: 0
