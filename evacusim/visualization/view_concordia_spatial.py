#!/usr/bin/env python3
"""
Spatial viewer for Station Concordia simulation.

Shows agent positions on station map in real-time alongside their decisions.
Requires matplotlib for visualization.

Usage:
    python tools/view_concordia_spatial.py --output-file PATH --geometry PATH
    python tools/view_concordia_spatial.py --output-file PATH --network-path PATH
"""

import argparse
import json
import sys
import time
from pathlib import Path

# Setup path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from matplotlib.patches import Polygon as MPLPolygon

    matplotlib.use("TkAgg")
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not available. Install with: pip install matplotlib")

from evacusim.visualization.video_generation_helper import RoleColourMap


class SpatialConcordiaViewer:
    """Real-time spatial viewer for Concordia simulation."""

    def __init__(
        self,
        output_file: Path,
        geometry_file: Path | None = None,
        network_path: Path | None = None,
    ):
        """Initialize spatial viewer."""
        self.output_file = output_file
        self.geometry_file = geometry_file
        self.network_path = network_path
        self.agent_positions = {}
        self.agent_decisions = {}
        self.agent_levels = {}  # Track which level each agent is on
        self.agent_roles: dict[str, str] = {}  # agent_id -> role label for director agents
        self._colour_map: RoleColourMap = RoleColourMap()
        self.last_update = 0
        self.blocked_exits = []  # Phase 4.2: Track blocked exits for visualization
        # Train boarding tracking: count agents who disappeared while their
        # last destination was a train_platform_* exit.
        self._initial_agents: set[str] = set()
        self._last_known_level: dict[str, str] = {}
        self._boarded_agents: set[str] = set()
        self._status_text = None       # single compact status badge on platform panel
        self.active_train_exits: list[str] = []  # populated from sidecar

        # Load geometry for both levels
        self.geometry_level_0 = None
        self.geometry_level_m1 = None
        if self.network_path and self.network_path.exists():
            self.geometry_level_0 = self._load_geometry_from_network(self.network_path, "0")
            self.geometry_level_m1 = self._load_geometry_from_network(self.network_path, "-1")

        # Setup matplotlib figure with 2 map panels (no decision log)
        self.fig, (self.ax_level_0, self.ax_level_m1) = plt.subplots(
            1, 2, figsize=(18, 9), gridspec_kw={"width_ratios": [1, 1]}
        )

        self.title_text = self.fig.suptitle(
            "Concordia Station Evacuation - Multi-Level View | Time: 0.0s", fontsize=14
        )
        self.current_time = 0.0

        self._setup_map_axes()

        # Agent visual elements per level
        self.agent_dots = {}
        self.agent_labels = {}
        self.blocked_exit_markers = []  # Phase 4.2: Visual markers for blocked exits
        self.current_data = {}  # Store current simulation data

    def _load_geometry(self, geometry_file: Path) -> dict:
        """Load station geometry from file."""
        try:
            with open(geometry_file) as f:
                return json.load(f)
        except Exception as e:
            print(f"Failed to load geometry: {e}")
            return None

    def _load_geometry_from_network(self, network_path: Path, level_id: str = "0") -> dict | None:
        """Load station geometry from SUMO network files for a specific level."""
        try:
            from evacusim.jps.geometry_loader import (
                load_entrance_areas,
                load_escalator_corridors,
                load_obstacles,
                load_platform_areas,
                load_train_entrance_areas,
                load_walkable_areas,
            )

            level_file = network_path / f"level_{level_id}.xml"
            walking_areas_file = network_path / "walking_areas.add.xml"

            if level_file.exists():
                geom_file = level_file
            elif level_id == "0" and walking_areas_file.exists():
                geom_file = walking_areas_file
            else:
                print(
                    f"Geometry file not found for level {level_id}: expected level_{level_id}.xml "
                    f"in {network_path}"
                )
                return None

            walkable_areas = load_walkable_areas(str(geom_file))
            entrance_areas = load_entrance_areas(str(geom_file))
            platform_areas = load_platform_areas(str(geom_file))
            obstacles = load_obstacles(str(geom_file))
            escalator_corridors = load_escalator_corridors(str(geom_file))
            train_entrance_areas = load_train_entrance_areas(str(geom_file))

            def poly_to_coords(poly):
                return list(poly.exterior.coords)

            geometry = {
                "walkable_areas": {
                    name: poly_to_coords(poly) for name, poly in walkable_areas.items()
                },
                "entrance_areas": {
                    name: poly_to_coords(poly) for name, poly in entrance_areas.items()
                },
                "platform_areas": {
                    name: poly_to_coords(poly) for name, poly in platform_areas.items()
                },
                "obstacles": [poly_to_coords(poly) for poly in obstacles],
                "escalator_corridors": {
                    name: poly_to_coords(poly) for name, poly in escalator_corridors.items()
                },
                "train_entrance_areas": {
                    name: poly_to_coords(poly) for name, poly in train_entrance_areas.items()
                },
            }

            print(
                f"Loaded geometry from {geom_file} (level {level_id}): "
                f"{len(walkable_areas)} walkable, "
                f"{len(entrance_areas)} entrances, "
                f"{len(platform_areas)} platforms, "
                f"{len(obstacles)} obstacles, "
                f"{len(escalator_corridors)} escalator corridors, "
                f"{len(train_entrance_areas)} train boarding zones"
            )

            return geometry
        except Exception as e:
            print(f"Failed to load geometry from network for level {level_id}: {e}")
            return None

    def _setup_map_axes(self):
        """Setup the map visualization axes for both levels."""
        # Setup Level 0 (Concourse)
        self.ax_level_0.set_title("Level 0 - Concourse", fontsize=12, weight="bold")
        self.ax_level_0.set_xlabel("X Position (m)")
        self.ax_level_0.set_ylabel("Y Position (m)")
        self.ax_level_0.grid(True, alpha=0.3)
        self.ax_level_0.set_aspect("equal")

        # Setup Level -1 (Platforms)
        self.ax_level_m1.set_title("Level -1 - Platforms", fontsize=12, weight="bold")
        self.ax_level_m1.set_xlabel("X Position (m)")
        self.ax_level_m1.set_ylabel("Y Position (m)")
        self.ax_level_m1.grid(True, alpha=0.3)
        self.ax_level_m1.set_aspect("equal")

        # Draw geometry for both levels
        if self.geometry_level_0:
            self._draw_geometry(self.ax_level_0, self.geometry_level_0)
            self._set_fixed_limits_from_geometry(self.ax_level_0, self.geometry_level_0)

        if self.geometry_level_m1:
            self._draw_geometry(self.ax_level_m1, self.geometry_level_m1)
            self._set_fixed_limits_from_geometry(self.ax_level_m1, self.geometry_level_m1)

    def _draw_geometry(self, ax, geometry):
        """Draw station geometry on a specific axes."""
        if not geometry:
            return

        # Draw walkable areas
        if "walkable_areas" in geometry:
            for _, coords in geometry["walkable_areas"].items():
                if coords:
                    polygon = MPLPolygon(
                        coords, fill=True, alpha=0.2, color="gray", label="Walkable"
                    )
                    ax.add_patch(polygon)

        # Draw entrances/exits
        if "entrance_areas" in geometry:
            for _, coords in geometry["entrance_areas"].items():
                if coords:
                    polygon = MPLPolygon(coords, fill=True, alpha=0.3, color="green", label="Exit")
                    ax.add_patch(polygon)

        # Draw platforms
        if "platform_areas" in geometry:
            for _, coords in geometry["platform_areas"].items():
                if coords:
                    polygon = MPLPolygon(
                        coords, fill=True, alpha=0.3, color="blue", label="Platform"
                    )
                    ax.add_patch(polygon)

        # Draw escalator corridors
        if "escalator_corridors" in geometry:
            for _, coords in geometry["escalator_corridors"].items():
                if coords:
                    polygon = MPLPolygon(
                        coords,
                        fill=True,
                        alpha=0.35,
                        facecolor="#FF8C00",
                        edgecolor="#FF4500",
                        linewidth=1.2,
                        label="Escalator",
                    )
                    ax.add_patch(polygon)

        # Draw obstacles
        if "obstacles" in geometry:
            for coords in geometry["obstacles"]:
                if coords:
                    polygon = MPLPolygon(
                        coords, fill=True, alpha=0.4, color="black", label="Obstacle"
                    )
                    ax.add_patch(polygon)

        # Draw train boarding zones (jupedsim.train_entrance polygons on level -1).
        # These are small markers at the end of each platform; draw them as
        # solid green rectangles without a text label — the platform number is
        # instead shown on the much larger platform walkable area below.
        if "train_entrance_areas" in geometry:
            for name, coords in geometry["train_entrance_areas"].items():
                if coords:
                    polygon = MPLPolygon(
                        coords,
                        fill=True,
                        alpha=0.8,
                        facecolor="#00CC44",
                        edgecolor="white",
                        linewidth=1.5,
                        label="Train",
                        zorder=5,
                    )
                    ax.add_patch(polygon)

        # Label each named platform walkable area (platform_1, platform_2, …)
        # with a large number so it is easy to read on the map at any zoom level.
        if "walkable_areas" in geometry:
            for name, coords in geometry["walkable_areas"].items():
                if not name.startswith("platform_") or not coords:
                    continue
                platform_num = name.rsplit("_", 1)[-1]
                outline = MPLPolygon(
                    coords,
                    fill=False,
                    edgecolor="#CC6600",
                    linewidth=1.5,
                    linestyle="--",
                    zorder=4,
                )
                ax.add_patch(outline)
                xs = [c[0] for c in coords]
                ys = [c[1] for c in coords]
                cx, cy = sum(xs) / len(xs), sum(ys) / len(ys)
                ax.text(
                    cx, cy, f"P{platform_num}",
                    ha="center", va="center",
                    fontsize=11, color="#994400", fontweight="bold",
                    clip_on=True,
                    zorder=6,
                )

    def _set_fixed_limits(self, ax, x_min: float, x_max: float, y_min: float, y_max: float):
        """Set fixed axis limits with small padding."""
        pad_x = (x_max - x_min) * 0.05 if x_max > x_min else 5.0
        pad_y = (y_max - y_min) * 0.05 if y_max > y_min else 5.0
        ax.set_xlim(x_min - pad_x, x_max + pad_x)
        ax.set_ylim(y_min - pad_y, y_max + pad_y)

    def _set_fixed_limits_from_geometry(self, ax, geometry):
        """Compute geometry bounds and lock axis limits."""
        coords_list = []
        for key in ("walkable_areas", "entrance_areas", "platform_areas", "obstacles"):
            areas = geometry.get(key, {}) if geometry else {}
            if isinstance(areas, dict):
                for coords in areas.values():
                    if coords:
                        coords_list.extend(coords)
            elif isinstance(areas, list):
                for coords in areas:
                    if coords:
                        coords_list.extend(coords)

        if coords_list:
            xs = [c[0] for c in coords_list]
            ys = [c[1] for c in coords_list]
            self._set_fixed_limits(ax, min(xs), max(xs), min(ys), max(ys))

    def _update_data(self):
        """Load latest data from output file.

        Positions, time, levels and blocked exits are read from the lightweight
        ``_positions.json`` sidecar (written every 0.5 s) when it exists.
        Decisions and messages are read from the main output file (written every
        10 s) so the viewer refreshes smoothly without waiting for heavy I/O.
        """
        if not self.output_file.exists():
            return False

        try:
            # --- Lightweight sidecar: positions, time, levels, blocked exits ---
            sidecar = self.output_file.with_name(
                self.output_file.stem + "_positions.json"
            )
            if sidecar.exists():
                with open(sidecar) as f:
                    pos_data = json.load(f)
                if "agent_positions" in pos_data:
                    self.agent_positions = pos_data["agent_positions"]
                if "agent_levels" in pos_data:
                    self.agent_levels = pos_data["agent_levels"]
                if "blocked_exits" in pos_data:
                    self.blocked_exits = pos_data["blocked_exits"]
                if "current_time" in pos_data:
                    self.current_time = pos_data["current_time"]
                if "agent_roles" in pos_data:
                    self.agent_roles = pos_data["agent_roles"]
                    self._colour_map = RoleColourMap.from_roles(self.agent_roles)
                if "active_train_exits" in pos_data:
                    self.active_train_exits = pos_data["active_train_exits"]
            else:
                # Sidecar not yet written — fall back to main file for positions.
                with open(self.output_file) as f:
                    data = json.load(f)
                if "agent_positions" in data:
                    self.agent_positions = data["agent_positions"]
                if "agent_levels" in data:
                    self.agent_levels = data["agent_levels"]
                if "blocked_exits" in data:
                    self.blocked_exits = data["blocked_exits"]
                if "current_time" in data:
                    self.current_time = data["current_time"]
                if "agent_roles" in data:
                    self.agent_roles = data["agent_roles"]
                    self._colour_map = RoleColourMap.from_roles(self.agent_roles)
                elif "final_time" in data:
                    self.current_time = data["final_time"]

            # --- Main file: decisions, messages (updated less frequently) ---
            with open(self.output_file) as f:
                data = json.load(f)

            if "agent_decisions" in data:
                self.agent_decisions = data["agent_decisions"]

            # Prefer sidecar time if already set above; otherwise use main file.
            if not sidecar.exists():
                pass  # already handled in fallback branch above
            elif "final_time" in data and self.current_time == 0.0:
                self.current_time = data["final_time"]

            # Update title with time
            self.title_text.set_text(
                f"Concordia Station Evacuation - Real-Time View | Time: {self.current_time:.1f}s"
            )

            # Store the full data for use in other methods
            self.current_data = data

            self.last_update = time.time()
            return True

        except (OSError, json.JSONDecodeError):
            return False

    def _update_visualization(self, frame):
        """Update visualization with latest data."""
        # Load new data
        if not self._update_data():
            return

        # Update agent positions on both levels
        self._update_agent_positions()

        # Update blocked exit markers (Phase 4.2)
        self._update_blocked_exits()

    def _update_blocked_exits(self):
        """Draw visual markers for blocked exits (Phase 4.2)."""
        # Remove old markers
        for marker in self.blocked_exit_markers:
            marker.remove()
        self.blocked_exit_markers = []

        if not self.blocked_exits or not self.geometry_level_0:
            return

        # Get entrance areas from level 0 geometry (exits are on concourse level)
        entrance_areas = self.geometry_level_0.get("entrance_areas", {})

        for exit_name in self.blocked_exits:
            if exit_name in entrance_areas:
                # Get exit polygon coordinates
                coords = entrance_areas[exit_name]
                if coords:
                    # Calculate centroid
                    xs = [c[0] for c in coords]
                    ys = [c[1] for c in coords]
                    center_x = sum(xs) / len(xs)
                    center_y = sum(ys) / len(ys)

                    # Draw red X over the exit on level 0
                    size = 8
                    marker1 = self.ax_level_0.plot(
                        [center_x - size, center_x + size],
                        [center_y - size, center_y + size],
                        "r-",
                        linewidth=4,
                    )[0]
                    marker2 = self.ax_level_0.plot(
                        [center_x - size, center_x + size],
                        [center_y + size, center_y - size],
                        "r-",
                        linewidth=4,
                    )[0]
                    label = self.ax_level_0.text(
                        center_x,
                        center_y - size - 3,
                        "🚧 BLOCKED",
                        ha="center",
                        fontsize=10,
                        color="red",
                        weight="bold",
                    )

                    self.blocked_exit_markers.extend([marker1, marker2, label])

    def _agent_colour(self, agent_id: str) -> tuple[str, str]:
        """Return (face_colour, edge_colour) for *agent_id* via the shared palette map."""
        return self._colour_map.get(agent_id, self.agent_roles)

    def _update_agent_positions(self):
        """Update agent position markers on both levels."""
        # Remove old dots
        for dot in self.agent_dots.values():
            dot.remove()
        for label in self.agent_labels.values():
            label.remove()

        self.agent_dots = {}
        self.agent_labels = {}

        # Draw new positions
        if not self.agent_positions:
            return  # No positions yet

        # Seed initial agent set on first populated frame
        if not self._initial_agents and self.agent_positions:
            self._initial_agents = set(self.agent_positions.keys())

        current_agents = set(self.agent_positions.keys())

        for agent_id, pos in self.agent_positions.items():
            if pos and len(pos) >= 2:
                x, y = pos[0], pos[1]  # Handle both list and tuple from JSON

                # Determine which level this agent is on
                agent_level = self.agent_levels.get(agent_id, "0")  # Default to level 0
                self._last_known_level[agent_id] = agent_level
                ax = self.ax_level_0 if agent_level == "0" else self.ax_level_m1

                face, edge = self._agent_colour(agent_id)
                is_director = bool(self.agent_roles.get(agent_id))
                size = 10 if is_director else 8
                dot = ax.plot(
                    x, y, "o",
                    color=face,
                    markeredgecolor=edge,
                    markeredgewidth=1.5,
                    markersize=size,
                )[0]
                label = ax.text(x, y + 1, agent_id, ha="center", fontsize=8)

                self.agent_dots[agent_id] = dot
                self.agent_labels[agent_id] = label

        # Detect newly-boarded agents: vanished from positions while on level -1
        # (i.e. they walked into a train exit polygon and were removed by JuPedSim).
        for agent_id in self._initial_agents - current_agents - self._boarded_agents:
            if self._last_known_level.get(agent_id) == "-1":
                self._boarded_agents.add(agent_id)

        # Update compact train/boarding status badge on the platform panel.
        boarded = len(self._boarded_agents)
        if self.active_train_exits:
            platforms = " ".join(
                f"P{n.rsplit('_',1)[-1]}"
                for n in sorted(self.active_train_exits)
            )
            status_str = f"🚂 Boarding: {platforms}   Boarded: {boarded}"
            status_color = "#006600"
            face_color = "#CCFFCC"
        elif boarded > 0:
            status_str = f"🚫 Train departed   Boarded: {boarded}"
            status_color = "#555555"
            face_color = "#F0F0F0"
        else:
            status_str = "Train not yet arrived"
            status_color = "#555555"
            face_color = "#F8F8F8"

        if self._status_text is not None:
            self._status_text.set_text(status_str)
            self._status_text.set_color(status_color)
            self._status_text.get_bbox_patch().set_facecolor(face_color)
        else:
            self._status_text = self.ax_level_m1.text(
                0.02, 0.97,
                status_str,
                transform=self.ax_level_m1.transAxes,
                ha="left", va="top",
                fontsize=8, color=status_color,
                bbox=dict(boxstyle="round,pad=0.25", facecolor=face_color, alpha=0.85),
                zorder=5,
            )

    def run(self):
        """Run the viewer with animation."""
        if not MATPLOTLIB_AVAILABLE:
            print("matplotlib not available - cannot run spatial viewer")
            return

        print(f"Monitoring: {self.output_file}")
        print("Close window to exit...")

        # Setup animation - update every 500ms to match file write frequency
        self.ani = FuncAnimation(
            self.fig, self._update_visualization, interval=500, cache_frame_data=False
        )

        plt.tight_layout()
        plt.show()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Spatial viewer for Concordia simulation")
    parser.add_argument(
        "--output-file", type=str, required=True, help="Path to agent decisions JSON file"
    )
    parser.add_argument(
        "--geometry", type=str, default=None, help="Path to geometry JSON file (optional)"
    )
    parser.add_argument(
        "--network-path",
        type=str,
        default=None,
        help="Path to station_sim network directory (optional)",
    )
    args = parser.parse_args()

    output_file = Path(args.output_file)
    geometry_file = Path(args.geometry) if args.geometry else None
    network_path = Path(args.network_path) if args.network_path else None

    viewer = SpatialConcordiaViewer(output_file, geometry_file, network_path)
    viewer.run()


if __name__ == "__main__":
    main()
