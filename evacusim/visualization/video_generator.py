"""
Video generator for Station Concordia simulations.

This module creates MP4 videos from simulation output data.
Videos show agent positions and decisions at regular time intervals,
without delays for LLM responses.
"""

import json
import math
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from matplotlib.patches import Polygon as MPLPolygon

from evacusim.utils.logger import get_logger
from evacusim.visualization.train_geometry import compute_train_polygons
from evacusim.visualization.video_generation_helper import RoleColourMap

logger = get_logger(__name__)

matplotlib.use("Agg")  # Non-interactive backend for video generation


def _fmt_clock(seconds: float) -> str:
    """Format a sim-time offset (seconds since midnight) as a HH:MM:SS clock.

    The simulation clock starts at midnight (t=0), so the elapsed sim time is
    already the time of day.  Values >= 24 h wrap via modulo so a slightly
    over-length run still reads as a clock.
    """
    s = int(seconds) % 86400
    return f"{s // 3600:02d}:{(s % 3600) // 60:02d}:{s % 60:02d}"


class VideoGenerator:
    """Generates MP4 videos from simulation output data."""

    def __init__(
        self,
        output_file: Path,
        geometry: dict | None = None,
        fps: int = 20,
        speedup: float = 1.0,
    ):
        """
        Initialize video generator.

        Args:
            output_file: Path to agent decisions JSON file
            geometry: Station geometry dict (or None to load from data)
            fps: Frames per second for output video
            speedup: Speed multiplier (1.0 = real-time, 2.0 = 2x speed)
        """
        self.output_file = output_file
        self.geometry = geometry
        self.fps = fps
        self.speedup = speedup

        # Load simulation data
        self.data = self._load_data()
        if not self.data:
            raise ValueError(f"Could not load data from {output_file}")

        # Extract agent levels mapping
        self.agent_levels = self.data.get("agent_levels", {})
        logger.info(f"Loaded agent levels for {len(self.agent_levels)} agents")

        # Extract agent roles (director agents have a non-empty role label)
        self.agent_roles: dict[str, str] = self.data.get("agent_roles", {})
        if self.agent_roles:
            logger.info(f"Loaded roles for {len(self.agent_roles)} director agent(s)")
        self._colour_map = RoleColourMap.from_roles(self.agent_roles)

        # Escalator conveyor geometry (written next to the results); riders are
        # drawn on their own panel and projected onto the floor plans.
        self.escalator_geometry: dict = {}
        esc_path = Path(output_file).parent / "escalators.json"
        if esc_path.exists():
            with open(esc_path) as f:
                self.escalator_geometry = json.load(f)

        # Build level bounds from geometry for coordinate-based inference
        self.level_bounds = self._build_level_bounds()

        # Extract time series data
        self.time_series = self._extract_time_series()
        if not self.time_series:
            raise ValueError("No position data found in output file")

        # Pre-computed train polygons (one per platform) for visualisation.
        # Keyed by exit name, e.g. "train_platform_1".
        self.train_polygons = compute_train_polygons(self.geometry or {})
        if self.train_polygons:
            logger.info(f"Pre-computed train polygons for: {list(self.train_polygons.keys())}")

        logger.info(
            f"Loaded {len(self.time_series)} time steps "
            f"from {self.data.get('current_time', 0):.1f}s simulation"
        )

    def _load_data(self) -> dict:
        """Load simulation output data."""
        try:
            with open(self.output_file) as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            return {}

    def _build_level_bounds(self) -> dict:
        """Build bounding box for each level from geometry."""
        bounds = {}

        if not self.geometry or "levels" not in self.geometry:
            return bounds

        for level_name, geom in self.geometry["levels"].items():
            coords_list = []
            for key in ("walkable_areas", "entrance_areas", "platform_areas", "obstacles"):
                areas = geom.get(key, {})
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
                bounds[level_name] = {
                    "x_min": min(xs),
                    "x_max": max(xs),
                    "y_min": min(ys),
                    "y_max": max(ys),
                }
                logger.info(
                    f"Level {level_name} bounds: X[{bounds[level_name]['x_min']:.1f}, {bounds[level_name]['x_max']:.1f}], Y[{bounds[level_name]['y_min']:.1f}, {bounds[level_name]['y_max']:.1f}]"
                )

        return bounds

    def _determine_agent_level(self, agent_id: str, position: list) -> str:
        """
        Determine which level an agent is on based on position and metadata.

        First checks agent_levels dict, then falls back to coordinate-based inference.
        """
        # First, check if we have explicit level info
        if agent_id in self.agent_levels:
            level = str(self.agent_levels[agent_id])
            return level if level.startswith("level_") else f"level_{level}"

        # Fall back to coordinate-based inference
        if not position or len(position) < 2 or not self.level_bounds:
            return "level_0"  # Default to level 0

        x, y = position[0], position[1]

        # Check which level's bounds contain this position
        for level_name, bounds in self.level_bounds.items():
            # Use generous padding for boundary check
            x_pad = (bounds["x_max"] - bounds["x_min"]) * 0.1
            y_pad = (bounds["y_max"] - bounds["y_min"]) * 0.1

            if (
                bounds["x_min"] - x_pad <= x <= bounds["x_max"] + x_pad
                and bounds["y_min"] - y_pad <= y <= bounds["y_max"] + y_pad
            ):
                return level_name

        # If no match, default to level 0
        return "level_0"

    def _extract_time_series(self) -> list[dict]:
        """
        Extract time series of agent positions and decisions.

        Returns:
            List of dicts with keys: time, positions, decisions, blocked_exits
        """
        time_series = []

        # Check if we have position history (saved separately for video generation)
        if "position_history" in self.data and self.data["position_history"]:
            logger.info(f"Using position history with {len(self.data['position_history'])} frames")
            # Use saved position history - already in correct format
            for frame in self.data["position_history"]:
                time_series.append(
                    {
                        "time": frame["time"],
                        "positions": frame["positions"],
                        "decisions": self.data.get("agent_decisions", {}),
                        "blocked_exits": frame.get("blocked_exits", []),
                        "agent_states": frame.get("agent_states", {}),
                        "active_train_exits": frame.get("active_train_exits", []),
                        "agent_levels": frame.get("agent_levels"),
                        "escalators": frame.get("escalators"),
                    }
                )
        else:
            # Fallback: use final state only (single frame)
            logger.warning(
                "No position history found - video will show final state only. "
                "Enable video generation during simulation for full animation."
            )
            agent_positions = self.data.get("agent_positions", {})
            agent_decisions = self.data.get("agent_decisions", {})
            blocked_exits = self.data.get("blocked_exits", [])
            final_time = self.data.get("current_time", self.data.get("final_time", 0))

            time_series.append(
                {
                    "time": final_time,
                    "positions": agent_positions,
                    "decisions": agent_decisions,
                    "blocked_exits": blocked_exits,
                    "agent_states": {},
                }
            )

        return time_series

    def _setup_figure(self) -> tuple:
        """
        Setup matplotlib figure and axes for multi-level visualization.

        Returns:
            (fig, axes_dict, title_text)
        """
        # Two floor plans side by side; with escalators, a conveyor strip
        # panel spans the bottom (riders are on no floor while riding).
        if self.escalator_geometry:
            fig = plt.figure(figsize=(16, 11))
            grid = fig.add_gridspec(2, 2, height_ratios=[3.2, 1.25], hspace=0.28)
            ax_level_0 = fig.add_subplot(grid[0, 0])
            ax_level_m1 = fig.add_subplot(grid[0, 1])
            ax_escalators = fig.add_subplot(grid[1, :])
        else:
            fig, (ax_level_0, ax_level_m1) = plt.subplots(
                1, 2, figsize=(16, 8), gridspec_kw={"width_ratios": [1, 1]}
            )
            ax_escalators = None

        title_text = fig.suptitle("Monument Station Evacuation | Time: 00:00:00", fontsize=14)

        # Setup Level 0 axes
        ax_level_0.set_title("Level 0 - Concourse", fontsize=12, fontweight="bold")
        ax_level_0.set_xlabel("X Position (m)")
        ax_level_0.set_ylabel("Y Position (m)")
        ax_level_0.grid(True, alpha=0.3)
        ax_level_0.set_aspect("equal")

        # Setup Level -1 axes
        ax_level_m1.set_title("Level -1 - Platforms", fontsize=12, fontweight="bold")
        ax_level_m1.set_xlabel("X Position (m)")
        ax_level_m1.set_ylabel("Y Position (m)")
        ax_level_m1.grid(True, alpha=0.3)
        ax_level_m1.set_aspect("equal")

        # Draw geometry for both levels
        if self.geometry and "levels" in self.geometry:
            self._draw_geometry(ax_level_0, "level_0")
            self._set_limits_from_geometry(ax_level_0, "level_0")

            self._draw_geometry(ax_level_m1, "level_-1")
            self._set_limits_from_geometry(ax_level_m1, "level_-1")

        axes_dict = {"0": ax_level_0, "-1": ax_level_m1}
        if ax_escalators is not None:
            self._setup_escalator_axes(ax_escalators)
            axes_dict["escalators"] = ax_escalators
        return fig, axes_dict, title_text

    # Okabe-Ito: blue / vermillion stay distinct under common colour-vision deficiencies.
    LANE_COLOURS = {"stand": "#0072B2", "walk": "#D55E00"}
    _LANE_ROW = {"stand": 0.0, "walk": 0.42}

    def _escalator_rows(self) -> list[str]:
        return sorted(self.escalator_geometry, key=lambda n: self.escalator_geometry[n]["letter"])

    def _setup_escalator_axes(self, ax) -> None:
        """One row per escalator, two lane tracks, x = metres along the incline."""
        names = self._escalator_rows()
        max_len = max(g["length_m"] for g in self.escalator_geometry.values())
        ticks, labels = [], []
        for i, name in enumerate(names):
            g = self.escalator_geometry[name]
            arrow = "↑" if g["direction"] == "up" else "↓"
            for lane, dy in self._LANE_ROW.items():
                y = i + dy
                ax.plot(
                    [0, g["length_m"]],
                    [y, y],
                    color="#C8CDD0",
                    linewidth=6,
                    solid_capstyle="butt",
                    zorder=1,
                )
                ticks.append(y)
                labels.append(f"{g['letter']}{arrow} {lane}" if lane == "stand" else "walk")
            ax.plot([g["length_m"]] * 2, [i - 0.15, i + 0.57], color="#555555", linewidth=1)
        ax.set_yticks(ticks)
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_ylim(len(names) - 0.4, -0.3)
        ax.set_xlim(-7.5, max_len + 4.5)
        ax.axvline(0, color="#555555", linewidth=1)
        ax.set_xlabel(
            "Distance along escalator from boarding comb (m)   ·   queue at left, alighting comb at right"
        )
        ax.set_title("Escalators (conveyor model)", fontsize=12, fontweight="bold")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for lane, colour in self.LANE_COLOURS.items():
            ax.plot([], [], "o", color=colour, label=f"{lane} lane")
        ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), fontsize=8, frameon=False, ncol=2)

    def _draw_escalators(self, axes_dict: dict, frame_data: dict) -> None:
        from evacusim.escalators.drawing import rider_floor_position

        state = frame_data.get("escalators") or {}
        ax = axes_dict.get("escalators")
        for i, name in enumerate(self._escalator_rows()):
            esc = state.get(name)
            if esc is None:
                continue
            g = self.escalator_geometry[name]
            for lane, colour in self.LANE_COLOURS.items():
                riders = [r for r in esc.get("riders", []) if r[1] == lane]
                if ax is not None and riders:
                    ax.plot(
                        [r[2] for r in riders],
                        [i + self._LANE_ROW[lane]] * len(riders),
                        "o",
                        color=colour,
                        markersize=5,
                        markeredgecolor="white",
                        markeredgewidth=0.5,
                        zorder=3,
                        label="_agent",
                    )
                for agent_id, _, s in riders:
                    level, x, y = rider_floor_position(g, s, lane)
                    floor_ax = axes_dict.get(level)
                    if floor_ax is not None:
                        floor_ax.plot(
                            x,
                            y,
                            "s",
                            color=colour,
                            markersize=5,
                            markeredgecolor="white",
                            markeredgewidth=0.5,
                            zorder=6,
                            label="_agent",
                        )
            if ax is None:
                continue
            queue = esc.get("queue", {})
            waiting = sum(queue.values())
            if waiting:
                ax.text(
                    -0.6,
                    i + 0.21,
                    f"{waiting} queueing",
                    ha="right",
                    va="center",
                    fontsize=8,
                    color="#333333",
                    label="_agent",
                )
            status = "CLOSED" if esc.get("closed") else ("PAUSED" if esc.get("stalled") else "")
            if status:
                ax.text(
                    g["length_m"] + 0.6,
                    i + 0.21,
                    status,
                    ha="left",
                    va="center",
                    fontsize=8,
                    color="#B00020",
                    fontweight="bold",
                    label="_agent",
                )

    def _draw_geometry(self, ax, level_name: str = None):
        """Draw station geometry on axes for a specific level."""
        if not self.geometry:
            return

        # Handle both old single-level and new multi-level geometry formats
        if "levels" in self.geometry:
            # Multi-level format
            if not level_name:
                level_name = "level_0"
            elif not level_name.startswith("level_"):
                level_name = f"level_{level_name}"

            if level_name not in self.geometry["levels"]:
                logger.warning(f"Level {level_name} not found in geometry")
                return
            geom = self.geometry["levels"][level_name]
        else:
            # Single-level format (backward compatibility)
            geom = self.geometry

        # Draw walkable areas
        if "walkable_areas" in geom:
            for _, coords in geom["walkable_areas"].items():
                if coords:
                    polygon = MPLPolygon(coords, fill=True, alpha=0.2, color="gray")
                    ax.add_patch(polygon)

        # Draw entrances/exits
        if "entrance_areas" in geom:
            for _, coords in geom["entrance_areas"].items():
                if coords:
                    polygon = MPLPolygon(coords, fill=True, alpha=0.3, color="green")
                    ax.add_patch(polygon)

        # Draw platforms
        if "platform_areas" in geom:
            for _, coords in geom["platform_areas"].items():
                if coords:
                    polygon = MPLPolygon(coords, fill=True, alpha=0.3, color="blue")
                    ax.add_patch(polygon)

        # Draw escalator corridors (distinct orange, outline only so walkable floor shows through)
        if "escalator_corridors" in geom:
            for _, coords in geom["escalator_corridors"].items():
                if coords:
                    polygon = MPLPolygon(
                        coords,
                        fill=True,
                        alpha=0.35,
                        facecolor="#FF8C00",
                        edgecolor="#FF4500",
                        linewidth=1.2,
                    )
                    ax.add_patch(polygon)

        # Draw obstacles
        if "obstacles" in geom:
            for coords in geom["obstacles"]:
                if coords:
                    polygon = MPLPolygon(coords, fill=True, alpha=0.4, color="black")
                    ax.add_patch(polygon)

        # Label each named platform walkable area (platform_1, platform_2, …)
        # with a prominent number so the platform is easy to identify.
        if "walkable_areas" in geom:
            for name, coords in geom["walkable_areas"].items():
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
                    cx,
                    cy,
                    f"P{platform_num}",
                    ha="center",
                    va="center",
                    fontsize=11,
                    color="#994400",
                    fontweight="bold",
                    clip_on=True,
                    zorder=6,
                )

    def _set_limits_from_geometry(self, ax, level_name: str = None):
        """Set axis limits from geometry for a specific level."""
        if not self.geometry:
            return

        # Handle both old single-level and new multi-level geometry formats
        if "levels" in self.geometry:
            # Multi-level format - convert level_0 or level_-1 to full key
            if not level_name:
                level_name = "level_0"
            elif not level_name.startswith("level_"):
                level_name = f"level_{level_name}"

            if level_name not in self.geometry["levels"]:
                logger.warning(f"Level {level_name} not found in geometry")
                return
            geom = self.geometry["levels"][level_name]
        else:
            # Single-level format (backward compatibility)
            geom = self.geometry

        coords_list = []
        for key in ("walkable_areas", "entrance_areas", "platform_areas", "obstacles"):
            areas = geom.get(key, {})
            if isinstance(areas, dict):
                for coords in areas.values():
                    if coords:
                        coords_list.extend(coords)
            elif isinstance(areas, list):
                for coords in areas:
                    if coords:
                        coords_list.extend(coords)

        if coords_list:
            if level_name == "level_-1":
                for coords in self.train_polygons.values():
                    coords_list.extend(coords)
            xs = [c[0] for c in coords_list]
            ys = [c[1] for c in coords_list]
            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)

            pad_x = (x_max - x_min) * 0.05 if x_max > x_min else 5.0
            pad_y = (y_max - y_min) * 0.05 if y_max > y_min else 5.0

            ax.set_xlim(x_min - pad_x, x_max + pad_x)
            ax.set_ylim(y_min - pad_y, y_max + pad_y)

    def _draw_frame(self, axes_dict, frame_data, title_text):
        """
        Draw a single frame of the video.

        Args:
            axes_dict: Dict of level_key -> axes for rendering each level
            frame_data: Dict with time, positions, decisions, etc.
            title_text: Title text object
        """
        # Clear previous frame (keep geometry but remove agents)
        for ax in axes_dict.values():
            for artist in ax.get_children():
                if hasattr(artist, "get_label") and artist.get_label() == "_agent":
                    artist.remove()

        # Update title
        time_val = frame_data["time"]
        title_text.set_text(f"Monument Station Evacuation | Time: {_fmt_clock(time_val)}")

        # Draw blocked exits (if multi-level geometry, show on appropriate level)
        blocked_exits = frame_data.get("blocked_exits", [])
        if blocked_exits and self.geometry and "levels" in self.geometry:
            for level_key, ax in axes_dict.items():
                level_name = f"level_{level_key}"
                geom = self.geometry["levels"].get(level_name)
                if not geom:
                    continue

                entrance_areas = geom.get("entrance_areas", {})
                for exit_name in blocked_exits:
                    if exit_name in entrance_areas:
                        coords = entrance_areas[exit_name]
                        if coords:
                            xs = [c[0] for c in coords]
                            ys = [c[1] for c in coords]
                            center_x = sum(xs) / len(xs)
                            center_y = sum(ys) / len(ys)

                            size = 8
                            ax.plot(
                                [center_x - size, center_x + size],
                                [center_y - size, center_y + size],
                                "r-",
                                linewidth=4,
                                label="_agent",
                            )
                            ax.plot(
                                [center_x - size, center_x + size],
                                [center_y + size, center_y - size],
                                "r-",
                                linewidth=4,
                                label="_agent",
                            )
                            ax.text(
                                center_x,
                                center_y - size - 3,
                                "🚧 BLOCKED",
                                ha="center",
                                fontsize=10,
                                color="red",
                                weight="bold",
                                label="_agent",
                            )

        # Draw agent positions on appropriate level
        positions = frame_data.get("positions", {})
        # Per-frame roles override (future support); fall back to self.agent_roles
        frame_roles = frame_data.get("agent_roles", self.agent_roles)
        # Per-frame agent_levels (accurate for this timestep); fall back to static final-state.
        frame_agent_levels: dict[str, str] | None = frame_data.get("agent_levels")
        for agent_id, pos in positions.items():
            if pos and len(pos) >= 2:
                x, y = pos[0], pos[1]

                # Determine which level this agent is on — prefer the per-frame
                # snapshot so agents that transferred mid-simulation appear on the
                # correct panel at every point in time.
                if frame_agent_levels is not None and agent_id in frame_agent_levels:
                    raw_level = str(frame_agent_levels[agent_id])
                    level_key = raw_level  # already "-1" or "0"
                else:
                    level_name = self._determine_agent_level(agent_id, pos)
                    level_key = level_name.replace("level_", "")

                if level_key not in axes_dict:
                    # Fallback: default to level 0
                    level_key = "0"

                ax = axes_dict[level_key]

                role = frame_roles.get(agent_id, "")
                face, edge = self._colour_map.get(agent_id, frame_roles)
                size = 10 if role else 8

                ax.plot(
                    x,
                    y,
                    "o",
                    color=face,
                    markeredgecolor=edge,
                    markeredgewidth=1.5,
                    markersize=size,
                    label="_agent",
                )
                ax.text(x, y + 1, agent_id, ha="center", fontsize=8, label="_agent")

        if self.escalator_geometry:
            self._draw_escalators(axes_dict, frame_data)

        # Draw train bodies on the platform panel (level -1) for every
        # train that is currently present in the station.
        active_exits = set(frame_data.get("active_train_exits", []))
        if "-1" in axes_dict and self.train_polygons:
            ax_plat = axes_dict["-1"]
            for exit_name, coords in self.train_polygons.items():
                if exit_name not in active_exits:
                    continue
                train_patch = MPLPolygon(
                    coords,
                    closed=True,
                    facecolor="#D7DEE2",
                    edgecolor="#26343A",
                    linewidth=1.5,
                    alpha=0.96,
                    zorder=7,
                    label="_agent",
                )
                ax_plat.add_patch(train_patch)
                x_mid = sum(point[0] for point in coords) / len(coords)
                y_mid = sum(point[1] for point in coords) / len(coords)
                edge_dx = coords[1][0] - coords[0][0]
                edge_dy = coords[1][1] - coords[0][1]
                label_rotation = math.degrees(math.atan2(edge_dy, edge_dx))
                if label_rotation > 90:
                    label_rotation -= 180
                elif label_rotation <= -90:
                    label_rotation += 180
                ax_plat.text(
                    x_mid,
                    y_mid,
                    f"TRAIN P{exit_name.rsplit('_', 1)[-1]}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="#222222",
                    fontweight="bold",
                    rotation=label_rotation,
                    zorder=8,
                    clip_on=True,
                    label="_agent",
                )

    def generate(self, output_path: Path, dpi: int = 100) -> bool:
        """
        Generate video file.

        Args:
            output_path: Path for output MP4 file
            dpi: Resolution (dots per inch)

        Returns:
            True if successful, False otherwise
        """
        logger.info(f"Generating video: {output_path}")
        logger.info(f"Video settings: {self.fps} fps, {self.speedup}x speed, {dpi} dpi")

        try:
            # Setup figure
            fig, axes_dict, title_text = self._setup_figure()

            # Setup video writer
            writer = FFMpegWriter(fps=self.fps, metadata={"artist": "NewcastleSim"})

            with writer.saving(fig, str(output_path), dpi=dpi):
                # For now, we only have one frame (final state)
                # In a proper implementation, we'd iterate through time series
                for frame_data in self.time_series:
                    self._draw_frame(axes_dict, frame_data, title_text)
                    writer.grab_frame()

            plt.close(fig)
            logger.info(f"Video saved: {output_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to generate video: {e}", exc_info=True)
            return False


def generate_video_from_output(
    output_file: Path,
    video_path: Path | None = None,
    geometry: dict | None = None,
    fps: int = 20,
    speedup: float = 1.0,
    dpi: int = 100,
) -> bool:
    """
    Generate video from simulation output file.

    Args:
        output_file: Path to agent decisions JSON file
        video_path: Output video path (default: same dir as output_file)
        geometry: Station geometry dict
        fps: Frames per second
        speedup: Speed multiplier
        dpi: Resolution

    Returns:
        True if successful
    """
    if video_path is None:
        video_path = output_file.parent / f"{output_file.stem}_video.mp4"

    generator = VideoGenerator(output_file, geometry, fps, speedup)
    return generator.generate(video_path, dpi)
