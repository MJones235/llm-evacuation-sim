"""
Video generator for Station Concordia simulations.

This module creates MP4 videos from simulation output data.
Videos show agent positions and decisions at regular time intervals,
without delays for LLM responses.
"""

import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from matplotlib.patches import Polygon as MPLPolygon

from evacusim.utils.logger import get_logger

logger = get_logger(__name__)

matplotlib.use("Agg")  # Non-interactive backend for video generation


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

        # Build level bounds from geometry for coordinate-based inference
        self.level_bounds = self._build_level_bounds()

        # Extract time series data
        self.time_series = self._extract_time_series()
        if not self.time_series:
            raise ValueError("No position data found in output file")

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
        # Create 2 subplots for Level 0 and Level -1
        fig, (ax_level_0, ax_level_m1) = plt.subplots(
            1, 2, figsize=(16, 8), gridspec_kw={"width_ratios": [1, 1]}
        )

        title_text = fig.suptitle("Monument Station Evacuation | Time: 0.0s", fontsize=14)

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
        return fig, axes_dict, title_text

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
        title_text.set_text(f"Monument Station Evacuation | Time: {time_val:.1f}s")

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
        for agent_id, pos in positions.items():
            if pos and len(pos) >= 2:
                x, y = pos[0], pos[1]

                # Determine which level this agent is on
                level_name = self._determine_agent_level(agent_id, pos)
                # Convert from "level_0" or "level_-1" to key "0" or "-1"
                level_key = level_name.replace("level_", "")

                if level_key not in axes_dict:
                    # Fallback: default to level 0
                    level_key = "0"

                ax = axes_dict[level_key]

                ax.plot(x, y, "o", color="red", markersize=8, label="_agent")
                ax.text(x, y + 1, agent_id, ha="center", fontsize=8, label="_agent")

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
