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
from evacusim.visualization.video_generation_helper import RoleColourMap

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

        # Extract agent roles (director agents have a non-empty role label)
        self.agent_roles: dict[str, str] = self.data.get("agent_roles", {})
        if self.agent_roles:
            logger.info(f"Loaded roles for {len(self.agent_roles)} director agent(s)")
        self._colour_map = RoleColourMap.from_roles(self.agent_roles)

        # Build level bounds from geometry for coordinate-based inference
        self.level_bounds = self._build_level_bounds()

        # Extract time series data
        self.time_series = self._extract_time_series()
        if not self.time_series:
            raise ValueError("No position data found in output file")

        # Derive the initial agent set and track train-boarded agents across frames.
        # An agent is considered to have boarded a train when they vanish from the
        # positions dict while their last recorded destination was a train_platform_*.
        self._initial_agents: set[str] = (
            set(self.time_series[0]["positions"].keys()) if self.time_series else set()
        )
        # Cumulative map: agent_id -> last known destination (updated each frame)
        self._last_known_destination: dict[str, str] = {}
        # Running count of agents confirmed to have boarded trains
        self._boarded_count: int = 0
        self._boarded_agents: set[str] = set()

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

        # Draw train boarding zones (jupedsim.train_entrance polygons on level -1).
        # Draw as solid green markers only — the platform number label lives on the
        # larger platform walkable area drawn below.
        if "train_entrance_areas" in geom:
            for name, coords in geom["train_entrance_areas"].items():
                if coords:
                    polygon = MPLPolygon(
                        coords,
                        fill=True,
                        alpha=0.85,
                        facecolor="#00CC44",
                        edgecolor="white",
                        linewidth=1.5,
                        zorder=5,
                    )
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
                    cx, cy, f"P{platform_num}",
                    ha="center", va="center",
                    fontsize=11, color="#994400", fontweight="bold",
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
            xs = [c[0] for c in coords_list]
            ys = [c[1] for c in coords_list]
            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)

            pad_x = (x_max - x_min) * 0.05 if x_max > x_min else 5.0
            pad_y = (y_max - y_min) * 0.05 if y_max > y_min else 5.0

            ax.set_xlim(x_min - pad_x, x_max + pad_x)
            ax.set_ylim(y_min - pad_y, y_max + pad_y)

    def _get_train_entrance_areas(self) -> dict[str, list]:
        """Return train_entrance_areas coord dict from geometry (level -1 only)."""
        if not self.geometry:
            return {}
        if "levels" in self.geometry:
            geom = self.geometry["levels"].get("level_-1", {})
        else:
            geom = self.geometry
        return geom.get("train_entrance_areas", {})

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
        # Per-frame roles override (future support); fall back to self.agent_roles
        frame_roles = frame_data.get("agent_roles", self.agent_roles)
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

                role = frame_roles.get(agent_id, "")
                face, edge = self._colour_map.get(agent_id, frame_roles)
                size = 10 if role else 8

                ax.plot(
                    x, y, "o",
                    color=face,
                    markeredgecolor=edge,
                    markeredgewidth=1.5,
                    markersize=size,
                    label="_agent",
                )
                ax.text(x, y + 1, agent_id, ha="center", fontsize=8, label="_agent")

        # Update last-known destination from this frame's agent_states.
        # Also detect agents who have just boarded a train (disappeared while
        # their last destination pointed at a train_platform_*).
        agent_states = frame_data.get("agent_states", {})
        for agent_id, state in agent_states.items():
            dest = (state.get("destination") or "")
            if dest:
                self._last_known_destination[agent_id] = dest

        current_agents = set(positions.keys())
        for agent_id in self._initial_agents - current_agents - self._boarded_agents:
            dest = self._last_known_destination.get(agent_id, "")
            if dest.startswith("train_platform_"):
                self._boarded_agents.add(agent_id)

        self._boarded_count = len(self._boarded_agents)

        # Compact train/boarding status badge on the platform panel (level -1).
        if "-1" in axes_dict:
            ax_plat = axes_dict["-1"]
            active_exits = set(frame_data.get("active_train_exits", []))
            boarded = self._boarded_count
            if active_exits:
                platforms = " ".join(
                    f"P{n.rsplit('_',1)[-1]}" for n in sorted(active_exits)
                )
                status_str = f"\U0001f682 Boarding: {platforms}   Boarded: {boarded}"
                status_color = "#006600"
                face_color = "#CCFFCC"
            elif boarded > 0:
                status_str = f"\U0001f6ab Train departed   Boarded: {boarded}"
                status_color = "#555555"
                face_color = "#F0F0F0"
            else:
                status_str = "Train not yet arrived"
                status_color = "#555555"
                face_color = "#F8F8F8"
            ax_plat.text(
                0.02, 0.97,
                status_str,
                transform=ax_plat.transAxes,
                ha="left", va="top",
                fontsize=8, color=status_color,
                bbox=dict(boxstyle="round,pad=0.25", facecolor=face_color, alpha=0.85),
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
