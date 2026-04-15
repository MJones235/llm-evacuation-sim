"""
Video generation utilities for Station Concordia simulations.

Handles post-simulation video generation including:
- Geometry loading from network files
- Position history merging
- Video file creation
- Role-based agent colouring
"""

import json
from pathlib import Path

from evacusim.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Palette — visually distinct colours, chosen to stand out against the red
# used for ordinary agents and against the grey/green/blue map background.
# Add more entries here if more than 8 distinct director roles are needed.
# ---------------------------------------------------------------------------
_DIRECTOR_PALETTE: list[tuple[str, str]] = [
    ("#FFD700", "#B8860B"),  # gold
    ("#FF6600", "#CC3300"),  # orange
    ("#00CC66", "#007A3D"),  # green
    ("#0080FF", "#004C99"),  # blue
    ("#CC44CC", "#882288"),  # purple
    ("#00CCCC", "#007A7A"),  # teal
    ("#FF99CC", "#CC3377"),  # pink
    ("#AAAAAA", "#555555"),  # grey (last resort)
]
_PASSENGER_COLOURS: tuple[str, str] = ("red", "darkred")


class RoleColourMap:
    """Assigns stable (face, edge) colour pairs to agent roles from a palette.

    Passenger agents (empty role string) always receive ``_PASSENGER_COLOURS``.
    Director agents are assigned palette entries in the order their role label
    is first seen.  The same role label always maps to the same colour within
    one session, and the mapping is deterministic across runs because roles
    are resolved by sorted insertion order when ``from_roles`` is used.

    Usage::

        colour_map = RoleColourMap.from_roles(agent_roles)
        face, edge = colour_map.get(agent_id, agent_roles)
    """

    def __init__(self) -> None:
        self._role_to_index: dict[str, int] = {}

    @classmethod
    def from_roles(cls, agent_roles: dict[str, str]) -> "RoleColourMap":
        """Pre-populate the map with all known roles (sorted for determinism)."""
        instance = cls()
        for role in sorted(set(agent_roles.values())):
            if role:
                instance._assign(role)
        return instance

    def _assign(self, role: str) -> int:
        """Lazily assign the next palette slot to *role* and return its index."""
        if role not in self._role_to_index:
            self._role_to_index[role] = len(self._role_to_index)
        return self._role_to_index[role]

    def get(self, agent_id: str, agent_roles: dict[str, str]) -> tuple[str, str]:
        """Return ``(face_colour, edge_colour)`` for *agent_id*."""
        role = agent_roles.get(agent_id, "")
        if not role:
            return _PASSENGER_COLOURS
        idx = self._assign(role) % len(_DIRECTOR_PALETTE)
        return _DIRECTOR_PALETTE[idx]


class VideoGenerationHelper:
    """Helper class for generating videos from simulation output."""

    @staticmethod
    def load_geometry_from_network(network_path: Path) -> dict | None:
        """
        Load station geometry from SUMO network files.

        Supports multi-level stations: if level_0.xml and level_-1.xml both
        exist the returned dict uses the ``{"levels": {"level_0": ...,
        "level_-1": ...}}`` format expected by VideoGenerator.  Train entrance
        areas (jupedsim.train_entrance polygons) are loaded from level_-1.xml
        and stored under the key ``"train_entrance_areas"`` so the video
        renderer can draw them as boarding zones.

        Args:
            network_path: Path to station network directory

        Returns:
            Dictionary with geometry data or None if loading fails
        """
        try:
            from evacusim.jps.geometry_loader import (
                load_entrance_areas,
                load_escalator_corridors,
                load_obstacles,
                load_platform_areas,
                load_train_entrance_areas,
                load_walkable_areas,
            )

            def poly_to_coords(poly):
                return list(poly.exterior.coords)

            def _load_level(xml_path: Path) -> dict:
                walkable_areas = load_walkable_areas(str(xml_path))
                entrance_areas = load_entrance_areas(str(xml_path))
                platform_areas = load_platform_areas(str(xml_path))
                obstacles = load_obstacles(str(xml_path))
                escalator_corridors = load_escalator_corridors(str(xml_path))
                train_entrance_areas = load_train_entrance_areas(str(xml_path))
                logger.info(
                    f"Loaded geometry from {xml_path.name}: "
                    f"{len(walkable_areas)} walkable, {len(entrance_areas)} entrances, "
                    f"{len(platform_areas)} platforms, {len(obstacles)} obstacles, "
                    f"{len(escalator_corridors)} escalator corridors, "
                    f"{len(train_entrance_areas)} train boarding zones"
                )
                return {
                    "walkable_areas": {n: poly_to_coords(p) for n, p in walkable_areas.items()},
                    "entrance_areas": {n: poly_to_coords(p) for n, p in entrance_areas.items()},
                    "platform_areas": {n: poly_to_coords(p) for n, p in platform_areas.items()},
                    "obstacles": [poly_to_coords(p) for p in obstacles],
                    "escalator_corridors": {
                        n: poly_to_coords(p) for n, p in escalator_corridors.items()
                    },
                    "train_entrance_areas": {
                        n: poly_to_coords(p) for n, p in train_entrance_areas.items()
                    },
                }

            level0_file = network_path / "level_0.xml"
            level_m1_file = network_path / "level_-1.xml"
            walking_areas_file = network_path / "walking_areas.add.xml"

            if level0_file.exists() and level_m1_file.exists():
                # Multi-level station: return nested levels dict
                return {
                    "levels": {
                        "level_0": _load_level(level0_file),
                        "level_-1": _load_level(level_m1_file),
                    }
                }
            elif level0_file.exists():
                return _load_level(level0_file)
            elif walking_areas_file.exists():
                return _load_level(walking_areas_file)
            else:
                logger.warning(
                    "Geometry file not found: expected level_0.xml or walking_areas.add.xml "
                    f"in {network_path}"
                )
                return None

        except Exception as e:
            logger.error(f"Failed to load geometry: {e}")
            return None

    @staticmethod
    def merge_position_history(decisions_file: Path, history_file: Path) -> Path | None:
        """
        Merge position history with decisions data.

        Supports both the legacy ``.json`` (wrapped) format and the streaming
        ``.jsonl`` format (one JSON object per line) produced by the position
        history tracker's streaming mode.

        Args:
            decisions_file: Path to agent decisions JSON
            history_file: Path to position history file (.json or .jsonl)

        Returns:
            Path to merged temporary file or None if merging fails
        """
        try:
            # Handle streaming .jsonl format (one frame per line)
            if history_file.suffix == ".jsonl":
                frames = []
                with open(history_file) as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            frames.append(json.loads(line))
                position_history = frames
            else:
                with open(history_file) as f:
                    history_data = json.load(f)
                position_history = history_data.get("position_history", [])

            with open(decisions_file) as f:
                decisions_data = json.load(f)

            # Merge position history into decisions data
            decisions_data["position_history"] = position_history

            # Save merged data temporarily
            merged_file = decisions_file.parent / f"{decisions_file.stem}_merged.json"
            with open(merged_file, "w") as f:
                json.dump(decisions_data, f)

            logger.info(f"Merged {len(position_history)} position frames with decisions")
            return merged_file

        except Exception as e:
            logger.error(f"Failed to merge position history: {e}")
            return None

    @staticmethod
    def generate_simulation_video(
        decisions_file: Path,
        run_id: str,
        network_path: Path,
        fps: int = 20,
        speedup: float = 1.0,
    ) -> bool:
        """
        Generate video from simulation output.

        Args:
            decisions_file: Path to agent decisions JSON file
            run_id: Unique run identifier
            network_path: Path to station network directory
            fps: Frames per second
            speedup: Speed multiplier

        Returns:
            True if video was generated successfully
        """
        from evacusim.visualization.video_generator import (
            generate_video_from_output,
        )

        logger.info("=" * 60)
        logger.info("Generating video...")
        logger.info("=" * 60)

        # Load geometry
        geometry = VideoGenerationHelper.load_geometry_from_network(network_path)

        # Check for position history — prefer the streaming .jsonl format produced
        # by the position tracker, fall back to the legacy .json wrapper.
        history_file = decisions_file.parent / f"{decisions_file.stem}_history.jsonl"
        if not history_file.exists():
            history_file = decisions_file.parent / f"{decisions_file.stem}_history.json"
        if not history_file.exists():
            logger.warning(
                "No position history found. Enable video generation in config "
                "to track positions during simulation."
            )
            return False

        # Merge position history with decisions
        merged_file = VideoGenerationHelper.merge_position_history(decisions_file, history_file)
        if not merged_file:
            return False

        try:
            # Generate video
            video_path = decisions_file.parent / f"{run_id}_video.mp4"
            video_success = generate_video_from_output(
                merged_file,
                video_path=video_path,
                geometry=geometry,
                fps=fps,
                speedup=speedup,
            )

            if video_success:
                logger.info(f"Video saved: {video_path}")
            else:
                logger.error("Video generation failed")

            return video_success

        finally:
            # Clean up merged file
            if merged_file and merged_file.exists():
                merged_file.unlink()
