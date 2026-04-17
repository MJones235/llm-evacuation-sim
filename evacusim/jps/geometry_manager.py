"""
Geometry management for JuPedSim simulation.

Handles loading, processing, and validation of station geometry including
walkable areas, entrance areas, platforms, and obstacles.
"""

from pathlib import Path
from typing import Any

import jupedsim as jps

from evacusim.utils.logger import get_logger
from evacusim.jps.geometry_processor import GeometryProcessor
from evacusim.jps.geometry_loader import (
    load_entrance_areas,
    load_escalator_corridors,
    load_exit_thresholds,
    load_obstacles,
    load_platform_areas,
    load_train_entrance_areas,
    load_walkable_areas,
)

logger = get_logger(__name__)


class GeometryManager:
    """
    Manages station geometry loading and JuPedSim simulation creation.

    Handles:
    - Loading geometry from SUMO network files
    - Integrating obstacles into walkable areas
    - Creating JuPedSim simulation with combined geometry
    - Providing access to geometry data for visualization
    """

    def __init__(self, network_path: Path, dt: float = 0.05, level_id: int | str = 0,
                 initially_blocked_exits: set[str] | None = None):
        """
        Initialize geometry manager and load station geometry.

        Args:
            network_path: Path to network directory containing level_*.xml or walking_areas.add.xml
            dt: Timestep in seconds (matches JuPedSim convention)
            level_id: Level ID to load (default: 0). Looks for level_{level_id}.xml file.
                     Falls back to walking_areas.add.xml if level file not found.
            initially_blocked_exits: Exits that are blocked from the start.  Their
                escalator corridors and transfer-zone walkable areas are removed from
                the navmesh before the JuPedSim simulation is created, so agents can
                never enter them.

        Raises:
            FileNotFoundError: If no geometry file found
            ValueError: If no walkable areas found in geometry
        """
        self.network_path = network_path
        self.dt = dt
        self.level_id = level_id
        self._initially_blocked_exits: set[str] = set(initially_blocked_exits or [])
        # Centroid positions of pre-blocked exits, recorded before their geometry
        # is removed from the navmesh.  Used by visibility/LOS checks so agents
        # can still perceive that a blocked exit is ahead of them.
        self.blocked_exit_positions: dict[str, tuple[float, float]] = {}

        # Load geometry
        logger.info("Loading station geometry from network files...")
        (
            self.walkable_areas,
            self.walkable_areas_with_obstacles,
            self.entrance_areas,
            self.platform_areas,
            self.obstacles,
            self.escalator_corridors,
            self.exit_thresholds,
            self.train_entrance_areas,
        ) = self._load_geometry()

        # Snapshot of escalator transfer-zone polygons taken before any blockage
        # removal so that StationLayoutBuilder can still compute entrance positions
        # for blocked escalators (whose TZ is popped from walkable_areas below).
        import re as _re_tz
        self.escalator_transfer_zones: dict[str, object] = {
            k: v for k, v in self.walkable_areas.items()
            if _re_tz.match(r"^L[^_]+_esc_[a-f]_", k)
        }

        # Remove corridor + transfer-zone polygons for pre-blocked escalators so
        # agents physically cannot enter the shaft from simulation start.
        if self._initially_blocked_exits:
            self._apply_initial_blockages()

        # Create JuPedSim simulation
        logger.info("Initializing JuPedSim simulation...")
        self.simulation = self._create_simulation()

        logger.info(
            f"Geometry loaded: "
            f"{len(self.walkable_areas)} walkable areas, "
            f"{len(self.entrance_areas)} entrances, "
            f"{len(self.platform_areas)} platforms, "
            f"{len(self.obstacles)} obstacles"
        )

    def _load_geometry(self) -> tuple[dict, dict, dict, dict, list]:
        """
        Load station geometry from SUMO network files.

        Looks for level_{level_id}.xml first (for multi-level support),
        then falls back to walking_areas.add.xml for backwards compatibility.

        Returns:
            Tuple of (walkable_areas, walkable_areas_with_obstacles,
                     entrance_areas, platform_areas, obstacles)

        Raises:
            FileNotFoundError: If no geometry file found
        """
        # Try level-specific file first
        level_file = self.network_path / f"level_{self.level_id}.xml"
        walking_areas_file = self.network_path / "walking_areas.add.xml"

        if level_file.exists():
            geom_file = level_file
            logger.info(f"Loading geometry from level file: {level_file.name}")
        elif walking_areas_file.exists():
            geom_file = walking_areas_file
            logger.info(f"Loading geometry from legacy file: {walking_areas_file.name}")
        else:
            raise FileNotFoundError(
                f"Geometry file not found. Looked for:\n"
                f"  - {level_file}\n"
                f"  - {walking_areas_file}"
            )

        walkable_areas = load_walkable_areas(str(geom_file))
        entrance_areas = load_entrance_areas(str(geom_file))
        platform_areas = load_platform_areas(str(geom_file))
        obstacles = load_obstacles(str(geom_file))
        escalator_corridors = load_escalator_corridors(str(geom_file))
        exit_thresholds = load_exit_thresholds(str(geom_file))
        train_entrance_areas = load_train_entrance_areas(str(geom_file))
        if exit_thresholds:
            logger.info(f"  Loaded {len(exit_thresholds)} exit thresholds: {list(exit_thresholds.keys())}")
        if train_entrance_areas:
            logger.info(f"  Loaded {len(train_entrance_areas)} train entrance areas: {list(train_entrance_areas.keys())}")

        # Integrate obstacles into walkable areas as polygon holes
        walkable_areas_with_obstacles, fixed_obstacles = GeometryProcessor.integrate_obstacles(
            walkable_areas, obstacles
        )

        logger.info(f"  Loaded {len(walkable_areas)} walkable areas")
        logger.info(f"  Loaded {len(entrance_areas)} entrance areas")
        logger.info(f"  Loaded {len(platform_areas)} platform areas")
        logger.info(f"  Loaded {len(obstacles)} obstacles")
        logger.info(f"  Integrated {len(fixed_obstacles)} obstacles into walkable areas")
        logger.info(f"  Loaded {len(escalator_corridors)} escalator corridors")

        return (
            walkable_areas,
            walkable_areas_with_obstacles,
            entrance_areas,
            platform_areas,
            fixed_obstacles,
            escalator_corridors,
            exit_thresholds,
            train_entrance_areas,
        )

    def _create_simulation(self) -> jps.Simulation:
        """
        Create JuPedSim simulation with loaded geometry.

        Returns:
            Configured JuPedSim simulation instance

        Raises:
            ValueError: If no walkable areas found
        """
        # Merge all walkable areas (with obstacles removed) into one geometry
        all_areas = list(self.walkable_areas_with_obstacles.values())

        if not all_areas:
            raise ValueError("No walkable areas found in geometry")

        # Combine into a single geometry
        main_area = GeometryProcessor.combine_geometry(all_areas)

        # Keep a live reference so add_obstacle_polygon() can rebuild geometry at runtime.
        self._combined_geometry = main_area

        # Create JuPedSim simulation
        simulation = jps.Simulation(
            model=jps.CollisionFreeSpeedModel(),
            geometry=main_area,
            dt=self.dt,
        )

        logger.info(f"  Created simulation with area: {main_area.area:.1f} m²")

        return simulation

    def _apply_initial_blockages(self) -> None:
        """Remove corridor and transfer-zone polygons for pre-blocked escalators.

        Called once at init before the JuPedSim simulation is created.  Works by
        mutating ``walkable_areas`` and ``walkable_areas_with_obstacles`` in-place
        so that ``_create_simulation`` builds the navmesh without those zones.

        Also records the centroid of each blocked transfer-zone in
        ``self.blocked_exit_positions`` *before* removal so that visibility
        checks (LOS) can still locate the physical position of a blocked exit.
        """
        from shapely.ops import unary_union as _union

        level = str(self.level_id)
        for exit_name in self._initially_blocked_exits:
            parts = exit_name.split("_")  # e.g. ["escalator", "d", "down"]
            if len(parts) < 2 or parts[0] != "escalator":
                continue
            esc_letter = parts[1]

            # Collect corridor + transfer-zone polygons for this escalator on this level.
            shapes = []
            corr_key = f"L{level}_esc_corridor_{esc_letter}"
            if corr_key in self.escalator_corridors:
                shapes.append(self.escalator_corridors[corr_key])
            tz_key = next(
                (k for k in self.walkable_areas if k.startswith(f"L{level}_esc_{esc_letter}_")),
                None,
            )
            if tz_key:
                tz_polygon = self.walkable_areas[tz_key]
                shapes.append(tz_polygon)
                # Record the platform-level entrance of this escalator corridor
                # as the blocked exit position.  This uses the same algorithm as
                # StationLayoutBuilder._escalator_entrance_position so that the
                # waypoint matches the firefighter spawn position exactly: find
                # the corridor edge farthest from the TZ centre (= the bottom /
                # platform-facing edge) and step 1.5 m outward along its normal.
                try:
                    import math as _math
                    corr_poly = self.escalator_corridors.get(corr_key)
                    if corr_poly is not None:
                        _coords = list(corr_poly.exterior.coords)[:-1]
                        _n = len(_coords)
                        _ref_x, _ref_y = tz_polygon.centroid.x, tz_polygon.centroid.y
                        _best_dist, _best_mid, _best_normal = -1.0, (0.0, 0.0), (0.0, 1.0)
                        for _i in range(_n):
                            _p1, _p2 = _coords[_i], _coords[(_i + 1) % _n]
                            _mid = ((_p1[0] + _p2[0]) / 2.0, (_p1[1] + _p2[1]) / 2.0)
                            _dist = _math.hypot(_mid[0] - _ref_x, _mid[1] - _ref_y)
                            if _dist > _best_dist:
                                _best_dist = _dist
                                _best_mid = _mid
                                _dx, _dy = _p2[0] - _p1[0], _p2[1] - _p1[1]
                                _ln = _math.hypot(_dx, _dy)
                                if _ln < 1e-9:
                                    continue
                                _nx1, _ny1 = -_dy / _ln, _dx / _ln
                                _dot1 = _nx1 * (_ref_x - _mid[0]) + _ny1 * (_ref_y - _mid[1])
                                _best_normal = (_nx1, _ny1) if _dot1 < 0 else (_dy / _ln, -_dx / _ln)
                        pos = (_best_mid[0] + _best_normal[0] * 1.5,
                               _best_mid[1] + _best_normal[1] * 1.5)
                        self.blocked_exit_positions[exit_name] = pos
                        logger.debug(
                            f"Stored blocked exit position for '{exit_name}': "
                            f"({pos[0]:.3f}, {pos[1]:.3f}) [firefighter entrance]"
                        )
                    else:
                        centroid = tz_polygon.centroid
                        self.blocked_exit_positions[exit_name] = (centroid.x, centroid.y)
                except Exception:
                    pass
                # Remove the transfer zone from the walkable areas dicts so
                # ExitManager never registers a stage inside it.
                self.walkable_areas.pop(tz_key, None)
                self.walkable_areas_with_obstacles.pop(tz_key, None)

            if not shapes:
                logger.debug(f"No geometry found to block for '{exit_name}' on level {level}")
                continue

            removal = _union(shapes).buffer(0.02)
            # Subtract from every remaining walkable area that overlaps.
            for key in list(self.walkable_areas_with_obstacles):
                poly = self.walkable_areas_with_obstacles[key]
                if poly.intersects(removal):
                    from evacusim.jps.geometry_processor import GeometryProcessor
                    new_poly = GeometryProcessor.fix_topology(poly.difference(removal))
                    if not new_poly.is_empty and hasattr(new_poly, 'exterior'):
                        self.walkable_areas_with_obstacles[key] = new_poly

            logger.info(
                f"🚧 Pre-blocked '{exit_name}' on level {level}: "
                f"removed corridor '{corr_key}' and transfer zone '{tz_key}' from navmesh"
            )

    def add_obstacle_polygon(self, obstacle_poly) -> None:
        """Remove *obstacle_poly* from the walkable geometry and call switch_geometry.

        Raises:
            RuntimeError: propagated from JuPedSim if the new geometry is invalid
                (e.g. disconnected area, stages outside bounds).  Callers should
                catch and handle/ignore as appropriate.
        """
        new_geometry = GeometryProcessor.fix_topology(
            self._combined_geometry.difference(obstacle_poly)
        )
        if new_geometry.is_empty:
            logger.warning("Obstacle subtraction produced empty geometry — skipping switch")
            return
        # May raise RuntimeError ("accessible area not connected", "stages outside
        # geometry", etc.).  Let the caller decide whether to swallow it.
        self.simulation.switch_geometry(new_geometry)
        self._combined_geometry = new_geometry
        logger.info(f"Geometry updated: obstacle removed ({obstacle_poly.area:.2f} m²)")

    def get_geometry_data(self) -> dict[str, Any]:
        """
        Get geometry information for visualization or analysis.

        Returns:
            Dictionary with all geometry components
        """
        return {
            "walkable_areas": self.walkable_areas,
            "walkable_areas_with_obstacles": self.walkable_areas_with_obstacles,
            "entrance_areas": self.entrance_areas,
            "platform_areas": self.platform_areas,
            "obstacles": self.obstacles,
        }
