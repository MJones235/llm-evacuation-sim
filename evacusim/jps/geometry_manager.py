"""
Geometry management for JuPedSim simulation.

Handles loading, processing, and validation of station geometry including
walkable areas, entrance areas, platforms, and obstacles.
"""

from pathlib import Path
from typing import Any

import jupedsim as jps

from evacusim.escalators.spec_loader import load_combs
from evacusim.jps.geometry_loader import (
    load_entrance_areas,
    load_escalator_corridors,
    load_exit_thresholds,
    load_obstacles,
    load_platform_areas,
    load_train_entrance_areas,
    load_walkable_areas,
)
from evacusim.jps.geometry_processor import GeometryProcessor
from evacusim.utils.logger import get_logger

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

    def __init__(
        self,
        network_path: Path,
        dt: float = 0.05,
        level_id: int | str = 0,
        initially_blocked_exits: set[str] | None = None,
    ):
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
            self.escalator_combs,
            self.exit_thresholds,
            self.train_entrance_areas,
        ) = self._load_geometry()

        # Escalators are not floor: riders leave the floor simulation at the
        # boarding comb and travel on a conveyor (evacusim.escalators). Remove
        # every corridor and transfer zone from the navmesh; the corridors are
        # kept only for drawing, and the comb records locate the landings.
        self._detach_escalators()

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
                f"Geometry file not found. Looked for:\n  - {level_file}\n  - {walking_areas_file}"
            )

        walkable_areas = load_walkable_areas(str(geom_file))
        entrance_areas = load_entrance_areas(str(geom_file))
        platform_areas = load_platform_areas(str(geom_file))
        obstacles = load_obstacles(str(geom_file))
        escalator_corridors = load_escalator_corridors(str(geom_file))
        escalator_combs = load_combs(geom_file, str(self.level_id))
        exit_thresholds = load_exit_thresholds(str(geom_file))
        train_entrance_areas = load_train_entrance_areas(str(geom_file))
        if exit_thresholds:
            logger.info(
                f"  Loaded {len(exit_thresholds)} exit thresholds: {list(exit_thresholds.keys())}"
            )
        if train_entrance_areas:
            logger.info(
                f"  Loaded {len(train_entrance_areas)} train entrance areas: {list(train_entrance_areas.keys())}"
            )

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
        logger.info(f"  Loaded {len(escalator_combs)} escalator combs")

        return (
            walkable_areas,
            walkable_areas_with_obstacles,
            entrance_areas,
            platform_areas,
            fixed_obstacles,
            escalator_corridors,
            escalator_combs,
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

    def _detach_escalators(self) -> None:
        """Remove every escalator corridor and transfer zone from the walkable floor.

        Also records, for escalators blocked from the start, the landing point in
        front of their boarding comb in ``blocked_exit_positions`` so visibility
        checks can still locate them.
        """
        from shapely.ops import unary_union as _union

        from evacusim.jps.geometry_processor import GeometryProcessor

        zones = [k for k in self.walkable_areas if k.startswith("esc.")]
        shapes = list(self.escalator_corridors.values()) + [self.walkable_areas[k] for k in zones]
        for key in zones:
            self.walkable_areas.pop(key, None)
            self.walkable_areas_with_obstacles.pop(key, None)
        if shapes:
            removal = _union(shapes).buffer(0.02)
            for key in list(self.walkable_areas_with_obstacles):
                poly = self.walkable_areas_with_obstacles[key]
                if poly.intersects(removal):
                    new_poly = GeometryProcessor.fix_topology(poly.difference(removal))
                    if new_poly.is_empty:
                        self.walkable_areas_with_obstacles.pop(key)
                    else:
                        self.walkable_areas_with_obstacles[key] = new_poly

        for comb in self.escalator_combs:
            if comb["role"] == "entry" and comb["exit_name"] in self._initially_blocked_exits:
                (ax, ay), (bx, by) = comb["a"], comb["b"]
                nx, ny = comb["floor_normal"]
                self.blocked_exit_positions[comb["exit_name"]] = (
                    (ax + bx) / 2 + nx * 1.5,
                    (ay + by) / 2 + ny * 1.5,
                )
                logger.info(f"🚧 Pre-blocked '{comb['exit_name']}' on level {self.level_id}")

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
