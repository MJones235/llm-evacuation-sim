"""
Station layout builder for Station Concordia simulations.

This module is responsible for:
- Building station layout dictionary from simulation geometry
- Processing entrance and platform areas
- Creating zone definitions
- Handling walkable areas and obstacles
"""

from typing import Any

from evacusim.utils.logger import get_logger
from evacusim.jps.simulation_interface import PedestrianSimulation

logger = get_logger(__name__)


class StationLayoutBuilder:
    """Handles creation of station layout from simulation geometry."""

    @staticmethod
    def build_layout(jps_sim: PedestrianSimulation, config: dict) -> dict[str, Any]:
        """
        Build station layout dictionary from pedestrian simulation geometry.

        Args:
            jps_sim: Pedestrian simulation instance (implements PedestrianSimulation)
            config: Configuration dictionary

        Returns:
            Dictionary containing station layout information including:
            - exits: Dictionary of exit names to (x, y) coordinates
            - exits_polygons: Dictionary of exit area polygons
            - exits: Dictionary of exit names to (x, y) coordinates
            - exits_polygons: Dictionary of exit area polygons
            - walkable_areas: Dictionary of walkable area polygons
            - zones: Dictionary of zone boundaries
            - zones_polygons: Dictionary of zone polygons
            - obstacles: List of obstacle polygons
        """
        # For multi-level simulations, consolidate exits from all levels
        if hasattr(jps_sim, "simulations"):
            # Multi-level: Consolidate exits and zones from ALL levels
            all_exits = {}
            all_exit_polygons = {}
            all_zones = {}
            all_zone_polygons = {}
            zone_labels_cfg: dict[str, str] = config.get("station", {}).get("zone_labels", {})

            # Collect exits from each level
            for level_id in sorted(jps_sim.simulations.keys()):
                level_sim = jps_sim.simulations[level_id]
                gm = level_sim.geometry_manager

                # Add street exits (from entrance areas)
                for name, poly in gm.entrance_areas.items():
                    all_exits[name] = (poly.centroid.x, poly.centroid.y)
                    all_exit_polygons[name] = poly

                # Add zones from explicitly-typed platform areas (jupedsim.platform)
                for zone_name, zone_poly in gm.platform_areas.items():
                    zone_key = (
                        f"{zone_name}_L{level_id}" if len(jps_sim.simulations) > 1 else zone_name
                    )
                    all_zones[zone_key] = StationLayoutBuilder._polygon_bounds(zone_poly)
                    all_zone_polygons[zone_key] = zone_poly

                # Also harvest platform zones from walkable areas whose names start
                # with "platform_" (the monument geometry uses jupedsim.walkable_area
                # type for these polys, so they are not detected by platform_areas).
                # Additionally, harvest any named walkable area referenced in zone_labels
                # (e.g. "concourse") so that configs can use those names in patrol_zones.
                for wa_name, wa_poly in gm.walkable_areas_with_obstacles.items():
                    if wa_name not in all_zone_polygons and (
                        wa_name.startswith("platform_") or wa_name in zone_labels_cfg
                    ):
                        all_zones[wa_name] = StationLayoutBuilder._polygon_bounds(wa_poly)
                        all_zone_polygons[wa_name] = wa_poly

            # Add escalator exits from all levels
            for level_sim in jps_sim.simulations.values():
                for exit_name in level_sim.exit_manager.evacuation_exits:
                    if exit_name.startswith("escalator_"):
                        all_exits[exit_name] = level_sim.exit_manager.exit_coordinates.get(
                            exit_name, (0, 0)
                        )

            # Collect per-level walkable areas for level-aware zone identification
            all_walkable_by_level = {
                str(level_id): level_sim.geometry_manager.walkable_areas_with_obstacles
                for level_id, level_sim in jps_sim.simulations.items()
            }
        else:
            # Single-level: Use geometry manager and exit manager
            gm = jps_sim.geometry_manager
            all_exits = {
                name: (poly.centroid.x, poly.centroid.y) for name, poly in gm.entrance_areas.items()
            }
            all_exit_polygons = gm.entrance_areas
            all_zones = StationLayoutBuilder._build_zones(jps_sim, gm.platform_areas)
            all_zone_polygons = StationLayoutBuilder._build_zone_polygons(
                jps_sim, gm.platform_areas
            )

        # Build down-access exits: concourse-level escalator zones that lead to platforms.
        # These are walkable-area level-transfer triggers, not JPS evacuation exits.
        # Exposing them as named exits lets agents explicitly choose between escalators,
        # stairs, lifts, etc. rather than having routing silently redirected.
        import re as _re

        down_access_exits: dict[str, tuple[float, float]] = {}
        custom_exit_display_names: dict[str, str] = {}
        platform_down_cfg: dict[str, list[str]] = config.get("station", {}).get(
            "platform_down_exits", {}
        )
        if platform_down_cfg and hasattr(jps_sim, "simulations"):
            # Invert config: esc_zone_name -> [platform names it serves]
            esc_to_platforms: dict[str, list[str]] = {}
            for plat_name, esc_zones in platform_down_cfg.items():
                for esc_zone in esc_zones:
                    esc_to_platforms.setdefault(esc_zone, []).append(plat_name)

            level_0_areas = all_walkable_by_level.get("0", {})
            for esc_zone, platforms in esc_to_platforms.items():
                if esc_zone in level_0_areas:
                    poly = level_0_areas[esc_zone]
                    down_access_exits[esc_zone] = (poly.centroid.x, poly.centroid.y)

                    m = _re.match(r"L[^_]+_esc_([a-f])_down", esc_zone)
                    if m:
                        letter = m.group(1).upper()
                        # Only include specific-numbered platforms (platform_N) in the label.
                        # Strip any parenthetical suffix from the zone label so we get a
                        # short name like "Platform 3" rather than the full
                        # "Platform 3 (Escalators B and C go up to the concourse)" label,
                        # which would produce a confusingly nested display string.
                        plat_labels = sorted(
                            {
                                zone_labels_cfg.get(p, p.replace("_", " ").title()).split("(")[0].strip()
                                for p in platforms
                                if _re.match(r"^platform_\d+$", p)
                            }
                        )
                        if plat_labels:
                            dest = " & ".join(plat_labels)
                            display = f"Escalator {letter} (down to {dest})"
                        else:
                            display = f"Escalator {letter} (going down)"
                        custom_exit_display_names[esc_zone] = display

        station_layout = {
            **config.get("station", {}),
            "exits": all_exits,
            "exits_polygons": all_exit_polygons,
            "walkable_areas": jps_sim.geometry_manager.walkable_areas_with_obstacles,
            "walkable_areas_by_level": (
                all_walkable_by_level if hasattr(jps_sim, "simulations") else {}
            ),
            "zones": all_zones,
            "zones_polygons": all_zone_polygons,
            "obstacles": jps_sim.geometry_manager.obstacles,
            # Concourse-level escalator zones leading to platforms (zone_name -> centroid)
            "down_access_exits": down_access_exits,
            # Custom display names for exits needing non-default labels.
            # YAML config entries (e.g. train platform names) take the base, then
            # code-generated escalator labels (more specific) override on top.
            "custom_exit_display_names": {
                **config.get("station", {}).get("custom_exit_display_names", {}),
                **custom_exit_display_names,
            },
        }

        logger.info(
            f"Built station layout with {len(all_exits)} exits (street + escalators) and {len(all_zones)} zones"
        )
        return station_layout

    @staticmethod
    def _build_zones(
        jps_sim: PedestrianSimulation, platform_areas: dict
    ) -> dict[str, dict[str, float]]:
        """
        Build zone boundary definitions.

        Args:
            jps_sim: Pedestrian simulation instance (implements PedestrianSimulation)
            platform_areas: Dictionary of platform area polygons

        Returns:
            Dictionary mapping zone names to boundary dictionaries
        """
        if platform_areas:
            return {
                name: StationLayoutBuilder._polygon_bounds(poly)
                for name, poly in platform_areas.items()
            }
        else:
            # Fallback to main walkable area
            main_area = list(jps_sim.geometry_manager.walkable_areas.values())[0]
            return {"main_area": StationLayoutBuilder._polygon_bounds(main_area)}

    @staticmethod
    def _build_zone_polygons(jps_sim, platform_areas: dict) -> dict[str, Any]:
        """
        Build zone polygon definitions.

        Args:
            jps_sim: JuPedSim simulation instance
            platform_areas: Dictionary of platform area polygons

        Returns:
            Dictionary mapping zone names to polygons
        """
        if platform_areas:
            return platform_areas
        else:
            # Fallback to main walkable area
            main_area = list(jps_sim.geometry_manager.walkable_areas.values())[0]
            return {"main_area": main_area}

    @staticmethod
    def _polygon_bounds(polygon) -> dict[str, float]:
        """
        Extract bounding box from a polygon.

        Args:
            polygon: Shapely polygon

        Returns:
            Dictionary with x_min, x_max, y_min, y_max keys
        """
        min_x, min_y, max_x, max_y = polygon.bounds
        return {"x_min": min_x, "x_max": max_x, "y_min": min_y, "y_max": max_y}
