"""
Geometry processing utilities for JuPedSim simulations.

Handles obstacle integration, topology fixes, and polygon operations.
"""

import shapely
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon
from shapely.ops import unary_union


class GeometryProcessor:
    """Processes geometry for JuPedSim simulations, handling obstacles and topology."""

    @staticmethod
    def fix_topology(polygon: Polygon) -> Polygon:
        """
        Fix invalid polygon topology using make_valid() or buffer(0).

        Uses Shapely's make_valid() when available (Shapely 2.0+),
        which is more robust than buffer(0) for self-intersecting polygons.

        Args:
            polygon: Input polygon that may have topology issues

        Returns:
            Topology-corrected polygon
        """
        # Try make_valid() first (Shapely 2.0+), then fall back to buffer(0)
        if hasattr(shapely, "make_valid"):
            return shapely.make_valid(polygon)
        elif hasattr(polygon, "make_valid"):
            return polygon.make_valid()
        else:
            # Fallback to buffer(0) for older Shapely versions
            return polygon.buffer(0)

    @staticmethod
    def integrate_obstacles(
        zones: dict[str, Polygon], obstacles: list[Polygon]
    ) -> tuple[dict[str, Polygon], list[Polygon]]:
        """
        Integrate obstacles into walkable zones by subtracting them.

        In JuPedSim, obstacles must be represented as holes in polygons,
        not as separate geometry. This method creates zones with obstacles
        removed using shapely difference operations.

        Args:
            zones: Dictionary mapping zone names to polygons
            obstacles: List of obstacle polygons to subtract

        Returns:
            Tuple of:
                - Dictionary mapping zone names to polygons with obstacles removed
                - List of topology-fixed obstacles that were successfully processed
        """
        # Fix any topology issues in obstacles
        fixed_obstacles = []
        for obs in obstacles:
            fixed_obs = GeometryProcessor.fix_topology(obs)
            if not fixed_obs.is_empty and fixed_obs.is_valid:
                fixed_obstacles.append(fixed_obs)

        # Subtract obstacles from zones where they intersect
        zones_with_obstacles = {}

        for zone_name, zone_polygon in zones.items():
            # Fix zone polygon topology
            zone_with_holes = GeometryProcessor.fix_topology(zone_polygon)

            # Check which obstacles intersect this zone
            for obstacle in fixed_obstacles:
                if zone_with_holes.intersects(obstacle):
                    # Subtract obstacle from zone
                    try:
                        zone_with_holes = zone_with_holes.difference(obstacle)
                        # Fix topology after difference operation (can produce invalid geometry)
                        zone_with_holes = GeometryProcessor.fix_topology(zone_with_holes)
                    except Exception as e:
                        print(f"Warning: Could not subtract obstacle from {zone_name}: {e}")

            # Final validation before adding to result
            if not zone_with_holes.is_empty and zone_with_holes.is_valid:
                # Handle MultiPolygon results (obstacles may split the zone)
                if isinstance(zone_with_holes, MultiPolygon):
                    # Keep the largest polygon if zone was split
                    zone_with_holes = max(zone_with_holes.geoms, key=lambda p: p.area)

                # Only accept proper Polygons
                if isinstance(zone_with_holes, Polygon):
                    zones_with_obstacles[zone_name] = zone_with_holes
                else:
                    print(
                        f"Warning: {zone_name} became non-Polygon after obstacle removal, skipping"
                    )
            elif not zone_with_holes.is_empty:
                # Zone is invalid even after fixes, try convex hull as last resort
                try:
                    zone_with_holes = zone_with_holes.convex_hull
                    if not zone_with_holes.is_empty:
                        zones_with_obstacles[zone_name] = zone_with_holes
                except Exception:
                    print(f"Warning: Could not recover {zone_name}, skipping")

        return zones_with_obstacles, fixed_obstacles

    @staticmethod
    def combine_geometry(polygons: list[Polygon]) -> Polygon:
        """
        Combine multiple polygons into a single geometry for JuPedSim.

        Validates and fixes all polygons before combining to ensure JuPedSim
        accepts the geometry without errors.

        Args:
            polygons: List of polygons to combine

        Returns:
            GeometryCollection or single Polygon if only one input
        """
        # Validate and fix each polygon before combining
        valid_polygons = []
        for i, poly in enumerate(polygons):
            # Fix topology
            fixed_poly = GeometryProcessor.fix_topology(poly)

            # Final validation
            if fixed_poly.is_empty:
                print(f"  Warning: Polygon {i} is empty after topology fixing, skipping")
                continue

            if not fixed_poly.is_valid:
                print(f"  Warning: Polygon {i} is still invalid after fixing")
                # Try one more aggressive fix
                fixed_poly = fixed_poly.convex_hull

            valid_polygons.append(fixed_poly)

        if not valid_polygons:
            raise ValueError("No valid polygons after geometry processing")

        if len(valid_polygons) == 1:
            return valid_polygons[0]

        combined = unary_union(valid_polygons)

        if isinstance(combined, Polygon):
            return combined

        if isinstance(combined, MultiPolygon):
            # JuPedSim requires a connected accessible area. Keep the largest component.
            largest = max(combined.geoms, key=lambda p: p.area)
            print(
                "Warning: Accessible area is disconnected. "
                "Using largest connected component to satisfy JuPedSim."
            )
            return largest

        if isinstance(combined, GeometryCollection):
            polygons_only = [g for g in combined.geoms if isinstance(g, Polygon)]
            if not polygons_only:
                raise ValueError("No polygon geometry after union")
            largest = max(polygons_only, key=lambda p: p.area)
            print(
                "Warning: Accessible area is disconnected. "
                "Using largest connected component to satisfy JuPedSim."
            )
            return largest

        raise ValueError("Unsupported geometry type after union")
