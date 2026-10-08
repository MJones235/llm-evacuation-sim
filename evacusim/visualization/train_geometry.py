"""Geometry helpers for drawing trains alongside station platforms."""

from __future__ import annotations

import numpy as np
from shapely.geometry import Polygon
from shapely.ops import unary_union


def compute_train_polygons(
    geometry: dict,
    train_width_m: float = 2.8,
    gap_m: float = 0.35,
    end_inset_m: float = 1.0,
) -> dict[str, list[tuple[float, float]]]:
    """Build one platform-aligned train rectangle per named platform.

    Each rectangle follows the platform's principal axis. Of the two possible
    sides, the one overlapping least with the station's walkable floor is used
    as the track side.
    """
    if not geometry:
        return {}
    level = geometry.get("levels", {}).get("level_-1", geometry)
    walkable = level.get("walkable_areas", {})
    floor_polygons = [Polygon(coords).buffer(0) for coords in walkable.values() if len(coords) >= 3]
    floor = unary_union(floor_polygons) if floor_polygons else None

    trains: dict[str, list[tuple[float, float]]] = {}
    for platform_name, coords in walkable.items():
        if not platform_name.startswith("platform_") or len(coords) < 3:
            continue

        points = np.asarray(coords, dtype=float)
        if np.allclose(points[0], points[-1]):
            points = points[:-1]
        center = points.mean(axis=0)
        covariance = np.cov(points - center, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        along = eigenvectors[:, int(np.argmax(eigenvalues))]
        normal = np.array([-along[1], along[0]])

        along_projection = (points - center) @ along
        normal_projection = (points - center) @ normal
        along_min = float(along_projection.min() + end_inset_m)
        along_max = float(along_projection.max() - end_inset_m)
        if along_min >= along_max:
            along_min = float(along_projection.min())
            along_max = float(along_projection.max())

        candidates = []
        for sign, edge in ((-1.0, normal_projection.min()), (1.0, normal_projection.max())):
            near = float(edge + sign * gap_m)
            far = float(near + sign * train_width_m)
            corners = [
                center + along * along_min + normal * near,
                center + along * along_max + normal * near,
                center + along * along_max + normal * far,
                center + along * along_min + normal * far,
            ]
            polygon = Polygon(corners)
            overlap = polygon.intersection(floor).area if floor is not None else 0.0
            candidates.append((overlap, corners))

        _, best = min(candidates, key=lambda candidate: candidate[0])
        platform_num = platform_name.rsplit("_", 1)[-1]
        trains[f"train_platform_{platform_num}"] = [
            (float(point[0]), float(point[1])) for point in best
        ]

    return trains
