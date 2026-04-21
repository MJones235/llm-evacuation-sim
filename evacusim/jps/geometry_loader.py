"""
Load and convert SUMO walking areas to JuPedSim geometry.
"""

import xml.etree.ElementTree as ET

import shapely
from shapely.geometry import Polygon


def parse_shape_string(shape_str: str) -> list[tuple[float, float]]:
    """
    Parse SUMO shape string into list of (x, y) coordinate tuples.

    Args:
        shape_str: Space-separated string of "x,y" coordinates

    Returns:
        List of (x, y) tuples
    """
    coords = []
    for point_str in shape_str.strip().split():
        x, y = point_str.split(",")
        coords.append((float(x), float(y)))
    return coords


def load_walkable_areas(xml_path: str) -> dict[str, Polygon]:
    """
    Load walkable areas from SUMO walking_areas.add.xml file.

    Args:
        xml_path: Path to walking_areas.add.xml

    Returns:
        Dictionary mapping zone names to Shapely Polygon objects
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    walkable_areas = {}

    for poly in root.findall('.//poly[@type="jupedsim.walkable_area"]'):
        poly_id = poly.get("id")
        zone_name = poly.get("name", poly_id)
        shape_str = poly.get("shape")

        if shape_str and zone_name is not None:
            coords = parse_shape_string(shape_str)
            polygon = Polygon(coords)
            walkable_areas[zone_name] = polygon

    return walkable_areas


def load_entrance_areas(xml_path: str) -> dict[str, Polygon]:
    """
    Load entrance areas from SUMO walking_areas.add.xml file.
    Entrance areas are small zones where agents spawn.

    Args:
        xml_path: Path to walking_areas.add.xml

    Returns:
        Dictionary mapping entrance names to Shapely Polygon objects
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    entrance_areas = {}

    for poly in root.findall('.//poly[@type="jupedsim.entrance"]'):
        poly_id = poly.get("id")
        entrance_name = poly.get("name", poly_id)
        shape_str = poly.get("shape")

        if shape_str and entrance_name is not None:
            coords = parse_shape_string(shape_str)
            polygon = Polygon(coords)
            entrance_areas[entrance_name] = polygon

    return entrance_areas


def load_platform_areas(xml_path: str) -> dict[str, Polygon]:
    """
    Load platform areas from SUMO walking_areas.add.xml file.
    Platform areas are zones where agents wait (e.g., for trains).

    Args:
        xml_path: Path to walking_areas.add.xml

    Returns:
        Dictionary mapping platform names to Shapely Polygon objects
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    platform_areas = {}

    for poly in root.findall('.//poly[@type="jupedsim.platform"]'):
        poly_id = poly.get("id")
        platform_name = poly.get("name", poly_id)
        shape_str = poly.get("shape")

        if shape_str and platform_name is not None:
            coords = parse_shape_string(shape_str)
            polygon = Polygon(coords)
            platform_areas[platform_name] = polygon

    return platform_areas


def load_obstacles(xml_path: str) -> list[Polygon]:
    """
    Load obstacle polygons from SUMO walking_areas.add.xml file.

    Args:
        xml_path: Path to walking_areas.add.xml

    Returns:
        List of Shapely Polygon objects representing obstacles
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    obstacles = []

    for poly in root.findall('.//poly[@type="jupedsim.obstacle"]'):
        shape_str = poly.get("shape")

        if shape_str:
            coords = parse_shape_string(shape_str)
            polygon = Polygon(coords)
            obstacles.append(polygon)

    return obstacles


def load_train_entrance_areas(xml_path: str) -> dict[str, Polygon]:
    """
    Load train entrance areas from a level geometry XML file.

    Train entrance areas (type ``jupedsim.train_entrance``) are small polygons
    placed at the track end of each platform, representing the position where
    agents board an evacuation train.  They are registered as JuPedSim exit
    stages at initialisation time but are hidden from agent observations until
    the corresponding ``train_arrival`` event fires.

    Args:
        xml_path: Path to the level geometry XML file.

    Returns:
        Dictionary mapping entrance names (e.g. ``train_platform_1``) to
        Shapely Polygon objects.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    areas: dict[str, Polygon] = {}
    for poly in root.findall('.//poly[@type="jupedsim.train_entrance"]'):
        poly_id = poly.get("id")
        name = poly.get("name", poly_id)
        shape_str = poly.get("shape")
        if shape_str and name is not None:
            coords = parse_shape_string(shape_str)
            areas[name] = Polygon(coords)

    return areas


def load_train_track_sides(xml_path: str) -> dict[str, str]:
    """
    Load the ``track_side`` attribute from each ``jupedsim.train_entrance`` polygon.

    The ``track_side`` attribute encodes which long side of the platform the
    running track (and therefore the train body) is on.  Valid values are
    ``"left"``, ``"right"``, ``"above"``, ``"below"``.  Platforms without the
    attribute are omitted from the returned dict (callers should fall back to a
    heuristic).

    Args:
        xml_path: Path to the level geometry XML file.

    Returns:
        Dictionary mapping entrance names (e.g. ``"train_platform_1"``) to
        track-side strings (e.g. ``"right"``).
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    sides: dict[str, str] = {}
    for poly in root.findall('.//poly[@type="jupedsim.train_entrance"]'):
        name = poly.get("name") or poly.get("id")
        side = poly.get("track_side")
        if name and side:
            sides[name] = side

    return sides


def load_exit_thresholds(xml_path: str) -> dict[str, Polygon]:
    """
    Load exit threshold markers from a level geometry file.

    Thresholds (type ``jupedsim.exit_threshold``) are small polygons placed at
    the mouth of each street-exit corridor, where it opens into the main
    concourse floor.  Their centroids are used for distance and line-of-sight
    calculations instead of the far entrance polygon centroid, making exit
    visibility robust from any open-concourse position.

    Args:
        xml_path: Path to a level_*.xml geometry file

    Returns:
        Dictionary mapping exit names to Shapely Polygon objects
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    thresholds: dict[str, Polygon] = {}
    for poly in root.findall('.//poly[@type="jupedsim.exit_threshold"]'):
        name = poly.get("name", poly.get("id"))
        shape_str = poly.get("shape")
        if name and shape_str:
            coords = parse_shape_string(shape_str)
            thresholds[name] = Polygon(coords)

    return thresholds


def load_escalator_corridors(xml_path: str) -> dict[str, Polygon]:
    """
    Load escalator corridor zones from a SUMO geometry file.

    Corridors are polygons of type 'jupedsim.escalator' that cover the full
    walkable shaft of each escalator run (the area between the railing
    obstacles).  They are NOT walkable_area polygons, so they do not affect
    JuPedSim path-planning.  They are loaded purely for physics enforcement
    (minimum belt speed) in MultiLevelJuPedSimulation.

    Args:
        xml_path: Path to a level_*.xml geometry file

    Returns:
        Dictionary mapping corridor names to Shapely Polygon objects
        (e.g. 'L0_esc_corridor_b' -> Polygon(...))
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    corridors: dict[str, Polygon] = {}
    for poly in root.findall('.//poly[@type="jupedsim.escalator"]'):
        name = poly.get("name")
        shape_str = poly.get("shape")
        if name and shape_str:
            coords = parse_shape_string(shape_str)
            corridors[name] = Polygon(coords)

    return corridors


def combine_walkable_geometry(
    walkable_areas: dict[str, Polygon], obstacles: list[Polygon] | None = None
) -> tuple:
    """
    Combine all walkable areas into a single geometry for JuPedSim.

    Args:
        walkable_areas: Dictionary of zone name to Polygon
        obstacles: List of obstacle Polygons to exclude

    Returns:
        Tuple of (accessible_geometry, excluded_geometry)
    """
    polygons = list(walkable_areas.values())

    if len(polygons) == 1:
        accessible = polygons[0]
    else:
        # Return as GeometryCollection for JuPedSim
        accessible = shapely.GeometryCollection(polygons)

    # Create excluded geometry from obstacles
    if obstacles and len(obstacles) > 0:
        excluded = shapely.GeometryCollection(obstacles) if len(obstacles) > 1 else obstacles[0]
    else:
        excluded = None

    return accessible, excluded


if __name__ == "__main__":
    # Test loading
    import os

    xml_path = os.path.join(
        os.path.dirname(__file__), "..", "station_sim", "network", "walking_areas.add.xml"
    )

    walkable_areas = load_walkable_areas(xml_path)
    print(f"Loaded {len(walkable_areas)} walkable areas:")
    for name, polygon in walkable_areas.items():
        print(f"  {name}: {len(polygon.exterior.coords)} vertices, area={polygon.area:.2f}")

    entrance_areas = load_entrance_areas(xml_path)
    print(f"\nLoaded {len(entrance_areas)} entrance areas:")
    for name, polygon in entrance_areas.items():
        print(f"  {name}: {len(polygon.exterior.coords)} vertices, area={polygon.area:.2f}")

    obstacles = load_obstacles(xml_path)
    print(f"\nLoaded {len(obstacles)} obstacles")
    for idx, obs in enumerate(obstacles):
        print(f"  Obstacle {idx}: {len(obs.exterior.coords)} vertices, area={obs.area:.2f}")

    accessible, excluded = combine_walkable_geometry(walkable_areas, obstacles)
    print(f"\nAccessible geometry type: {type(accessible).__name__}")
    if excluded:
        print(f"Excluded geometry type: {type(excluded).__name__}")
