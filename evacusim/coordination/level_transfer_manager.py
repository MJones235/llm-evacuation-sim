"""Level transfer manager for multi-level station simulations.

Maintains transfer-zone polygons and explicit departure->arrival links declared
by ``jupedsim.escalator_endpoint`` metadata in level geometry XML files.
"""

from pathlib import Path
import xml.etree.ElementTree as ET

from shapely.geometry import Point, Polygon

from evacusim.utils.logger import get_logger

logger = get_logger(__name__)


class LevelTransferManager:
    """Manages escalator/stair transfers between levels."""

    def __init__(self, network_path: Path, levels: list[str]):
        self.network_path = Path(network_path)
        self.levels = [str(level) for level in levels]
        self.escalator_zones: dict[str, Polygon] = {}
        self.escalator_zones_by_level: dict[str, dict[str, Polygon]] = {}
        self.transfer_mappings: dict[str, tuple[str, str]] = {}
        self.transfer_history: dict[str, float] = {}

        self._load_escalator_zones()
        self._build_transfer_mappings()

    @staticmethod
    def _parse_shape_string(shape_str: str) -> list[tuple[float, float]]:
        coords: list[tuple[float, float]] = []
        for point_str in shape_str.strip().split():
            x, y = point_str.split(",")
            coords.append((float(x), float(y)))
        return coords

    def _load_escalator_zones(self):
        """Load transfer-zone polygons referenced by endpoint metadata."""
        for level in self.levels:
            level_file = self.network_path / f"level_{level}.xml"

            if not level_file.exists():
                logger.warning(f"Level file not found: {level_file}")
                continue

            try:
                tree = ET.parse(level_file)
                root = tree.getroot()
            except Exception as e:
                logger.error(f"Error loading escalator zones from {level_file}: {e}")
                continue

            walkable_areas: dict[str, Polygon] = {}
            for poly in root.findall('.//poly[@type="jupedsim.walkable_area"]'):
                zone_name = (poly.get("name") or "").strip()
                shape_str = (poly.get("shape") or "").strip()
                if not zone_name or not shape_str:
                    continue
                walkable_areas[zone_name] = Polygon(self._parse_shape_string(shape_str))

            level_zones: dict[str, Polygon] = {}
            for ep in root.findall('.//poly[@type="jupedsim.escalator_endpoint"]'):
                zone_name = (ep.get("transfer_zone") or "").strip()
                if not zone_name:
                    continue
                zone_poly = walkable_areas.get(zone_name)
                if zone_poly is None:
                    continue
                self.escalator_zones[zone_name] = zone_poly
                level_zones[zone_name] = zone_poly
                logger.debug(f"  Loaded escalator zone: {zone_name}")

            self.escalator_zones_by_level[str(level)] = level_zones

        logger.info(
            f"Loaded {len(self.escalator_zones)} escalator zones: {list(self.escalator_zones.keys())}"
        )

    def _build_transfer_mappings(self):
        """Build transfer mappings from endpoint departure metadata."""
        for level in self.levels:
            level_file = self.network_path / f"level_{level}.xml"
            if not level_file.exists():
                continue

            try:
                tree = ET.parse(level_file)
                root = tree.getroot()
            except Exception as e:
                logger.error(f"Error building transfer mappings from {level_file}: {e}")
                continue

            for ep in root.findall('.//poly[@type="jupedsim.escalator_endpoint"]'):
                role = (ep.get("role") or "").strip().lower()
                if role != "departure":
                    continue

                from_zone = (ep.get("transfer_zone") or "").strip()
                to_zone = (ep.get("transfer_to_zone") or "").strip()
                if not from_zone or not to_zone:
                    continue

                to_level = ""
                for candidate_level, zones in self.escalator_zones_by_level.items():
                    if to_zone in zones:
                        to_level = candidate_level
                        break

                if not to_level:
                    logger.warning(
                        f"Transfer mapping target zone '{to_zone}' not found for source '{from_zone}'"
                    )
                    continue

                self.transfer_mappings[from_zone] = (to_zone, to_level)
                logger.info(f"  Created transfer: {from_zone} -> {to_zone} (L{to_level})")

        logger.info(f"Built {len(self.transfer_mappings)} transfer mappings")

    def check_transfer(
        self,
        agent_id: str,
        position: tuple[float, float],
        current_zone: str,
        current_level: str,
        current_time: float,
    ) -> tuple[str, str, tuple[float, float]] | None:
        """Check if an agent in a transfer zone should move to another level."""
        cooldown_time = 2.0
        if agent_id in self.transfer_history:
            if current_time - self.transfer_history[agent_id] < cooldown_time:
                return None

        if current_zone not in self.transfer_mappings:
            return None

        target_zone, target_level = self.transfer_mappings[current_zone]
        source_poly = self.escalator_zones[current_zone]
        target_poly = self.escalator_zones[target_zone]

        point = Point(position)
        buffered_source = source_poly.buffer(3.0)
        is_inside = source_poly.contains(point)
        is_near = buffered_source.contains(point) if not is_inside else False
        if not (is_inside or is_near):
            return None

        new_position = self._map_position_safe(source_poly, target_poly, position)
        self.transfer_history[agent_id] = current_time

        logger.debug(
            f"Transfer {agent_id}: {current_zone} (L{current_level}) -> "
            f"{target_zone} (L{target_level}) at pos {position} -> {new_position}"
        )

        return target_zone, target_level, new_position

    def _map_position_safe(
        self, source_poly: Polygon, target_poly: Polygon, source_pos: tuple[float, float]
    ) -> tuple[float, float]:
        """Map source transfer-zone relative position into target zone."""
        source_bounds = source_poly.bounds
        target_bounds = target_poly.bounds

        sx_min, sy_min, sx_max, sy_max = source_bounds
        tx_min, ty_min, tx_max, ty_max = target_bounds

        sx_range = sx_max - sx_min if sx_max > sx_min else 1.0
        sy_range = sy_max - sy_min if sy_max > sy_min else 1.0
        tx_range = tx_max - tx_min if tx_max > tx_min else 1.0
        ty_range = ty_max - ty_min if ty_max > ty_min else 1.0

        norm_x = (source_pos[0] - sx_min) / sx_range
        norm_y = (source_pos[1] - sy_min) / sy_range

        target_x = tx_min + norm_x * tx_range
        target_y = ty_min + norm_y * ty_range

        eroded_target = target_poly.buffer(-0.3)
        test_point = Point(target_x, target_y)

        if not eroded_target.is_empty and eroded_target.contains(test_point):
            return (target_x, target_y)

        if not eroded_target.is_empty:
            centroid = eroded_target.centroid
            return (centroid.x, centroid.y)

        centroid = target_poly.centroid
        return (centroid.x, centroid.y)

    def get_transfer_info(self) -> dict:
        """Get debugging info about transfers."""
        return {
            "escalator_zones_count": len(self.escalator_zones),
            "transfer_mappings_count": len(self.transfer_mappings),
            "zones": list(self.escalator_zones.keys()),
            "transfers": {k: f"{v[0]} (L{v[1]})" for k, v in self.transfer_mappings.items()},
        }

    def get_level_zones(self, level_id: str) -> dict[str, Polygon]:
        """Return transfer-zone polygons for one level."""
        return self.escalator_zones_by_level.get(str(level_id), {})
