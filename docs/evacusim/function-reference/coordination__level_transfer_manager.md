# Function Reference: evacusim/coordination/level_transfer_manager.py

- Source file: [evacusim/coordination/level_transfer_manager.py](../../../evacusim/coordination/level_transfer_manager.py)
- Top-level classes: 1
- Top-level functions: 0

## Module Summary

Level transfer manager for multi-level station simulations.

## Classes and Methods

### Class LevelTransferManager

- Location: [evacusim/coordination/level_transfer_manager.py#L29](../../../evacusim/coordination/level_transfer_manager.py#L29)
- Summary: Manages escalator/stair transfers between levels.
- Method count: 7

#### __init__

- Signature: def __init__(self, network_path, levels)
- Location: [evacusim/coordination/level_transfer_manager.py#L32](../../../evacusim/coordination/level_transfer_manager.py#L32)
- Summary: Initialize transfer manager.
- Key calls: Path, self._load_escalator_zones, self._build_transfer_mappings

#### _load_escalator_zones

- Signature: def _load_escalator_zones(self)
- Location: [evacusim/coordination/level_transfer_manager.py#L53](../../../evacusim/coordination/level_transfer_manager.py#L53)
- Summary: Load all escalator zones from geometry files.
- Key calls: re.compile, logger.info, level_file.exists, logger.warning, ET.parse, tree.getroot, root.findall, poly.get, escalator_pattern.match, self._parse_shape_string

#### _build_transfer_mappings

- Signature: def _build_transfer_mappings(self)
- Location: [evacusim/coordination/level_transfer_manager.py#L100](../../../evacusim/coordination/level_transfer_manager.py#L100)
- Summary: Build automatic transfer mappings from escalator zones.
- Key calls: re.compile, self.escalator_zones.keys, escalators_by_key.items, logger.info, escalator_pattern.match, match.groups, sorted, range, len, logger.warning

#### check_transfer

- Signature: def check_transfer(self, agent_id, position, current_zone, current_level, current_time) -> tuple[str, str, tuple[float, float]] | None
- Location: [evacusim/coordination/level_transfer_manager.py#L154](../../../evacusim/coordination/level_transfer_manager.py#L154)
- Summary: Check if agent should transfer to another level.
- Key calls: Point, source_poly.buffer, source_poly.contains, self._map_position_safe, logger.debug, buffered_source.contains

#### _map_position_safe

- Signature: def _map_position_safe(self, source_poly, target_poly, source_pos) -> tuple[float, float]
- Location: [evacusim/coordination/level_transfer_manager.py#L224](../../../evacusim/coordination/level_transfer_manager.py#L224)
- Summary: Map agent position from source escalator polygon to target escalator polygon.
- Key calls: target_poly.buffer, Point, logger.debug, eroded_target.contains

#### _parse_shape_string

- Signature: def _parse_shape_string(self, shape_str) -> list[tuple[float, float]]
- Location: [evacusim/coordination/level_transfer_manager.py#L291](../../../evacusim/coordination/level_transfer_manager.py#L291)
- Summary: Parse SUMO shape string into coordinate list.
- Key calls: split, point_str.split, coords.append, shape_str.strip, float

#### get_transfer_info

- Signature: def get_transfer_info(self) -> dict
- Location: [evacusim/coordination/level_transfer_manager.py#L299](../../../evacusim/coordination/level_transfer_manager.py#L299)
- Summary: Get debugging info about transfers.
- Key calls: len, list, self.escalator_zones.keys, self.transfer_mappings.items

## Coverage Notes

- Nested helper functions documented in this file: 0
