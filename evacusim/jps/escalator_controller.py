"""Escalator controller driven by explicit geometry metadata.

This controller has no naming-convention inference. Every escalator transfer
and in-corridor movement rule is declared in level geometry files via
``jupedsim.escalator_endpoint`` polygons.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import xml.etree.ElementTree as ET
from typing import Any

from shapely.geometry import Point, Polygon

from evacusim.utils.logger import get_logger

logger = get_logger(__name__)


class EscalatorAgentState(str, Enum):
    """Agent movement state relative to escalator lifecycle."""

    NORMAL = "normal_navigation"
    APPROACHING = "approaching_escalator"
    BOARDING = "boarding"
    IN_TRANSIT = "in_transit"
    DISEMBARKING = "disembarking"


@dataclass(frozen=True)
class EscalatorEndpoint:
    """One explicitly declared endpoint record from geometry metadata."""

    level_id: str
    endpoint_id: str
    role: str
    transfer_zone_name: str
    transfer_zone_polygon: Polygon
    corridor_name: str
    exit_name: str | None
    transfer_to_zone_name: str | None
    spawn_point: tuple[float, float] | None
    egress_target: tuple[float, float] | None


@dataclass(frozen=True)
class EscalatorEdge:
    """Directed transfer edge from a departure endpoint to an arrival endpoint."""

    from_endpoint_id: str
    to_endpoint_id: str
    from_level: str
    to_level: str
    from_exit_name: str
    from_zone_name: str
    to_zone_name: str
    from_corridor_name: str
    to_corridor_name: str
    to_spawn_point: tuple[float, float]
    to_egress_target: tuple[float, float]


@dataclass
class EscalatorRegistry:
    """Resolved escalator model from explicit metadata."""

    endpoints_by_zone: dict[str, EscalatorEndpoint]
    endpoints_by_level: dict[str, list[EscalatorEndpoint]]
    corridor_endpoints_by_level: dict[tuple[str, str], list[EscalatorEndpoint]]
    edges: list[EscalatorEdge]
    edges_by_exit: dict[tuple[str, str], EscalatorEdge]

    @property
    def escalator_ids(self) -> list[str]:
        return sorted({edge.from_endpoint_id for edge in self.edges})


class EscalatorController:
    """Escalator transfer and motion controller.

    Source of truth is metadata of type ``jupedsim.escalator_endpoint`` in each
    level_*.xml file.
    """

    def __init__(
        self,
        network_path: Path,
        levels: list[str],
        simulations: dict[str, Any],
    ):
        self.network_path = Path(network_path)
        self.levels = [str(level) for level in levels]
        self.simulations = simulations
        self.endpoints: list[EscalatorEndpoint] = self._load_endpoints_from_geometry()
        self.registry = self._build_registry()
        self.agent_states: dict[str, EscalatorAgentState] = {}
        self._agent_motion_context: dict[str, str] = {}

    @staticmethod
    def _parse_shape_string(shape_str: str) -> list[tuple[float, float]]:
        coords: list[tuple[float, float]] = []
        for point_str in shape_str.strip().split():
            x, y = point_str.split(",")
            coords.append((float(x), float(y)))
        return coords

    def _load_endpoints_from_geometry(self) -> list[EscalatorEndpoint]:
        """Load explicit endpoint records from level geometry XML files."""
        endpoints: list[EscalatorEndpoint] = []

        for level_id in self.levels:
            level_file = self.network_path / f"level_{level_id}.xml"
            if not level_file.exists():
                logger.warning(f"Missing geometry file for level {level_id}: {level_file}")
                continue

            try:
                tree = ET.parse(level_file)
                root = tree.getroot()
            except Exception as exc:
                logger.error(f"Could not parse geometry file {level_file}: {exc}")
                continue

            walkable_areas: dict[str, Polygon] = {}
            for wpoly in root.findall('.//poly[@type="jupedsim.walkable_area"]'):
                wname = (wpoly.get("name") or "").strip()
                wshape = (wpoly.get("shape") or "").strip()
                if not wname or not wshape:
                    continue
                try:
                    walkable_areas[wname] = Polygon(self._parse_shape_string(wshape))
                except Exception:
                    continue

            for poly in root.findall('.//poly[@type="jupedsim.escalator_endpoint"]'):
                endpoint_id = (
                    (poly.get("endpoint_id") or "").strip()
                    or (poly.get("name") or "").strip()
                    or (poly.get("id") or "").strip()
                )
                role = (poly.get("role") or "").strip().lower()
                zone_name = (poly.get("transfer_zone") or "").strip()
                corridor_name = (poly.get("corridor") or "").strip()
                exit_name = (poly.get("exit_name") or "").strip()
                transfer_to_zone_name = (poly.get("transfer_to_zone") or "").strip()

                if not endpoint_id:
                    logger.warning(
                        f"Skipping escalator endpoint without endpoint_id/name in {level_file.name}"
                    )
                    continue

                if role not in {"departure", "arrival"}:
                    logger.warning(
                        f"Skipping invalid escalator endpoint in {level_file.name}: "
                        f"endpoint_id={endpoint_id!r}, role={role!r}"
                    )
                    continue
                if not zone_name or not corridor_name:
                    logger.warning(
                        f"Skipping endpoint with missing transfer_zone/corridor in {level_file.name}: "
                        f"endpoint_id={endpoint_id}, role={role}"
                    )
                    continue
                zone_polygon = walkable_areas.get(zone_name)
                if zone_polygon is None:
                    logger.warning(
                        f"Skipping endpoint with unknown transfer_zone in {level_file.name}: "
                        f"endpoint_id={endpoint_id}, transfer_zone={zone_name!r}"
                    )
                    continue

                if role == "departure" and (not exit_name or not transfer_to_zone_name):
                    logger.warning(
                        f"Skipping departure endpoint with missing exit_name/transfer_to_zone in "
                        f"{level_file.name}: endpoint_id={endpoint_id}"
                    )
                    continue

                spawn_point: tuple[float, float] | None = None
                sx = poly.get("spawn_x")
                sy = poly.get("spawn_y")
                if sx is not None and sy is not None:
                    try:
                        spawn_point = (float(sx), float(sy))
                    except ValueError:
                        logger.warning(
                            f"Invalid spawn coordinates in {level_file.name}: "
                            f"endpoint_id={endpoint_id}, spawn_x={sx!r}, spawn_y={sy!r}"
                        )

                egress_target: tuple[float, float] | None = None
                if role == "arrival":
                    ex = poly.get("egress_x")
                    ey = poly.get("egress_y")
                    if ex is None or ey is None:
                        logger.warning(
                            f"Arrival endpoint missing egress point in {level_file.name}: "
                            f"endpoint_id={endpoint_id}"
                        )
                    else:
                        try:
                            egress_target = (float(ex), float(ey))
                        except ValueError:
                            logger.warning(
                                f"Invalid arrival egress coordinates in {level_file.name}: "
                                f"egress_x={ex!r}, egress_y={ey!r}"
                            )

                    if spawn_point is None:
                        logger.warning(
                            f"Arrival endpoint missing spawn point in {level_file.name}: "
                            f"endpoint_id={endpoint_id}"
                        )

                endpoints.append(
                    EscalatorEndpoint(
                        level_id=str(level_id),
                        endpoint_id=endpoint_id,
                        role=role,
                        transfer_zone_name=zone_name,
                        transfer_zone_polygon=zone_polygon,
                        corridor_name=corridor_name,
                        exit_name=exit_name or None,
                        transfer_to_zone_name=transfer_to_zone_name or None,
                        spawn_point=spawn_point,
                        egress_target=egress_target,
                    )
                )

        return endpoints

    def _build_registry(self) -> EscalatorRegistry:
        endpoints_by_zone: dict[str, EscalatorEndpoint] = {}
        endpoints_by_level: dict[str, list[EscalatorEndpoint]] = {}
        corridor_endpoints_by_level: dict[tuple[str, str], list[EscalatorEndpoint]] = {}
        edges: list[EscalatorEdge] = []
        edges_by_exit: dict[tuple[str, str], EscalatorEdge] = {}

        for ep in self.endpoints:
            endpoints_by_zone[ep.transfer_zone_name] = ep
            endpoints_by_level.setdefault(ep.level_id, []).append(ep)
            corridor_endpoints_by_level.setdefault((ep.level_id, ep.corridor_name), []).append(ep)

        arrivals_by_zone: dict[str, EscalatorEndpoint] = {
            ep.transfer_zone_name: ep for ep in self.endpoints if ep.role == "arrival"
        }
        for dep in (ep for ep in self.endpoints if ep.role == "departure"):
            if dep.exit_name is None or dep.transfer_to_zone_name is None:
                continue
            arr = arrivals_by_zone.get(dep.transfer_to_zone_name)
            if arr is None or arr.spawn_point is None or arr.egress_target is None:
                continue

            edge = EscalatorEdge(
                from_endpoint_id=dep.endpoint_id,
                to_endpoint_id=arr.endpoint_id,
                from_level=dep.level_id,
                to_level=arr.level_id,
                from_exit_name=dep.exit_name,
                from_zone_name=dep.transfer_zone_name,
                to_zone_name=arr.transfer_zone_name,
                from_corridor_name=dep.corridor_name,
                to_corridor_name=arr.corridor_name,
                to_spawn_point=arr.spawn_point,
                to_egress_target=arr.egress_target,
            )
            edges.append(edge)
            edges_by_exit[(dep.level_id, dep.exit_name)] = edge

        return EscalatorRegistry(
            endpoints_by_zone=endpoints_by_zone,
            endpoints_by_level=endpoints_by_level,
            corridor_endpoints_by_level=corridor_endpoints_by_level,
            edges=edges,
            edges_by_exit=edges_by_exit,
        )

    def validate_registry(self) -> list[str]:
        """Return validation issues. Empty list means the model is consistent."""
        issues: list[str] = []

        if not self.endpoints:
            issues.append("No escalator endpoints were loaded from geometry metadata.")

        if not self.registry.edges:
            issues.append("No escalator transfer edges were built.")

        for edge in self.registry.edges:
            if edge.from_level == edge.to_level:
                issues.append(
                    f"Edge {edge.from_zone_name} -> {edge.to_zone_name} stays on the same level."
                )

        for ep in self.endpoints:
            if ep.transfer_zone_polygon.is_empty:
                issues.append(
                    f"Endpoint zone polygon is empty for endpoint '{ep.endpoint_id}'."
                )
            level_sim = self.simulations.get(ep.level_id)
            if level_sim is None:
                issues.append(f"Endpoint level '{ep.level_id}' has no simulation instance.")
                continue
            corridors = getattr(level_sim.geometry_manager, "escalator_corridors", {})
            if ep.corridor_name not in corridors:
                issues.append(
                    f"Endpoint corridor '{ep.corridor_name}' missing on level {ep.level_id}."
                )
            if ep.role == "departure":
                if ep.exit_name is None:
                    issues.append(f"Departure endpoint '{ep.endpoint_id}' missing exit_name.")
                else:
                    available_exits = getattr(level_sim.exit_manager, "evacuation_exits", {})
                    if ep.exit_name not in available_exits:
                        blocked_letters = getattr(level_sim.exit_manager, "_blocked_esc_letters", set())
                        parts = ep.exit_name.split("_")
                        letter = parts[1] if len(parts) >= 2 and parts[0] == "escalator" else None
                        if letter not in blocked_letters:
                            issues.append(
                                f"Departure endpoint '{ep.endpoint_id}' exit '{ep.exit_name}' "
                                f"not registered on level {ep.level_id}."
                            )
                if ep.transfer_to_zone_name is None:
                    issues.append(
                        f"Departure endpoint '{ep.endpoint_id}' missing transfer_to_zone."
                    )
            if ep.role == "arrival" and ep.spawn_point is None:
                issues.append(
                    f"Arrival endpoint missing spawn point: endpoint_id={ep.endpoint_id}."
                )
            if ep.role == "arrival" and ep.egress_target is None:
                issues.append(
                    f"Arrival endpoint missing egress point: endpoint_id={ep.endpoint_id}."
                )

        endpoint_ids = [ep.endpoint_id for ep in self.endpoints]
        if len(endpoint_ids) != len(set(endpoint_ids)):
            issues.append("Duplicate escalator endpoint_id values detected.")

        zones = [ep.transfer_zone_name for ep in self.endpoints]
        if len(zones) != len(set(zones)):
            issues.append("Duplicate transfer_zone assignment across endpoints.")

        for dep in (ep for ep in self.endpoints if ep.role == "departure"):
            if dep.transfer_to_zone_name is None:
                continue
            arrival = self.registry.endpoints_by_zone.get(dep.transfer_to_zone_name)
            if arrival is None:
                issues.append(
                    f"Departure endpoint '{dep.endpoint_id}' references missing target zone "
                    f"'{dep.transfer_to_zone_name}'."
                )
                continue
            if arrival.role != "arrival":
                issues.append(
                    f"Departure endpoint '{dep.endpoint_id}' targets zone "
                    f"'{dep.transfer_to_zone_name}' that is not an arrival endpoint."
                )

        return issues

    def summary(self) -> dict[str, Any]:
        """Compact startup summary for logs and diagnostics."""
        return {
            "endpoint_count": len(self.registry.endpoints_by_zone),
            "edge_count": len(self.registry.edges),
            "escalator_ids": self.registry.escalator_ids,
            "arrival_egress_target_count": len(
                [e for e in self.endpoints if e.role == "arrival" and e.egress_target is not None]
            ),
            "validation_issue_count": len(self.validate_registry()),
        }

    def get_edge_for_exit(self, level_id: str, exit_name: str) -> EscalatorEdge | None:
        """Resolve transfer edge by explicit (level_id, exit_name) mapping."""
        return self.registry.edges_by_exit.get((str(level_id), exit_name))

    def get_zone_polygon(self, zone_name: str):
        """Return the transfer-zone polygon for a zone name, if present."""
        endpoint = self.registry.endpoints_by_zone.get(zone_name)
        if endpoint is None:
            return None
        return endpoint.transfer_zone_polygon

    def get_level_zone_names(self, level_id: str) -> list[str]:
        """Return all escalator zone names on a level."""
        return [
            ep.transfer_zone_name for ep in self.registry.endpoints_by_level.get(str(level_id), [])
        ]

    def get_spawn_point_for_edge(self, edge: EscalatorEdge) -> tuple[float, float]:
        """Return explicit arrival spawn point for a transfer edge."""
        return edge.to_spawn_point

    def enforce_motion_for_agent(
        self,
        agent_id: str,
        level_id: str,
        position: tuple[float, float],
        level_sim: Any,
    ) -> bool:
        """Apply deterministic escalator direction control for one agent.

        Returns True if escalator geometry handling was applied, else False.
        """
        p = Point(position)
        level_key = str(level_id)

        # 1) Zone-based handling (strongest signal; includes explicit role).
        level_endpoints = self.registry.endpoints_by_level.get(level_key, [])
        for endpoint in level_endpoints:
            zone_poly = self.get_zone_polygon(endpoint.transfer_zone_name)
            if zone_poly is None:
                continue
            if not zone_poly.contains(p):
                continue

            self._apply_endpoint_motion(agent_id, endpoint, level_sim)
            return True

        # 2) Corridor-based handling (continue last known escalator direction).
        corridors = getattr(level_sim.geometry_manager, "escalator_corridors", {})
        for corridor_name, corridor_poly in corridors.items():
            if not corridor_poly.contains(p):
                continue

            corridor_endpoints = self.registry.corridor_endpoints_by_level.get(
                (level_key, corridor_name), []
            )
            if not corridor_endpoints:
                return True

            selected_endpoint: EscalatorEndpoint | None = None
            context_endpoint_id = self._agent_motion_context.get(agent_id)
            if context_endpoint_id is not None:
                selected_endpoint = next(
                    (ep for ep in corridor_endpoints if ep.endpoint_id == context_endpoint_id),
                    None,
                )
            if selected_endpoint is None:
                selected_endpoint = next(
                    (ep for ep in corridor_endpoints if ep.role == "departure"),
                    corridor_endpoints[0],
                )

            self._apply_endpoint_motion(agent_id, selected_endpoint, level_sim)
            return True

        self._agent_motion_context.pop(agent_id, None)
        return False

    def _apply_endpoint_motion(
        self,
        agent_id: str,
        endpoint: EscalatorEndpoint,
        level_sim: Any,
    ) -> None:
        """Apply deterministic movement control for departure-role endpoints only.

        Arrival-role endpoints are intentionally skipped: transferred agents receive
        a random destination waypoint at spawn time that routes them clear of the
        landing zone. Overriding that journey here would cause them to stop at the
        corridor egress point instead of walking through to their destination.
        """
        self._agent_motion_context[agent_id] = endpoint.endpoint_id
        if endpoint.role == "departure" and endpoint.exit_name is not None:
            if endpoint.exit_name in level_sim.exit_manager.evacuation_exits:
                # Avoid redundant journey resets while the agent remains inside
                # the same corridor/zone. Reissuing identical journeys every step
                # increases route-churn under congestion and can trap agents near
                # transfer mouths without ever crossing the exit threshold.
                current_exit = getattr(level_sim, "agent_assigned_exits", {}).get(agent_id)
                if current_exit != endpoint.exit_name:
                    level_sim.set_agent_destination_exit(agent_id, endpoint.exit_name)

    def set_agent_state(self, agent_id: str, state: EscalatorAgentState) -> None:
        """State primitive for later lifecycle-driven control."""
        self.agent_states[agent_id] = state

    def get_agent_state(self, agent_id: str) -> EscalatorAgentState:
        """Return known state or NORMAL by default."""
        return self.agent_states.get(agent_id, EscalatorAgentState.NORMAL)

    def clear_agent_state(self, agent_id: str) -> None:
        """Remove per-agent escalator state."""
        self.agent_states.pop(agent_id, None)
