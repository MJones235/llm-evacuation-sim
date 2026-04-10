"""
Population monitor for Station Concordia simulations.

Tracks the number of people in each station zone over time, recording
snapshots at configurable intervals throughout the simulation.  At the end
of the run it can display a formatted time-series table and save the data
to CSV and JSON files for further analysis.

Zone definitions are supplied via configuration and are therefore fully
general — the same monitor works for any station scenario without code
changes.

Zone spec format
----------------
Each zone entry in the config ``monitoring.zones`` list supports the
following keys:

``name`` (str, required)
    Identifier used in the output data files.
``description`` (str, optional)
    Human-readable label for reports and table headers.
``type`` (str, optional, default ``"level"``)
    How to count agents:

    * ``"exited"``  — cumulative count of agents who have fully evacuated
      (``level`` / area settings are ignored).
    * ``"level"``   — count active agents filtered by level and/or area.

``level`` (str, optional)
    JuPedSim level ID (e.g. ``"0"``, ``"-1"``).
    If omitted, agents on all levels are counted.
``area_patterns`` (list[str], optional)
    Include only agents whose position falls inside *at least one* polygon
    whose name starts with any of these prefix strings.
``exclude_area_patterns`` (list[str], optional)
    Exclude agents whose position falls inside *any* polygon whose name
    starts with any of these prefix strings.

Area pattern matching searches these geometry sources:
* ``transfer_manager.escalator_zones``  (multi-level simulations)
* Each level's ``geometry_manager.walkable_areas_with_obstacles``
* Each level's ``geometry_manager.escalator_corridors``
* Each level's ``geometry_manager.platform_areas``

Default zones (used when ``monitoring`` is absent from the config)
------------------------------------------------------------------
The defaults replicate the four standard zones for a two-level station
(concourse + underground platforms, e.g. Monument Station):

1. ``left_station``     — cumulative exits  (type: ``exited``)
2. ``concourse``        — level ``"0"`` agents
3. ``escalator_bottom`` — level ``"-1"`` agents in ``L-1_esc*`` polygons
4. ``platform``         — level ``"-1"`` agents *not* in ``L-1_esc*`` polygons
"""

import csv
import json
from pathlib import Path
from typing import Any

from shapely.geometry import Point

from evacusim.utils.logger import get_logger

logger = get_logger(__name__)

# Default zone specifications used when no monitoring config is provided.
# Replicate the four standard zones for a two-level station layout.
DEFAULT_ZONE_SPECS: list[dict] = [
    {
        "name": "left_station",
        "description": "Agents who have left the station",
        "type": "exited",
    },
    {
        "name": "concourse",
        "description": "Agents on the concourse (level 0)",
        "level": "0",
    },
    {
        "name": "escalator_bottom",
        "description": "Agents at the bottom of escalators (level -1, escalator zones)",
        "level": "-1",
        "area_patterns": ["L-1_esc"],
    },
    {
        "name": "platform",
        "description": "Agents on a platform (level -1)",
        "level": "-1",
        "exclude_area_patterns": ["L-1_esc"],
    },
]


class PopulationMonitor:
    """
    Records station zone occupancy counts at regular intervals.

    Zone definitions are supplied as a list of spec dicts (typically loaded
    from the YAML config ``monitoring.zones``).  See module docstring for the
    spec format.  If no specs are provided the default ones for a two-level
    station (concourse + underground platforms) are used.

    Usage::

        monitor = PopulationMonitor(
            jps_sim,
            zone_specs=config["monitoring"]["zones"],
            interval_seconds=60.0,
        )

        # Inside the simulation loop:
        monitor.record_snapshot(current_sim_time, exited_agents)

        # After the loop:
        monitor.display_summary()
        monitor.save(output_dir)
    """

    def __init__(
        self,
        jps_sim: Any,
        zone_specs: list[dict] | None = None,
        interval_seconds: float = 60.0,
    ) -> None:
        """
        Initialise the monitor.

        Args:
            jps_sim: The running pedestrian simulation (either
                ``ConcordiaJuPedSimulation`` or ``MultiLevelJuPedSimulation``).
            zone_specs: List of zone specification dicts (see module docstring).
                If ``None``, ``DEFAULT_ZONE_SPECS`` are used.
            interval_seconds: Interval between snapshots in *simulation* seconds.
                Defaults to 60 s (one minute of sim time).
        """
        self.jps_sim = jps_sim
        self.interval = interval_seconds
        self._next_record_time: float = 0.0

        # Resolve zone specs into internal representation with polygon lists.
        self._zones: list[dict] = self._resolve_zone_specs(
            zone_specs if zone_specs is not None else DEFAULT_ZONE_SPECS
        )

        # Time series storage: sim_times + per-zone count lists keyed by name.
        self.sim_times: list[float] = []
        self.counts: dict[str, list[int]] = {z["name"]: [] for z in self._zones}

        zone_summary = ", ".join(z["name"] for z in self._zones)
        logger.info(
            f"PopulationMonitor initialised — zones=[{zone_summary}], "
            f"interval={interval_seconds}s"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record_snapshot(self, sim_time: float, exited_agents: set[str]) -> None:
        """
        Record a snapshot if the simulation has reached the next interval.

        Safe to call every step — it is a no-op before the next interval.

        Args:
            sim_time: Current simulation time in seconds.
            exited_agents: Set of agent IDs that have fully evacuated.
        """
        if sim_time < self._next_record_time:
            return

        positions = self.jps_sim.get_all_agent_positions()
        agent_levels: dict[str, str] = getattr(self.jps_sim, "agent_levels", {})
        is_multi_level = hasattr(self.jps_sim, "simulations")

        self.sim_times.append(sim_time)

        for zone in self._zones:
            if zone["type"] == "exited":
                self.counts[zone["name"]].append(len(exited_agents))
                continue

            count = 0
            for agent_id, pos in positions.items():
                level = agent_levels.get(agent_id, "0") if is_multi_level else "0"

                # Level filter
                if zone["level"] is not None and level != zone["level"]:
                    continue

                # Polygon filters — only create Point when actually needed
                point: Point | None = None

                if zone["include_polys"]:
                    point = Point(pos)
                    if not any(p.contains(point) for p in zone["include_polys"]):
                        continue

                if zone["exclude_polys"]:
                    if point is None:
                        point = Point(pos)
                    if any(p.contains(point) for p in zone["exclude_polys"]):
                        continue

                count += 1

            self.counts[zone["name"]].append(count)

        count_summary = "  ".join(f"{z['name']}={self.counts[z['name']][-1]}" for z in self._zones)
        logger.debug(f"[PopulationMonitor] t={sim_time:.0f}s  {count_summary}")

        self._next_record_time += self.interval

    def display_summary(self) -> None:
        """
        Print a formatted time-series table to the console using *rich*.

        Falls back to plain-text if *rich* is unavailable.
        """
        if not self.sim_times:
            logger.info("[PopulationMonitor] No snapshots recorded.")
            return

        try:
            from rich.console import Console
            from rich.table import Table

            table = Table(title="Population Monitor — Zone Occupancy Over Time")
            table.add_column("Sim Time", justify="right", style="cyan")
            for zone in self._zones:
                table.add_column(zone["description"], justify="right")

            for i, t in enumerate(self.sim_times):
                mins, secs = divmod(int(t), 60)
                row = [f"{mins}m {secs:02d}s"] + [
                    str(self.counts[z["name"]][i]) for z in self._zones
                ]
                table.add_row(*row)

            Console().print(table)

        except ImportError:
            # Plain-text fallback
            col_w = 14
            headers = ["Time".rjust(8)] + [z["name"][:col_w].rjust(col_w) for z in self._zones]
            header_line = " | ".join(headers)
            sep = "-" * len(header_line)
            lines = [
                "\n=== Population Monitor — Zone Occupancy Over Time ===",
                header_line,
                sep,
            ]
            for i, t in enumerate(self.sim_times):
                mins, secs = divmod(int(t), 60)
                row = [f"{mins}m{secs:02d}s".rjust(8)] + [
                    str(self.counts[z["name"]][i]).rjust(col_w) for z in self._zones
                ]
                lines.append(" | ".join(row))
            print("\n".join(lines))

    def save(self, output_dir: Path) -> None:
        """
        Persist the time series to ``population_timeseries.json`` and
        ``population_timeseries.csv`` inside *output_dir*.

        Args:
            output_dir: Directory in which to write the output files.
        """
        if not self.sim_times:
            return

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # --- JSON ---
        data = self.to_dict()
        json_path = output_dir / "population_timeseries.json"
        with open(json_path, "w") as fh:
            json.dump(data, fh, indent=2)
        logger.info(f"Population time series saved to {json_path}")

        # --- CSV ---
        csv_path = output_dir / "population_timeseries.csv"
        zone_names = [z["name"] for z in self._zones]
        with open(csv_path, "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["sim_time_s", "sim_time_min"] + zone_names)
            for i, t in enumerate(self.sim_times):
                writer.writerow(
                    [f"{t:.1f}", f"{t / 60:.3f}"] + [self.counts[name][i] for name in zone_names]
                )
        logger.info(f"Population time series CSV saved to {csv_path}")

        # --- PNG graph ---
        self._save_graph(output_dir)

    def _save_graph(self, output_dir: Path) -> None:
        """
        Save a line graph of zone occupancy over time as
        ``population_timeseries.png``.

        Silently skipped if matplotlib is not installed.
        """
        try:
            import matplotlib

            matplotlib.use("Agg")  # non-interactive, safe for background calls
            import matplotlib.pyplot as plt
        except ImportError:
            logger.debug("matplotlib not available — skipping population graph")
            return

        times_min = [t / 60.0 for t in self.sim_times]

        fig, ax = plt.subplots(figsize=(10, 5))

        for zone in self._zones:
            ax.plot(
                times_min,
                self.counts[zone["name"]],
                marker="o",
                markersize=3,
                linewidth=1.5,
                label=zone["description"],
            )

        ax.set_xlabel("Simulation time (minutes)")
        ax.set_ylabel("Number of people")
        ax.set_title("Population Monitor — Zone Occupancy Over Time")
        ax.legend(loc="best", fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)

        ax.xaxis.set_major_locator(plt.MultipleLocator(1))

        fig.tight_layout()
        graph_path = output_dir / "population_timeseries.png"
        fig.savefig(graph_path, dpi=150)
        plt.close(fig)
        logger.info(f"Population time series graph saved to {graph_path}")

    def to_dict(self) -> dict[str, Any]:
        """Return the time series as a serialisable dictionary."""
        return {
            "description": {z["name"]: z["description"] for z in self._zones},
            "interval_seconds": self.interval,
            "snapshots": [
                {
                    "sim_time_s": self.sim_times[i],
                    "sim_time_min": round(self.sim_times[i] / 60, 3),
                    **{z["name"]: self.counts[z["name"]][i] for z in self._zones},
                }
                for i in range(len(self.sim_times))
            ],
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_zone_specs(self, specs: list[dict]) -> list[dict]:
        """
        Resolve area-pattern strings in each spec dict to actual Shapely
        polygon lists, returning a list of enriched zone dicts.
        """
        resolved = []
        for spec in specs:
            zone: dict = {
                "name": spec["name"],
                "description": spec.get("description", spec["name"]),
                "type": spec.get("type", "level"),
                "level": spec.get("level"),  # None → no level filter
                "include_polys": [],
                "exclude_polys": [],
            }
            if zone["type"] == "level":
                for pattern in spec.get("area_patterns", []):
                    zone["include_polys"].extend(self._collect_polygons_by_prefix(pattern))
                for pattern in spec.get("exclude_area_patterns", []):
                    zone["exclude_polys"].extend(self._collect_polygons_by_prefix(pattern))
            resolved.append(zone)
            logger.debug(
                f"Zone '{zone['name']}': type={zone['type']}, level={zone['level']}, "
                f"include_polys={len(zone['include_polys'])}, "
                f"exclude_polys={len(zone['exclude_polys'])}"
            )
        return resolved

    def _collect_polygons_by_prefix(self, prefix: str) -> list:
        """
        Return all named Shapely polygons from the simulation whose name
        starts with *prefix*.

        Searches, where available:
        * ``transfer_manager.escalator_zones``  (multi-level)
        * Each level's ``geometry_manager.walkable_areas_with_obstacles``
        * Each level's ``geometry_manager.escalator_corridors``
        * Each level's ``geometry_manager.platform_areas``
        """
        polys: list = []

        def _scan(mapping: dict) -> None:
            for name, poly in mapping.items():
                if name.startswith(prefix):
                    polys.append(poly)

        if hasattr(self.jps_sim, "simulations"):
            # Multi-level: transfer manager + all level geometry managers
            transfer_manager = getattr(self.jps_sim, "transfer_manager", None)
            if transfer_manager is not None:
                _scan(transfer_manager.escalator_zones)
            for level_sim in self.jps_sim.simulations.values():
                gm = level_sim.geometry_manager
                _scan(gm.walkable_areas_with_obstacles)
                _scan(getattr(gm, "escalator_corridors", {}))
                _scan(getattr(gm, "platform_areas", {}))
        else:
            # Single-level: geometry manager on jps_sim directly
            gm = getattr(self.jps_sim, "geometry_manager", None)
            if gm is not None:
                _scan(gm.walkable_areas_with_obstacles)
                _scan(getattr(gm, "escalator_corridors", {}))
                _scan(getattr(gm, "platform_areas", {}))

        return polys
