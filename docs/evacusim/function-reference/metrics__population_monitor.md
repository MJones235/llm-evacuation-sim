# Function Reference: evacusim/metrics/population_monitor.py

- Source file: [evacusim/metrics/population_monitor.py](../../../evacusim/metrics/population_monitor.py)
- Top-level classes: 1
- Top-level functions: 0

## Module Summary

Population monitor for Station Concordia simulations.

## Classes and Methods

### Class PopulationMonitor

- Location: [evacusim/metrics/population_monitor.py#L95](../../../evacusim/metrics/population_monitor.py#L95)
- Summary: Records station zone occupancy counts at regular intervals.
- Method count: 8

#### __init__

- Signature: def __init__(self, jps_sim, zone_specs=None, interval_seconds=60.0) -> None
- Location: [evacusim/metrics/population_monitor.py#L120](../../../evacusim/metrics/population_monitor.py#L120)
- Summary: Initialise the monitor.
- Key calls: self._resolve_zone_specs, join, logger.info

#### record_snapshot

- Signature: def record_snapshot(self, sim_time, exited_agents, force=False) -> None
- Location: [evacusim/metrics/population_monitor.py#L160](../../../evacusim/metrics/population_monitor.py#L160)
- Summary: Record a snapshot if the simulation has reached the next interval.
- Key calls: self.jps_sim.get_all_agent_positions, getattr, hasattr, self.sim_times.append, join, logger.debug, positions.items, append, len, agent_levels.get

#### display_summary

- Signature: def display_summary(self) -> None
- Location: [evacusim/metrics/population_monitor.py#L219](../../../evacusim/metrics/population_monitor.py#L219)
- Summary: Print a formatted time-series table to the console using *rich*.
- Key calls: logger.info, Table, table.add_column, enumerate, print, divmod, table.add_row, join, int, Console

#### save

- Signature: def save(self, output_dir) -> None
- Location: [evacusim/metrics/population_monitor.py#L266](../../../evacusim/metrics/population_monitor.py#L266)
- Summary: Persist the time series to ``population_timeseries.
- Key calls: Path, output_dir.mkdir, self.to_dict, logger.info, self._save_graph, open, json.dump, csv.writer, writer.writerow, enumerate

#### _save_graph

- Signature: def _save_graph(self, output_dir) -> None
- Location: [evacusim/metrics/population_monitor.py#L302](../../../evacusim/metrics/population_monitor.py#L302)
- Summary: Save a line graph of zone occupancy over time as ``population_timeseries.
- Key calls: plt.subplots, ax.set_xlabel, ax.set_ylabel, ax.set_title, ax.legend, ax.grid, ax.set_xlim, ax.set_ylim, ax.xaxis.set_major_locator, fig.tight_layout

#### to_dict

- Signature: def to_dict(self) -> dict[str, Any]
- Location: [evacusim/metrics/population_monitor.py#L348](../../../evacusim/metrics/population_monitor.py#L348)
- Summary: Return the time series as a serialisable dictionary.
- Key calls: round, range, len

#### _resolve_zone_specs

- Signature: def _resolve_zone_specs(self, specs) -> list[dict]
- Location: [evacusim/metrics/population_monitor.py#L367](../../../evacusim/metrics/population_monitor.py#L367)
- Summary: Resolve area-pattern strings in each spec dict to actual Shapely polygon lists, returning a list of enriched zone dicts.
- Key calls: resolved.append, logger.debug, spec.get, extend, self._collect_polygons_by_prefix, len

#### _collect_polygons_by_prefix

- Signature: def _collect_polygons_by_prefix(self, prefix) -> list
- Location: [evacusim/metrics/population_monitor.py#L395](../../../evacusim/metrics/population_monitor.py#L395)
- Summary: Return all named Shapely polygons from the simulation whose name starts with *prefix*.
- Key calls: hasattr, mapping.items, getattr, self.jps_sim.simulations.values, name.startswith, _scan, polys.append
- Nested function count: 1

##### Nested helpers inside _collect_polygons_by_prefix

###### _collect_polygons_by_prefix._scan

- Signature: def _scan(mapping) -> None
- Location: [evacusim/metrics/population_monitor.py#L408](../../../evacusim/metrics/population_monitor.py#L408)
- Summary: No nested-function docstring summary available.
- Key calls: mapping.items, name.startswith, polys.append

## Coverage Notes

- Nested helper functions documented in this file: 1
