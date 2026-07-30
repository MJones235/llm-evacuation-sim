# Function Reference: evacusim/jps/exit_tracker.py

- Source file: [evacusim/jps/exit_tracker.py](../../../evacusim/jps/exit_tracker.py)
- Top-level classes: 1
- Top-level functions: 0

## Module Summary

Exit Tracker Manages tracking of agents who have exited the simulation through evacuation exits.

## Classes and Methods

### Class ExitTracker

- Location: [evacusim/jps/exit_tracker.py#L17](../../../evacusim/jps/exit_tracker.py#L17)
- Summary: Tracks agents who have exited the simulation.
- Method count: 3

#### __init__

- Signature: def __init__(self, concordia_agents, exited_agents, agent_destinations, jps_sim, station_layout=None, exit_validation_radius=15.0)
- Location: [evacusim/jps/exit_tracker.py#L20](../../../evacusim/jps/exit_tracker.py#L20)
- Summary: Initialize exit tracker.

#### check_exited_agents

- Signature: def check_exited_agents(self, current_sim_time, current_step)
- Location: [evacusim/jps/exit_tracker.py#L50](../../../evacusim/jps/exit_tracker.py#L50)
- Summary: Check for agents who have reached exits and mark them as exited.
- Key calls: self.jps_sim.get_all_agent_positions, current_positions.items, len, list, self.concordia_agents.keys, logger.info, logger.warning, self.agent_destinations.get, self.last_known_positions.get, exit_name.startswith

#### _validate_exit_reached

- Signature: def _validate_exit_reached(self, agent_id, exit_name, last_position) -> bool
- Location: [evacusim/jps/exit_tracker.py#L121](../../../evacusim/jps/exit_tracker.py#L121)
- Summary: Validate that an agent actually reached their target exit.
- Key calls: self.station_layout.get, logger.debug

## Coverage Notes

- Nested helper functions documented in this file: 0
