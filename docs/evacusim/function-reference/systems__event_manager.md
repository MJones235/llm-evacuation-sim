# Function Reference: evacusim/systems/event_manager.py

- Source file: [evacusim/systems/event_manager.py](../../../evacusim/systems/event_manager.py)
- Top-level classes: 1
- Top-level functions: 0

## Module Summary

Event management for Station Concordia simulations.

## Classes and Methods

### Class EventManager

- Location: [evacusim/systems/event_manager.py#L19](../../../evacusim/systems/event_manager.py#L19)
- Summary: Manages simulation events like exit blocking and announcements.
- Method count: 8

#### __init__

- Signature: def __init__(self, station_layout, jps_sim)
- Location: [evacusim/systems/event_manager.py#L30](../../../evacusim/systems/event_manager.py#L30)
- Summary: Initialize the event manager.
- Key calls: set

#### check_and_trigger_events

- Signature: def check_and_trigger_events(self, current_sim_time, agents=None, message_system=None, exited_agents=None, zone_id_for_agent_fn=None) -> bool
- Location: [evacusim/systems/event_manager.py#L56](../../../evacusim/systems/event_manager.py#L56)
- Summary: Check for and trigger simulation events.
- Key calls: self._check_train_departures, event.get, self._fire_block_exit, self._fire_train_arrival, self._fire_pa_announcement, self.broadcast_event

#### _fire_block_exit

- Signature: def _fire_block_exit(self, event, current_sim_time) -> None
- Location: [evacusim/systems/event_manager.py#L158](../../../evacusim/systems/event_manager.py#L158)
- Summary: Handle a ``block_exit`` event from config.
- Key calls: event.get, isinstance, logger.info, self.block_exit

#### _fire_pa_announcement

- Signature: def _fire_pa_announcement(self, event, current_sim_time, agents, message_system, exited_agents, zone_id_for_agent_fn) -> None
- Location: [evacusim/systems/event_manager.py#L182](../../../evacusim/systems/event_manager.py#L182)
- Summary: Deliver a PA announcement event via MessageSystem.
- Key calls: event.get, message_system.deliver_pa, list, self.event_history.append, agents.keys, next, agents.values, set, iter, agent.observe

#### _fire_train_arrival

- Signature: def _fire_train_arrival(self, event, current_sim_time, agents, message_system, exited_agents, zone_id_for_agent_fn) -> None
- Location: [evacusim/systems/event_manager.py#L215](../../../evacusim/systems/event_manager.py#L215)
- Summary: Handle a train_arrival event: activate train exits and notify agents.
- Key calls: event.get, float, self.active_train_exits.add, bool, logger.info, self.event_history.append, list, message_system.deliver_pa, agents.keys, self.broadcast_event

#### _check_train_departures

- Signature: def _check_train_departures(self, current_sim_time, agents, message_system, exited_agents, zone_id_for_agent_fn) -> bool
- Location: [evacusim/systems/event_manager.py#L268](../../../evacusim/systems/event_manager.py#L268)
- Summary: Remove any train exits whose dwell period has ended.
- Key calls: self.active_train_exits.discard, logger.info, self._train_departure_announce.pop, self._train_departure_messages.pop, self._train_departure_sender_labels.pop, self._train_departure_times.items, exit_name.rsplit, self.event_history.append, message_system.deliver_pa, self.broadcast_event

#### block_exit

- Signature: def block_exit(self, exit_name) -> None
- Location: [evacusim/systems/event_manager.py#L326](../../../evacusim/systems/event_manager.py#L326)
- Summary: Mark an exit as blocked.
- Key calls: self.blocked_exits.add, logger.info, exit_name.startswith, hasattr, self.jps_sim.add_geometry_obstacle_for_exit, logger.warning

#### broadcast_event

- Signature: def broadcast_event(self, event_message, current_sim_time, agents) -> None
- Location: [evacusim/systems/event_manager.py#L362](../../../evacusim/systems/event_manager.py#L362)
- Summary: Broadcast an event to all agents.
- Key calls: logger.info, self.event_history.append, agents.values, agent.observe

## Coverage Notes

- Nested helper functions documented in this file: 0
