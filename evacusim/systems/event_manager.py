"""
Event management for Station Concordia simulations.

Handles:
- Event triggering based on test scenarios
- Exit blocking (physical obstacles + agent discovery)
- Event broadcasting to agents
- Event history tracking
"""

from typing import Any

from evacusim.utils.logger import get_logger
from evacusim.jps.simulation_interface import PedestrianSimulation

logger = get_logger(__name__)


class EventManager:
    """
    Manages simulation events like exit blocking and announcements.

    Responsible for:
    - Triggering scheduled events (exit blocking at specific times)
    - Broadcasting announcements to all agents
    - Tracking event history
    - Managing blocked exits state
    """

    def __init__(self, station_layout: dict[str, Any], jps_sim: PedestrianSimulation):
        """
        Initialize the event manager.

        Args:
            station_layout: Station geometry and exit information
            jps_sim: Pedestrian simulation instance (implements PedestrianSimulation)
        """
        self.station_layout = station_layout
        self.jps_sim = jps_sim

        # Event state
        self.event_history: list[dict[str, Any]] = []
        self.blocked_exits: set[str] = set()
        self.scheduled_events: list[dict[str, Any]] = []  # timed events from config

        # Train state — populated by train_arrival events.
        # active_train_exits: exits currently open for boarding.
        # _train_departure_times: exit_name -> sim time when the train departs (doors close).
        self.active_train_exits: set[str] = set()
        self._train_departure_times: dict[str, float] = {}

    def check_and_trigger_events(
        self,
        current_sim_time: float,
        agents: dict[str, Any] | None = None,
        message_system: Any | None = None,
        exited_agents: set[str] | None = None,
        zone_id_for_agent_fn: Any | None = None,
    ) -> bool:
        """
        Check for and trigger simulation events.

        Supports two event types in the config:

        Standard event (broadcast to Concordia memory)::

            - time: 15.0
              message: "The fire alarm is sounding."

        PA announcement with optional per-zone messages (via MessageSystem)::

            - time: 20.0
              pa_announcement: true
              sender_label: "Station PA"      # optional, default "PA system"
              message: "Please evacuate."     # default text for unmatched zones
              repeat_interval: 20.0           # optional: re-broadcast every N seconds
              zone_messages:                  # optional per-zone overrides
                platform_abc: "Board the train now."
                concourse: "Leave via the nearest exit."

        Train arrival event (activates train boarding exits)::

            - time: 10.0
              type: train_arrival
              platforms: [1, 2, 3, 4]         # platform numbers with waiting trains
              message: "A train is waiting at all platforms."

        Args:
            current_sim_time: Current simulation time in seconds.
            agents: Dictionary of agent_id -> agent entity (required for standard broadcasts).
            message_system: MessageSystem instance (required for PA announcements).
            exited_agents: Set of agent IDs that have already evacuated (for PA filtering).
            zone_id_for_agent_fn: Callable(agent_id) -> zone_id | None (for PA zone routing).

        Returns:
            True if one or more new events were fired this step, False otherwise.
        """
        fired = False

        # Check whether any previously activated train exits have now departed.
        # This runs every step (not only when a scheduled event is due).
        if self._check_train_departures(current_sim_time, agents, message_system,
                                        exited_agents, zone_id_for_agent_fn):
            fired = True

        for event in self.scheduled_events:
            # Determine whether this event is due to fire.
            repeat_interval = event.get("repeat_interval")
            if repeat_interval:
                # Repeating event: fire on first due time, then every repeat_interval seconds.
                if current_sim_time < event["time"]:
                    continue
                last_fired = event.get("_last_fired")
                if last_fired is not None and current_sim_time < last_fired + repeat_interval:
                    continue
            else:
                # One-shot event.
                if event.get("_fired") or current_sim_time < event["time"]:
                    continue

            event_type = event.get("type", "")
            is_pa = event.get("pa_announcement") or event_type == "pa_announcement"
            is_train_arrival = event_type == "train_arrival"

            if is_train_arrival:
                self._fire_train_arrival(event, current_sim_time, agents, message_system,
                                         exited_agents, zone_id_for_agent_fn)
            elif is_pa and message_system is not None:
                self._fire_pa_announcement(event, current_sim_time, agents, message_system,
                                            exited_agents, zone_id_for_agent_fn)
            elif agents and event.get("message"):
                self.broadcast_event(event["message"], current_sim_time, agents)

            if repeat_interval:
                event["_last_fired"] = current_sim_time
            else:
                event["_fired"] = True
            fired = True

        return fired

    def _fire_pa_announcement(
        self,
        event: dict[str, Any],
        current_sim_time: float,
        agents: dict[str, Any] | None,
        message_system: Any,
        exited_agents: set[str] | None,
        zone_id_for_agent_fn: Any | None,
    ) -> None:
        """Deliver a PA announcement event via MessageSystem."""
        sender_label = event.get("sender_label", "PA system")
        default_msg = event.get("message", "")
        zone_messages = event.get("zone_messages", None)
        all_ids = list(agents.keys()) if agents else []
        message_system.deliver_pa(
            sender_label=sender_label,
            message_text=default_msg,
            current_sim_time=current_sim_time,
            all_agent_ids=all_ids,
            exited_agents=exited_agents or set(),
            messages_by_zone=zone_messages,
            zone_id_for_agent_fn=zone_id_for_agent_fn,
        )
        # Record in event_history so the ongoing-situation line reflects the PA.
        if default_msg or zone_messages:
            display_msg = default_msg or next(iter(zone_messages.values()), "PA announcement")
            self.event_history.append(
                {"time": current_sim_time, "message": f"[{sender_label}] {display_msg}"}
            )
            if agents:
                for agent in agents.values():
                    agent.observe(f"[{sender_label}] {display_msg}")

    def _fire_train_arrival(
        self,
        event: dict[str, Any],
        current_sim_time: float,
        agents: dict[str, Any] | None,
        message_system: Any | None,
        exited_agents: set[str] | None,
        zone_id_for_agent_fn: Any | None,
    ) -> None:
        """Handle a train_arrival event: activate train exits and notify agents."""
        platforms = event.get("platforms", [])
        message = event.get("message", "")
        dwell_seconds = float(event.get("dwell_seconds", 30.0))
        departure_time = current_sim_time + dwell_seconds

        # Activate the train exits for the specified platforms.
        for platform_id in platforms:
            exit_name = f"train_platform_{platform_id}"
            self.active_train_exits.add(exit_name)
            self._train_departure_times[exit_name] = departure_time
            logger.info(
                f"Train arrived at platform {platform_id} — exit '{exit_name}' activated; "
                f"departs at t={departure_time:.0f}s (dwell={dwell_seconds:.0f}s)."
            )

        # Broadcast the arrival message to all agents.
        if message:
            if message_system is not None and agents:
                sender_label = event.get("sender_label", "PA system")
                all_ids = list(agents.keys())
                message_system.deliver_pa(
                    sender_label=sender_label,
                    message_text=message,
                    current_sim_time=current_sim_time,
                    all_agent_ids=all_ids,
                    exited_agents=exited_agents or set(),
                    messages_by_zone=None,
                    zone_id_for_agent_fn=zone_id_for_agent_fn,
                )
            elif agents:
                self.broadcast_event(message, current_sim_time, agents)

            self.event_history.append({"time": current_sim_time, "message": message})

    def _check_train_departures(
        self,
        current_sim_time: float,
        agents: dict[str, Any] | None,
        message_system: Any | None,
        exited_agents: set[str] | None,
        zone_id_for_agent_fn: Any | None,
    ) -> bool:
        """
        Remove any train exits whose dwell period has ended.

        Called on every simulation step so that departures are detected
        promptly regardless of whether a scheduled event is also due.

        Returns:
            True if at least one train departed this step.
        """
        now_departed = [
            name for name, t in self._train_departure_times.items()
            if current_sim_time >= t
        ]
        if not now_departed:
            return False

        for exit_name in now_departed:
            self.active_train_exits.discard(exit_name)
            del self._train_departure_times[exit_name]
            platform_id = exit_name.rsplit("_", 1)[-1]
            logger.info(
                f"Train departed from platform {platform_id} at t={current_sim_time:.1f}s "
                f"— exit '{exit_name}' deactivated."
            )

        # Broadcast a departure announcement so agents know the train has gone.
        departure_msg = (
            "The train doors have closed and the train has departed. "
            "If you are still on the platform, please use the escalators to leave."
        )
        if message_system is not None and agents:
            message_system.deliver_pa(
                sender_label="PA system",
                message_text=departure_msg,
                current_sim_time=current_sim_time,
                all_agent_ids=list(agents.keys()),
                exited_agents=exited_agents or set(),
                messages_by_zone=None,
                zone_id_for_agent_fn=zone_id_for_agent_fn,
            )
        elif agents:
            self.broadcast_event(departure_msg, current_sim_time, agents)

        self.event_history.append({"time": current_sim_time, "message": departure_msg})
        return True

    def block_exit(self, exit_name: str) -> None:
        """
        Block an exit by placing a physical obstacle in JuPedSim.

        Agents will discover the blockage when they get close (visual range ~20m)
        or observe others turning back from it.

        Args:
            exit_name: Name of the exit to block
        """
        if exit_name not in self.station_layout["exits"]:
            logger.warning(f"Cannot block unknown exit: {exit_name}")
            return

        exit_pos = self.station_layout["exits"][exit_name]

        # Add to blocked exits set (for observations)
        self.blocked_exits.add(exit_name)

        # Place physical obstacle in JuPedSim
        # Use a radius that blocks the entrance (typically 2-3m wide, so 3-4m radius covers it)
        # This makes the exit unreachable in pathfinding - agents cannot get close enough
        # to evacuate through it, and will naturally reroute when they observe the blockage
        try:
            # Obstacle radius sized for typical entrance width (2-3m)
            obstacle_radius = 4.0
            self.jps_sim.add_obstacle(exit_pos, radius=obstacle_radius)
            logger.info(
                f"🚧 Exit {exit_name} physically blocked at {exit_pos} "
                f"(obstacle radius: {obstacle_radius}m)"
            )
        except Exception as e:
            logger.warning(f"Failed to add physical obstacle at {exit_name}: {e}")

        # NO announcement - agents discover naturally through observation
        # NOTE: We do NOT immediately reroute agents heading to this exit.
        # Agents should be allowed to travel to the blocked exit and discover it naturally,
        # then they will observe the blockage and choose alternative routes in their next decision.
        logger.info(f"Exit {exit_name} blocked - agents will discover naturally")

    def broadcast_event(
        self, event_message: str, current_sim_time: float, agents: dict[str, Any]
    ) -> None:
        """
        Broadcast an event to all agents.

        Args:
            event_message: The event message to broadcast
            current_sim_time: Current simulation time
            agents: Dictionary of agent_id -> agent entity
        """
        logger.info(f"Broadcasting event: {event_message}")

        # Store event
        self.event_history.append(
            {
                "time": current_sim_time,
                "message": event_message,
            }
        )

        # Notify all agents
        for agent in agents.values():
            agent.observe(f"[ANNOUNCEMENT] {event_message}")
