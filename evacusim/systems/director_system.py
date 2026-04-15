"""
Director System

A configurable rule-based system for non-LLM director agents.

Director agents exist as physics agents in JuPedSim and periodically broadcast
directive messages to nearby agents via the MessageSystem.  They are driven
entirely by configuration — no LLM calls are made.

--------------------------------------------------------------------------
Single-phase (flat) config — backward-compatible
--------------------------------------------------------------------------

    systems:
      my_staff:
        enabled: true
        role_label: "staff member"       # label shown to other agents
        activate_on_event: false         # if true, waits for the first event
        movement: hold                   # "hold" (static) or "zone_patrol"
        patrol_zones:                    # required when movement = zone_patrol
          - zone: concourse
          - zone: platform_abc
            level_id: "-1"              # level for multi-level simulations
        patrol_dwell_time: 20.0          # seconds at each zone
        directive_radius: 12.0           # metres — broadcast reach
        directive_interval: 10.0         # seconds between broadcasts
        message: "Please evacuate now."  # directive text
        messages_by_zone:                # optional per-recipient-zone overrides
          platform_abc: "Use the stairs."
          concourse: "Leave via the exits."
        walking_speed: 1.2               # m/s
        spawn_positions:
          - zone: concourse
          - zone: platform_abc
            level_id: "-1"
          - position: [12.5, 34.2]
            level_id: "0"

--------------------------------------------------------------------------
Multi-phase config
--------------------------------------------------------------------------
Use ``phases:`` to define sequential behaviour that changes mid-simulation.
Each phase has a trigger that fires the transition from the previous phase.

Trigger types
~~~~~~~~~~~~~
- ``immediate``      — enters the phase at simulation start (first-phase default)
- ``on_event``       — enters after the first simulation event fires
- ``on_reach_zone``  — enters when the agent reaches ``trigger_zone``
                       (optionally gated to ``trigger_level_id``)
- ``after_seconds``  — enters N seconds after the *previous* phase was activated

Phase-local keys override the system defaults for movement/messages/timing.
``walking_speed`` is system-wide only (set at the top level).

Example:

    systems:
      rci_concourse:
        role_label: "Revenue Control Inspector"
        walking_speed: 1.6
        spawn_positions:
          - zone: platform_3
            level_id: "-1"
        phases:
          - trigger: on_event
            movement: zone_patrol
            patrol_zones:
              - zone: platform_3
                level_id: "-1"
              - zone: concourse
                level_id: "0"
            patrol_dwell_time: 5.0
            directive_radius: 12.0
            directive_interval: 10.0
            message: "Please evacuate via the concourse!"
          - trigger: on_reach_zone
            trigger_zone: concourse
            trigger_level_id: "0"
            movement: hold
            directive_radius: 15.0
            directive_interval: 10.0
            message: "Please leave the station. Do not enter or wait."

Cross-level patrol is supported: include ``level_id`` on each ``patrol_zones``
entry and the director routes to the nearest escalator when the waypoint is on
a different level.  MultiLevelSimulation teleports the agent once they step into
the escalator zone, and the patrol resumes on the new level.

Director agents are registered in the shared ``agent_roles`` dict so that
nearby Concordia agents can see their role label in observations.
"""

from typing import Any

from evacusim.utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Phase normalisation helper (module-level so it can be unit-tested)
# ---------------------------------------------------------------------------

def _normalize_phases(system_config: dict) -> list[dict]:
    """
    Convert either a flat config or a ``phases`` list into a normalised list
    of phase dicts.

    Flat config: a single phase is synthesised; ``activate_on_event: true``
    maps to ``trigger: on_event``.
    """
    if "phases" in system_config:
        raw_phases: list[dict] = system_config["phases"]
    else:
        trigger = "on_event" if system_config.get("activate_on_event") else "immediate"
        raw_phases = [
            {
                "trigger": trigger,
                "movement": system_config.get("movement", "hold"),
                "patrol_zones": system_config.get("patrol_zones", []),
                "patrol_dwell_time": system_config.get("patrol_dwell_time", 20.0),
                "directive_radius": system_config.get("directive_radius", 10.0),
                "directive_interval": system_config.get("directive_interval", 10.0),
                "message": system_config.get("message", ""),
                "messages_by_zone": system_config.get("messages_by_zone", {}),
            }
        ]

    normalized: list[dict] = []
    for raw in raw_phases:
        tl = raw.get("trigger_level_id")
        phase: dict = {
            # Transition trigger
            "trigger": raw.get("trigger", "immediate"),
            "trigger_zone": raw.get("trigger_zone"),
            "trigger_level_id": str(tl) if tl is not None else None,
            "trigger_after_seconds": (
                float(raw["after_seconds"]) if "after_seconds" in raw else None
            ),
            # Movement
            "movement": raw.get("movement", "hold"),
            # For hold mode: optional zone to hold at (centroid); defaults to
            # current agent position at phase activation time.
            "hold_zone": raw.get("hold_zone"),
            "hold_level_id": str(raw["hold_level_id"]) if "hold_level_id" in raw else None,
            # Patrol — raw entries; waypoints resolved to {pos, level_id} in setup()
            "patrol_zones": raw.get("patrol_zones", []),
            "patrol_waypoints": [],
            "patrol_dwell_time": float(raw.get("patrol_dwell_time", 20.0)),
            # Messaging
            "directive_radius": float(raw.get("directive_radius", 10.0)),
            "directive_interval": float(raw.get("directive_interval", 10.0)),
            "message": raw.get("message", ""),
            "messages_by_zone": raw.get("messages_by_zone", {}),
        }
        normalized.append(phase)
    return normalized


# ---------------------------------------------------------------------------
# DirectorSystem
# ---------------------------------------------------------------------------

class DirectorSystem:
    """
    Rule-based director agent system driven entirely by configuration.

    Instances are created and owned by HybridSimulationRunner.
    """

    # How often (seconds) to re-issue the hold target to resist crowd pressure.
    _HOLD_REFRESH_INTERVAL: float = 30.0

    def __init__(
        self,
        system_name: str,
        system_config: dict[str, Any],
    ) -> None:
        self.system_name = system_name
        self.role_label: str = system_config.get("role_label", system_name.replace("_", " "))
        # walking_speed is system-wide (agents are spawned once and keep this speed)
        self.walking_speed: float = float(system_config.get("walking_speed", 1.2))
        self._spawn_configs: list[dict[str, Any]] = system_config.get("spawn_positions", [])

        # Normalised phase list — populated in __init__, waypoints resolved in setup()
        self._phases: list[dict] = _normalize_phases(system_config)

        # Populated during setup()
        self.agent_ids: list[str] = []
        self._spawn_positions: list[tuple[float, float]] = []
        # Kept for hold_zone centroid resolution after setup
        self._zones_polygons_ref: dict[str, Any] = {}

        # Per-agent phase state
        # -1 = not yet entered phase 0 (waiting for trigger)
        self._agent_phase: dict[str, int] = {}
        self._agent_phase_activated_at: dict[str, float] = {}

        # Per-agent patrol state
        self._patrol_index: dict[str, int] = {}
        self._patrol_arrived_at: dict[str, float] = {}

        # Per-agent hold state
        # None = not yet set; populated on first _step_hold_agent call in each phase.
        self._hold_targets: dict[str, tuple[float, float] | None] = {}
        self._last_hold_refresh: dict[str, float] = {}

        # Per-agent cross-level routing state.
        # Stores the target level currently being navigated to via escalator.
        # None means no cross-level route is in progress.
        # Set when _route_to_escalator fires; cleared when the agent arrives on
        # the target level so that the dwell timer restarts cleanly.
        self._cross_level_routing: dict[str, str | None] = {}

        # Suppress on_reach_zone triggers for this many seconds after a level
        # transfer completes.  Without this grace period, the trigger fires the
        # instant the agent is teleported to the new level — while they're still
        # standing in the escalator arrival zone — causing hold mode to lock
        # their position at the exit point rather than the intended zone centre.
        # 8 seconds: long enough to walk out of the arrival zone (~3–4 m at
        # 1.6 m/s) but short enough that a 10s patrol dwell at the destination
        # zone will still trigger the on_reach_zone check.
        self._TRANSFER_GRACE_PERIOD: float = 8.0
        # agent_id -> sim_time after which on_reach_zone triggers are re-enabled
        self._phase_transition_grace_until: dict[str, float] = {}

        # Per-agent directive throttle
        self._last_directive_time: dict[str, float] = {}

        # System-level event flag; set by notify_event_fired()
        self._event_fired: bool = False

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def setup(
        self,
        jps_sim: Any,
        station_layout: dict[str, Any],
        agent_roles: dict[str, str],
    ) -> list[str]:
        """
        Spawn director agents in JuPedSim and register their roles.

        Must be called before the simulation loop starts.

        Returns:
            List of director agent IDs.
        """
        zones_polygons: dict[str, Any] = station_layout.get("zones_polygons", {})
        self._zones_polygons_ref = zones_polygons

        # Resolve patrol waypoints for every phase up front
        for phase in self._phases:
            phase["patrol_waypoints"] = self._resolve_patrol_waypoints(
                phase["patrol_zones"], zones_polygons
            )
            if phase["movement"] == "zone_patrol" and not phase["patrol_waypoints"]:
                logger.warning(
                    f"[{self.system_name}] A phase has movement=zone_patrol but no "
                    "patrol_zones could be resolved — falling back to 'hold'."
                )
                phase["movement"] = "hold"

        first_phase = self._phases[0] if self._phases else None

        for idx, entry in enumerate(self._spawn_configs):
            agent_id = f"{self.system_name}_{idx}"
            level_id: str = str(entry.get("level_id", "0"))
            position = self._resolve_position(entry, zones_polygons, agent_id)
            if position is None:
                logger.warning(
                    f"[{self.system_name}] Could not resolve spawn position for entry {idx} "
                    f"({entry}); skipping this director agent."
                )
                continue

            try:
                if hasattr(jps_sim, "simulations"):
                    jps_sim.add_agent(agent_id, position, self.walking_speed, level_id)
                else:
                    jps_sim.add_agent(agent_id, position, self.walking_speed)

                self.agent_ids.append(agent_id)
                self._spawn_positions.append(position)

                # Determine whether to enter phase 0 immediately
                if first_phase and first_phase["trigger"] == "immediate":
                    self._agent_phase[agent_id] = 0
                    self._agent_phase_activated_at[agent_id] = 0.0
                    self._init_movement(agent_id, 0, jps_sim, position, level_id)
                else:
                    # Waiting for trigger — stand still
                    self._agent_phase[agent_id] = -1
                    self._agent_phase_activated_at[agent_id] = -1.0
                    try:
                        jps_sim.set_agent_target(agent_id, position)
                    except Exception:
                        pass

                # Initialise per-agent state
                dwell = first_phase["patrol_dwell_time"] if first_phase else 20.0
                interval = first_phase["directive_interval"] if first_phase else 10.0
                self._patrol_index[agent_id] = 0
                self._patrol_arrived_at[agent_id] = -dwell
                self._last_directive_time[agent_id] = -interval
                self._hold_targets[agent_id] = None
                self._last_hold_refresh[agent_id] = 0.0
                self._phase_transition_grace_until[agent_id] = 0.0

                agent_roles[agent_id] = self.role_label

                logger.info(
                    f"[{self.system_name}] Spawned director agent {agent_id} "
                    f"at {position} (level {level_id}, "
                    f"{len(self._phases)} phase(s))"
                )

            except Exception as exc:
                logger.error(
                    f"[{self.system_name}] Failed to add director agent {agent_id}: {exc}",
                    exc_info=True,
                )

        logger.info(
            f"[{self.system_name}] {len(self.agent_ids)} director agent(s) ready "
            f"(role: '{self.role_label}', {len(self._phases)} phase(s))"
        )

        # Register director agents as non-evacuating so the MultiLevelSimulation
        # escalator-zone handler does not override their patrol/hold targets by
        # routing them to evacuation exits.
        if hasattr(jps_sim, "non_evacuating_agents"):
            for aid in self.agent_ids:
                jps_sim.non_evacuating_agents.add(aid)

        return self.agent_ids

    # ------------------------------------------------------------------
    # Activation
    # ------------------------------------------------------------------

    def notify_event_fired(self) -> None:
        """Signal that a simulation event has fired."""
        self._event_fired = True
        logger.info(f"[{self.system_name}] Event fired — on_event phase transitions may activate.")

    # ------------------------------------------------------------------
    # Per-step update
    # ------------------------------------------------------------------

    def step(
        self,
        current_sim_time: float,
        jps_sim: Any,
        message_system: Any,
        state_queries: Any,
        exited_agents: set[str],
        zone_id_for_agent_fn: Any | None = None,
    ) -> None:
        """Per-step update for all director agents."""
        if not self.agent_ids:
            return

        for agent_id in self.agent_ids:
            # Try to advance forward through any number of immediately-satisfied
            # phase transitions (loop in case multiple triggers are met at once).
            while True:
                current_idx = self._agent_phase.get(agent_id, -1)
                next_idx = current_idx + 1
                if next_idx >= len(self._phases):
                    break
                next_phase = self._phases[next_idx]
                if self._is_trigger_met(
                    agent_id, next_phase, current_sim_time, jps_sim, zone_id_for_agent_fn
                ):
                    self._activate_phase(
                        agent_id, next_idx, current_sim_time, jps_sim, state_queries
                    )
                else:
                    break

            current_idx = self._agent_phase.get(agent_id, -1)
            if current_idx < 0:
                continue  # Still waiting for first trigger

            phase = self._phases[current_idx]
            position = state_queries.get_agent_position(agent_id)
            if position is None:
                continue

            # Periodic debug log so phase/level/position can be verified in logs
            if int(current_sim_time) % 10 == 0 and (current_sim_time % 1.0) < self.dt if hasattr(self, 'dt') else int(current_sim_time) % 10 == 0:
                current_level = (
                    jps_sim.agent_levels.get(agent_id)
                    if hasattr(jps_sim, "agent_levels") else "?"
                )
                grace_until = self._phase_transition_grace_until.get(agent_id, 0.0)
                grace_note = f" [grace until t={grace_until:.0f}s]" if grace_until > current_sim_time else ""
                logger.debug(
                    f"[{self.system_name}] {agent_id} t={current_sim_time:.1f}s "
                    f"phase={current_idx} movement={phase['movement']} "
                    f"level={current_level} pos={position}{grace_note}"
                )

            # Execute movement for this phase
            if phase["movement"] == "zone_patrol":
                self._step_patrol(agent_id, phase, current_sim_time, jps_sim, position)
            elif phase["movement"] == "hold":
                self._step_hold_agent(
                    agent_id, phase, position, current_sim_time, jps_sim
                )

            # Broadcast directive
            if not phase["message"] and not phase["messages_by_zone"]:
                continue
            last_t = self._last_directive_time.get(agent_id, -phase["directive_interval"])
            if current_sim_time - last_t < phase["directive_interval"]:
                continue
            message_system.deliver_directive(
                sender_id=agent_id,
                message_text=phase["message"],
                messages_by_zone=phase["messages_by_zone"],
                sender_position=position,
                current_sim_time=current_sim_time,
                state_queries=state_queries,
                exited_agents=exited_agents,
                radius=phase["directive_radius"],
                zone_id_for_agent_fn=zone_id_for_agent_fn,
            )
            self._last_directive_time[agent_id] = current_sim_time

    # ------------------------------------------------------------------
    # Phase management helpers
    # ------------------------------------------------------------------

    def _is_trigger_met(
        self,
        agent_id: str,
        phase: dict,
        current_sim_time: float,
        jps_sim: Any,
        zone_id_for_agent_fn: Any | None,
    ) -> bool:
        trigger = phase["trigger"]

        if trigger == "immediate":
            return True

        if trigger == "on_event":
            return self._event_fired

        if trigger == "on_reach_zone":
            # Suppressed during the grace period that follows a level transfer.
            if current_sim_time < self._phase_transition_grace_until.get(agent_id, 0.0):
                return False
            if zone_id_for_agent_fn is None:
                return False
            current_zone = zone_id_for_agent_fn(agent_id)
            if current_zone != phase["trigger_zone"]:
                return False
            # Optionally constrain to a specific level
            required_level = phase["trigger_level_id"]
            if required_level is not None and hasattr(jps_sim, "agent_levels"):
                if jps_sim.agent_levels.get(agent_id) != required_level:
                    return False
            return True

        if trigger == "after_seconds":
            activated_at = self._agent_phase_activated_at.get(agent_id, -1.0)
            if activated_at < 0:
                return False
            threshold = phase.get("trigger_after_seconds") or 0.0
            return (current_sim_time - activated_at) >= threshold

        return False

    def _activate_phase(
        self,
        agent_id: str,
        phase_idx: int,
        current_sim_time: float,
        jps_sim: Any,
        state_queries: Any,
    ) -> None:
        """Transition an agent to the given phase index."""
        phase = self._phases[phase_idx]
        self._agent_phase[agent_id] = phase_idx
        self._agent_phase_activated_at[agent_id] = current_sim_time

        # Reset per-agent patrol and hold state for the new phase
        self._patrol_index[agent_id] = 0
        self._patrol_arrived_at[agent_id] = -phase["patrol_dwell_time"]
        self._hold_targets[agent_id] = None  # re-acquire hold position on first tick
        self._last_hold_refresh[agent_id] = 0.0
        self._cross_level_routing[agent_id] = None
        # Don't reset the grace period here — if the phase transition itself was
        # triggered by on_reach_zone (i.e. after a cross-level transfer), the
        # grace window set by _step_patrol is still meaningful.
        # Fire directive immediately on entering new phase
        self._last_directive_time[agent_id] = current_sim_time - phase["directive_interval"]

        position = state_queries.get_agent_position(agent_id)
        current_level: str | None = (
            jps_sim.agent_levels.get(agent_id) if hasattr(jps_sim, "agent_levels") else None
        )
        self._init_movement(agent_id, phase_idx, jps_sim, position, current_level)

        logger.info(
            f"[{self.system_name}] {agent_id} → phase {phase_idx} "
            f"(trigger={phase['trigger']}, movement={phase['movement']})"
        )

    def _init_movement(
        self,
        agent_id: str,
        phase_idx: int,
        jps_sim: Any,
        position: tuple[float, float] | None,
        current_level: str | None,
    ) -> None:
        """Set an initial JuPedSim target for the agent entering a phase."""
        phase = self._phases[phase_idx]
        if phase["movement"] == "zone_patrol" and phase["patrol_waypoints"]:
            first_wp = phase["patrol_waypoints"][0]
            # Route directly only if on the same level; escalator routing handles
            # the cross-level case on the next tick.
            if current_level is None or first_wp["level_id"] == current_level:
                if position is not None:
                    try:
                        jps_sim.set_agent_target(agent_id, first_wp["pos"])
                    except Exception:
                        pass
        # Hold: target is set lazily on the first _step_hold_agent call

    # ------------------------------------------------------------------
    # Movement helpers
    # ------------------------------------------------------------------

    def _step_hold_agent(
        self,
        agent_id: str,
        phase: dict,
        position: tuple[float, float],
        current_sim_time: float,
        jps_sim: Any,
    ) -> None:
        """Maintain a stationary hold target for one agent."""
        hold_target = self._hold_targets.get(agent_id)
        if hold_target is None:
            # First call in this phase: resolve the hold position.
            if phase.get("hold_zone"):
                polygon = self._zones_polygons_ref.get(phase["hold_zone"])
                if polygon is not None:
                    c = polygon.centroid
                    hold_target = (c.x, c.y)
            if hold_target is None:
                # Default: stay where the agent is right now
                hold_target = position
            self._hold_targets[agent_id] = hold_target
            # Set immediately
            try:
                jps_sim.set_agent_target(agent_id, hold_target)
            except Exception:
                pass
            self._last_hold_refresh[agent_id] = current_sim_time
            return

        # Periodic refresh to resist crowd-pressure drift
        if current_sim_time - self._last_hold_refresh.get(agent_id, 0.0) >= self._HOLD_REFRESH_INTERVAL:
            try:
                jps_sim.set_agent_target(agent_id, hold_target)
            except Exception:
                pass
            self._last_hold_refresh[agent_id] = current_sim_time

    def _step_patrol(
        self,
        agent_id: str,
        phase: dict,
        current_sim_time: float,
        jps_sim: Any,
        position: tuple[float, float],
    ) -> None:
        """Advance patrol waypoint logic for a single director agent (phase-aware)."""
        waypoints = phase["patrol_waypoints"]
        if not waypoints:
            return

        current_idx = self._patrol_index.get(agent_id, 0)
        waypoint = waypoints[current_idx]
        target_pos = waypoint["pos"]
        target_level = waypoint["level_id"]

        current_level: str | None = (
            jps_sim.agent_levels.get(agent_id) if hasattr(jps_sim, "agent_levels") else None
        )

        # -----------------------------------------------------------------
        # Cross-level movement: route to the nearest escalator ONCE per leg.
        #
        # Calling set_agent_target every physics tick (0.05 s) creates a new
        # JuPedSim waypoint stage each time and resets the agent's journey
        # before they can take a single step — they appear stuck.  We issue
        # the routing command only when the target level changes, then leave
        # the agent's journey unchanged until MultiLevelSimulation teleports
        # them to the correct level.
        # -----------------------------------------------------------------
        if current_level is not None and current_level != target_level:
            routing_to = self._cross_level_routing.get(agent_id)
            if routing_to != target_level:
                self._route_to_escalator(agent_id, position, current_level, target_level, jps_sim)
                self._cross_level_routing[agent_id] = target_level
            return  # nothing else to do until the level transfer completes

        # Agent is on the correct level.  If we just finished a level transfer,
        # reset the dwell timer so it starts fresh on the new level, and start
        # the grace period that suppresses on_reach_zone triggers until the agent
        # has had time to walk clear of the escalator arrival zone.
        if self._cross_level_routing.get(agent_id) is not None:
            self._cross_level_routing[agent_id] = None
            self._patrol_arrived_at[agent_id] = -phase["patrol_dwell_time"]
            grace_until = current_sim_time + self._TRANSFER_GRACE_PERIOD
            self._phase_transition_grace_until[agent_id] = grace_until
            logger.debug(
                f"[{self.system_name}] {agent_id} transferred to level {current_level}; "
                f"on_reach_zone suppressed until t={grace_until:.1f}s"
            )

        arrived_at = self._patrol_arrived_at.get(agent_id, -phase["patrol_dwell_time"])

        dx = position[0] - target_pos[0]
        dy = position[1] - target_pos[1]
        dist = (dx * dx + dy * dy) ** 0.5
        arrival_threshold = 2.0

        if dist < arrival_threshold:
            if arrived_at < 0 or current_sim_time - arrived_at < 0:
                self._patrol_arrived_at[agent_id] = current_sim_time
            elif current_sim_time - arrived_at >= phase["patrol_dwell_time"]:
                next_idx = (current_idx + 1) % len(waypoints)
                self._patrol_index[agent_id] = next_idx
                next_wp = waypoints[next_idx]
                # Clear cross-level state so the next leg issues a fresh route
                self._cross_level_routing[agent_id] = None
                self._patrol_arrived_at[agent_id] = -phase["patrol_dwell_time"]
                if current_level is None or next_wp["level_id"] == current_level:
                    try:
                        jps_sim.set_agent_target(agent_id, next_wp["pos"])
                    except Exception:
                        pass
                logger.debug(
                    f"[{self.system_name}] {agent_id} → patrol waypoint "
                    f"{next_idx}: {next_wp['pos']} (level {next_wp['level_id']})"
                )
        else:
            if arrived_at < 0:
                try:
                    jps_sim.set_agent_target(agent_id, target_pos)
                except Exception:
                    pass

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _route_to_escalator(
        self,
        agent_id: str,
        position: tuple[float, float],
        current_level: str,
        target_level: str,
        jps_sim: Any,
    ) -> None:
        """
        Set the agent's target to the nearest *departure* escalator zone on
        ``current_level`` that leads toward ``target_level``.

        Only zones whose direction suffix matches the travel direction are
        considered (``_down`` when descending, ``_up`` when ascending).  This
        avoids routing the agent into arrival-only zones which would trigger a
        direction-violation and reroute them back via an evacuation exit.
        """
        if not hasattr(jps_sim, "transfer_manager"):
            logger.debug(
                f"[{self.system_name}] {agent_id}: cross-level patrol requested but "
                "sim has no transfer_manager — cannot route to escalator."
            )
            return

        # Determine travel direction
        going_down = float(target_level) < float(current_level)
        direction_suffix = "_down" if going_down else "_up"

        prefix = f"L{current_level}_esc_"
        esc_zones = {
            name: poly
            for name, poly in jps_sim.transfer_manager.escalator_zones.items()
            if name.startswith(prefix) and name.endswith(direction_suffix)
        }
        if not esc_zones:
            logger.warning(
                f"[{self.system_name}] {agent_id}: no '{direction_suffix}' escalator zones "
                f"found on level {current_level} — cannot route to next patrol waypoint."
            )
            return

        # Pick the nearest zone and derive the exit name from the zone name.
        # Zone naming: L{level}_esc_{letter}_{direction}
        # Exit naming: escalator_{letter}_{direction}
        # JuPedSim only removes an agent from a level (triggering the transfer)
        # when they reach a registered *exit* stage, not a plain waypoint.  We
        # must use set_agent_evacuation_exit so the agent is properly processed
        # by _process_escalator_exits on the next step.
        best_zone: str | None = None
        best_dist = float("inf")
        for zone_name, poly in esc_zones.items():
            c = poly.centroid
            d = (position[0] - c.x) ** 2 + (position[1] - c.y) ** 2
            if d < best_dist:
                best_dist = d
                best_zone = zone_name

        if best_zone is not None:
            # L{level}_esc_{letter}_{direction} → escalator_{letter}_{direction}
            parts = best_zone.split("_esc_", 1)
            exit_name = f"escalator_{parts[1]}" if len(parts) == 2 else None
            try:
                if exit_name and hasattr(jps_sim, "set_agent_evacuation_exit"):
                    jps_sim.set_agent_evacuation_exit(agent_id, exit_name)
                    logger.debug(
                        f"[{self.system_name}] {agent_id} routed to escalator exit "
                        f"'{exit_name}' (level {current_level} → {target_level})"
                    )
                else:
                    # Fallback: single-level sim — use waypoint
                    c = esc_zones[best_zone].centroid
                    jps_sim.set_agent_target(agent_id, (c.x, c.y))
            except Exception as exc:
                logger.warning(
                    f"[{self.system_name}] {agent_id}: could not route to escalator "
                    f"'{exit_name}': {exc}"
                )

    def _resolve_patrol_waypoints(
        self,
        entries: list[dict[str, Any]],
        zones_polygons: dict[str, Any],
    ) -> list[dict]:
        """Resolve patrol zone entries to dicts of {pos: (x, y), level_id: str}."""
        waypoints: list[dict] = []
        for entry in entries:
            pos = self._resolve_position(entry, zones_polygons, f"{self.system_name}_patrol")
            if pos is not None:
                level_id = str(entry.get("level_id", "0"))
                waypoints.append({"pos": pos, "level_id": level_id})
        return waypoints

    @staticmethod
    def _resolve_position(
        entry: dict[str, Any],
        zones_polygons: dict[str, Any],
        label: str,
    ) -> tuple[float, float] | None:
        """
        Resolve a spawn / patrol entry to (x, y).

        Explicit ``position`` takes precedence over ``zone`` centroid lookup.
        """
        if "position" in entry:
            raw = entry["position"]
            try:
                return (float(raw[0]), float(raw[1]))
            except (TypeError, IndexError, ValueError) as exc:
                logger.warning(f"[{label}] Invalid position entry {raw}: {exc}")
                return None

        zone_name = entry.get("zone")
        if zone_name:
            polygon = zones_polygons.get(zone_name)
            if polygon is None:
                logger.warning(
                    f"[{label}] Zone '{zone_name}' not found in zones_polygons; "
                    f"available: {list(zones_polygons.keys())}"
                )
                return None
            try:
                centroid = polygon.centroid
                return (centroid.x, centroid.y)
            except Exception as exc:
                logger.warning(
                    f"[{label}] Could not compute centroid for zone '{zone_name}': {exc}"
                )
                return None

        logger.warning(f"[{label}] Spawn entry has neither 'position' nor 'zone': {entry}")
        return None
