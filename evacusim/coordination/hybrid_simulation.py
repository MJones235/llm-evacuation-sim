"""
Hybrid simulation runner that integrates Concordia with JuPedSim.

This module implements the translation layer between:
- Concordia: Agent cognition and decision-making
- JuPedSim: Pedestrian movement simulation

Key features:
- Event-driven LLM queries (not every timestep)
- Batch processing of agent decisions
- Translation of NL actions to waypoints
- Observation generation from simulation state
"""

import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any

from concordia.language_model import language_model
from concordia.typing import entity as entity_lib
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from evacusim.utils.logger import get_logger
from evacusim.concordia.agent_builder import AgentBuilder
from evacusim.coordination.observation_coordinator import ObservationCoordinator
from evacusim.coordination.simulation_state_queries import SimulationStateQueries
from evacusim.decision.action_executor import ActionExecutor
from evacusim.decision.decision_processor import DecisionProcessor
from evacusim.jps.exit_tracker import ExitTracker
from evacusim.jps.simulation_interface import PedestrianSimulation
from evacusim.metrics.llm_cost_reporter import FinancialReporter
from evacusim.metrics.population_monitor import PopulationMonitor
from evacusim.metrics.results_writer import ResultsWriter
from evacusim.systems.event_manager import EventManager
from evacusim.systems.messaging import MessageSystem
from evacusim.translation import ActionTranslator, ObservationGenerator
from evacusim.systems.director_system import DirectorSystem
from evacusim.utils.performance_monitor import PerformanceTimer
from evacusim.visualization.position_history import PositionHistoryTracker

logger = get_logger(__name__)


class HybridSimulationRunner:
    """
    Manages the hybrid Concordia + JuPedSim simulation.

    Architecture:
    1. JuPedSim runs continuously at fine time resolution (dt=0.05s)
    2. Concordia agents make decisions at coarse intervals (5-10s)
    3. Decisions are triggered by events (announcements, observations)
    4. Actions are translated to JuPedSim waypoints
    5. Simulation state is converted to observations for agents
    """

    @staticmethod
    def build_systems_for_pre_spawn(
        systems_config: dict[str, Any] | None,
        jps_sim: Any,
        station_layout: dict[str, Any],
    ) -> tuple[list, dict]:
        """
        Create and set up director systems independently of runner init so that
        director agents occupy JuPedSim positions *before* random passengers are
        spawned.  This prevents spawn-position collisions where a passenger lands
        on top of a fire-marshal spawn point.

        Returns:
            (systems_list, agent_roles_dict) — pass these as ``pre_built_systems``
            and ``pre_built_agent_roles`` to ``HybridSimulationRunner.__init__()``.
        """
        systems: list = []
        agent_roles: dict = {}
        systems_config = HybridSimulationRunner._normalize_systems_config(systems_config)
        for name, cfg in systems_config.items():
            if not cfg.get("enabled", False):
                continue
            system = DirectorSystem(name, cfg)
            system.setup(jps_sim, station_layout, agent_roles)
            systems.append(system)
            logger.info(f"[pre-spawn] System '{name}' set up ({len(system.agent_ids)} director agent(s))")
        return systems, agent_roles

    @staticmethod
    def _normalize_systems_config(systems_config: Any) -> dict[str, Any]:
        """Return a safe systems config mapping.

        ``systems: null`` (or missing systems) is valid and treated as no systems.
        """
        if systems_config is None:
            return {}
        if not isinstance(systems_config, dict):
            logger.warning(
                "Invalid 'systems' config type %s; expected mapping. Ignoring systems.",
                type(systems_config).__name__,
            )
            return {}
        return systems_config

    def __init__(
        self,
        jupedsim_simulation: PedestrianSimulation,
        agents_config: list[dict[str, Any]],
        station_layout: dict[str, Any],
        language_model: language_model.LanguageModel,
        embedder: Any,  # Sentence embedder function
        decision_interval: float = 5.0,
        max_steps: int = 3600,
        output_file: Path | None = None,
        enable_video: bool = False,
        monitoring_config: dict[str, Any] | None = None,
        performance_config: dict[str, Any] | None = None,
        systems_config: dict[str, Any] | None = None,
        decision_prompt_template_path: str | None = None,
        pace_to_realtime: bool = False,
        pre_built_systems: list | None = None,
        pre_built_agent_roles: dict | None = None,
    ):
        """
        Initialize the hybrid simulation runner.

        Args:
            jupedsim_simulation: Pedestrian simulation backend (implements PedestrianSimulation)
            agents_config: List of agent configuration dictionaries
            station_layout: Station geometry and exit information
            language_model: LLM for Concordia agents
            embedder: Sentence embedding function
            decision_interval: Time between Concordia decisions (seconds)
            max_steps: Maximum simulation steps
            output_file: Path to output file for saving results
            enable_video: Whether to track position history for video generation
            monitoring_config: Optional monitoring configuration dict with keys
                ``interval_seconds`` and ``zones`` (list of zone spec dicts).
                If ``None``, PopulationMonitor defaults are used.
            performance_config: Optional performance tuning config dict.
                Supported keys include ``max_parallel_agents``,
                ``decision_timeout_seconds``, and ``wait_nudge_enabled``.
            systems_config: Optional ``systems`` block from the scenario config.
                Each key is a system name (e.g. ``"staff"``); the value is the
                system configuration dict.  Systems with ``enabled: true`` are
                initialised and stepped each simulation cycle.
            decision_prompt_template_path: Optional path to a text template used
                to build each agent decision prompt. If omitted, the built-in
                default template file is used.
            pace_to_realtime: When True each simulation step sleeps until
                ``jps_sim.dt`` seconds have elapsed, matching wall-clock to
                simulation time (useful for live viewers).  When False the
                simulation runs as fast as possible.  Defaults to False.
        """
        self.jps_sim = jupedsim_simulation
        self.station_layout = station_layout
        self.model = language_model
        self.embedder = embedder
        self.decision_interval = decision_interval
        self.max_steps = max_steps
        self.output_file = output_file
        self.enable_video = enable_video
        self.pace_to_realtime = pace_to_realtime
        self.performance_config = performance_config or {}

        # Roles map: agent_id → human-readable role label.
        # Populated during setup (e.g. by StaffSystem) before the simulation loop.
        # The label appears in other agents' observations and message attributions.
        self.agent_roles: dict[str, str] = {}

        # Initialise and setup any configured rule-based systems (e.g. staff)
        # BEFORE Concordia agents are built so that system agents are already in
        # JuPedSim and registered in agent_roles when the first observations fire.
        # When pre_built_systems is provided the systems were already set up (and
        # their director agents already added to JuPedSim) before random passengers
        # were spawned, preventing spawn-position collisions.
        self._staff_systems: list[StaffSystem] = []
        if pre_built_systems is not None:
            self._staff_systems = list(pre_built_systems)
            self.agent_roles.update(pre_built_agent_roles or {})
        else:
            self._init_systems(systems_config or {}, jupedsim_simulation, station_layout)

        # Simulation state queries
        self.state_queries = SimulationStateQueries(jupedsim_simulation)

        # Store LLM provider reference (for usage stats)
        # The language_model is an AzureLLMConcordia instance directly
        self.llm_provider = language_model if hasattr(language_model, "get_usage_stats") else None

        # Translation layer components
        self.action_translator = ActionTranslator(station_layout, language_model, self.jps_sim)
        self.observation_generator = ObservationGenerator(station_layout, self.jps_sim)

        # Build Concordia agents (each with their own memory bank)
        self.concordia_agents: dict[str, entity_lib.Entity] = {}
        self.agent_configs = agents_config

        # Agent state tracking (three independent dimensions)
        # 1. Physical capability: Is agent injured/slow?
        self.agent_injured: set[str] = set()

        # 2. Current action: What are they doing right now?
        self.agent_action: dict[str, str] = {}  # agent_id -> "moving"|"waiting"

        # 3. Memory of last decision: What did they commit to?
        self.agent_last_decision: dict[str, dict] = {}  # agent_id -> translated_action dict

        # Build agents using AgentBuilder (parallel initialization for faster startup)
        agent_builder = AgentBuilder(
            language_model=language_model,
            embedder=embedder,
            station_layout=station_layout,
        )

        # Build agents asynchronously for faster initialization
        import asyncio

        self.concordia_agents, injured_agents = asyncio.run(
            agent_builder.build_agents(agents_config)
        )
        self.agent_injured = injured_agents

        # Tracking
        # Seeded below once group cadence is known.
        self.last_decision_time = 0.0
        self.current_sim_time = 0.0
        self.current_step = 0  # Track current simulation step for logging
        self.agent_decisions: dict[str, dict[str, Any]] = {}
        self.last_observations: dict[str, str] = {}  # Cache observations for change detection
        self.last_actions: dict[str, str] = {}  # Cache actions to reuse

        # Route changing tracking
        self.agent_destinations: dict[str, str] = {}  # agent_id -> current exit name

        # Track exited agents (those who have evacuated)
        self.exited_agents: set[str] = set()  # agent_ids who have reached exits

        # Event management
        self.event_manager = EventManager(station_layout, jupedsim_simulation)

        # Exit tracking with validation
        self.exit_tracker = ExitTracker(
            concordia_agents=self.concordia_agents,
            exited_agents=self.exited_agents,
            agent_destinations=self.agent_destinations,
            jps_sim=jupedsim_simulation,
            station_layout=station_layout,  # For exit validation
            exit_validation_radius=15.0,  # Agents must be within 15m of exit
        )

        # Waiting and information seeking tracking
        self.wait_events: list[dict[str, Any]] = []  # Track all wait decisions with reasons

        # Agent-to-agent messaging
        self.message_system = MessageSystem(
            default_radius=10.0,
            memory_window=60.0,
        )
        # Performance profiling (must be initialized before decision_processor)
        self.perf_timer = PerformanceTimer()
        # Action execution
        self.action_executor = ActionExecutor(
            jps_sim=jupedsim_simulation,
            state_queries=self.state_queries,
            event_manager=self.event_manager,
            station_layout=station_layout,
            agent_injured=self.agent_injured,
            agent_action=self.agent_action,
            agent_last_decision=self.agent_last_decision,
            agent_destinations=self.agent_destinations,
            wait_events=self.wait_events,
            agent_configs=agents_config,
        )

        # Decision processing
        self.decision_processor = DecisionProcessor(
            concordia_agents=self.concordia_agents,
            exited_agents=self.exited_agents,
            action_translator=self.action_translator,
            action_executor=self.action_executor,
            message_system=self.message_system,
            state_queries=self.state_queries,
            station_layout=station_layout,
            agent_decisions=self.agent_decisions,
            agent_destinations=self.agent_destinations,
            last_observations=self.last_observations,
            last_actions=self.last_actions,
            perf_timer=self.perf_timer,
            jps_sim=self.jps_sim,
            agent_configs=agents_config,
            enable_group_decisions=bool(
                self.performance_config.get("enable_group_decisions", False)
            ),
            group_decision_min_size=max(
                2, int(self.performance_config.get("group_decision_min_size", 3))
            ),
            llm_semaphore_limit=int(self.performance_config.get("max_parallel_agents", 10)),
            per_agent_timeout_secs=self.performance_config.get("decision_timeout_seconds", 30.0),
            min_redecision_interval_secs=float(
                self.performance_config.get("min_redecision_interval_seconds", 0.0)
            ),
            wait_nudge_enabled=bool(self.performance_config.get("wait_nudge_enabled", False)),
            decision_prompt_template_path=decision_prompt_template_path,
        )

        # Observation coordination
        self.observation_coordinator = ObservationCoordinator(
            concordia_agents=self.concordia_agents,
            exited_agents=self.exited_agents,
            observation_generator=self.observation_generator,
            state_queries=self.state_queries,
            event_manager=self.event_manager,
            message_system=self.message_system,
            agent_destinations=self.agent_destinations,
            agent_injured=self.agent_injured,
            agent_action=self.agent_action,
            agent_last_decision=self.agent_last_decision,
            jps_sim=self.jps_sim,
            agent_roles=self.agent_roles,
        )

        # Position history tracker for video generation
        self.position_tracker = None
        if self.enable_video:
            # Use streaming mode when an output file is configured so that
            # frames are written immediately instead of accumulating in memory.
            streaming_path = None
            if output_file is not None:
                streaming_path = output_file.parent / f"{output_file.stem}_history.jsonl"
            self.position_tracker = PositionHistoryTracker(
                save_interval=0.5, streaming_path=streaming_path
            )
            logger.info("Position history tracking enabled for video generation")

        # Population monitor — records zone occupancy at configured intervals
        monitoring_config = monitoring_config or {}
        self.population_monitor = PopulationMonitor(
            jupedsim_simulation,
            zone_specs=monitoring_config.get("zones"),  # None → use defaults
            interval_seconds=monitoring_config.get("interval_seconds", 60.0),
        )

        # Background thread pool for non-blocking incremental file writes.
        # A single worker is enough — we only ever have one pending write at a time.
        self._io_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="results_io")
        self._pending_write: Future | None = None

        # Write every 200 steps (10 s at dt=0.05 s) instead of every 10 steps (0.5 s).
        # This reduces I/O traffic by 20× while keeping the live viewer reasonably fresh.
        self._write_interval_steps: int = 200

        # Staggered decision groups (Opt 9c).
        # Agents are divided into N_GROUPS groups; only one group is processed each
        # decision cycle.  This spreads LLM requests evenly across time, reducing
        # peak concurrency and preventing Azure rate-limit bursts.
        # Configure via performance.decision_groups (default 3). Set to 1 for
        # all-at-once behaviour each decision tick.
        self._decision_groups: int = max(
            1, int(self.performance_config.get("decision_groups", 3))
        )
        agent_ids_sorted = sorted(self.concordia_agents.keys())
        n = len(agent_ids_sorted)
        self._agent_groups: list[list[str]] = [
            agent_ids_sorted[i::self._decision_groups]
            for i in range(self._decision_groups)
        ]
        # Decision ticks happen every decision_interval / groups so each individual
        # agent still re-decides roughly every decision_interval.
        self._group_decision_interval: float = (
            self.decision_interval / self._decision_groups
            if self._decision_groups > 1
            else self.decision_interval
        )
        # Without bootstrap we still want the first staggered batch at t=0.
        self.last_decision_time = -self._group_decision_interval
        self._current_group_index: int = 0
        logger.info(
            f"Staggered decision groups: {self._decision_groups} groups "
            f"of ~{n // self._decision_groups} agents each "
            f"(group tick {self._group_decision_interval:.2f}s, "
            f"per-agent cadence ~{self.decision_interval:.1f}s)"
        )

        # Agents that need a decision at the *next* scheduled cycle regardless of
        # which rotation group they belong to.  Populated when agents transfer
        # between levels so they don't wait up to one full per-agent cadence
        # before their group's turn fires.
        self._pending_immediate_decisions: set[str] = set()

        # Bootstrap: run a full all-agent decision cycle at t=0 so every agent
        # has an LLM-chosen destination before physics starts. Can be disabled
        # via performance.bootstrap_initial_decisions: false if needed.
        self._bootstrap_initial_decisions_enabled = bool(
            self.performance_config.get("bootstrap_initial_decisions", True)
        )
        if self._bootstrap_initial_decisions_enabled:
            self.last_decision_time = -self._group_decision_interval
            self._bootstrap_initial_decisions()
        else:
            logger.info(
                "Skipping t=0 all-agent bootstrap decisions; using staggered runtime decisions."
            )

    # ------------------------------------------------------------------
    # System management
    # ------------------------------------------------------------------

    def _init_systems(
        self,
        systems_config: dict[str, Any] | None,
        jps_sim: Any,
        station_layout: dict[str, Any],
    ) -> None:
        """Initialise and setup all enabled rule-based systems from config."""
        systems_config = self._normalize_systems_config(systems_config)
        for name, cfg in systems_config.items():
            if not cfg.get("enabled", False):
                continue
            system = DirectorSystem(name, cfg)
            system.setup(jps_sim, station_layout, self.agent_roles)
            self._staff_systems.append(system)
            logger.info(f"System '{name}' initialised ({len(system.agent_ids)} director agent(s))")

    def _step_systems(self, current_sim_time: float) -> None:
        """Call step() on all active rule-based systems."""
        for system in self._staff_systems:
            system.step(
                current_sim_time=current_sim_time,
                jps_sim=self.jps_sim,
                message_system=self.message_system,
                state_queries=self.state_queries,
                exited_agents=self.exited_agents,
                zone_id_for_agent_fn=self._get_zone_id_for_agent,
            )

    def _get_zone_id_for_agent(self, agent_id: str) -> str | None:
        """Return the zone_id the given agent is currently in, or None."""
        pos = self.state_queries.get_agent_position(agent_id)
        if pos is None:
            return None
        zones_polygons = getattr(self.action_translator, "zones_polygons", {})
        if not zones_polygons:
            return None
        from shapely.geometry import Point as _Point
        pt = _Point(pos)
        for z_id, poly in zones_polygons.items():
            try:
                if poly.covers(pt) or poly.contains(pt):
                    return z_id
            except Exception:
                pass
        return None

    def _bootstrap_initial_decisions(self) -> None:
        """Run one decision cycle at t=0 before the first JuPedSim step.

        At bootstrap we process ALL agents regardless of group so every agent
        has an initial journey before physics starts.
        """
        try:
            logger.info("Bootstrapping initial agent decisions at t=0.0s")
            initial_time = 0.0
            observations = self.observation_coordinator.generate_all_observations(initial_time)
            self.last_decision_time = self.decision_processor.process_all_agents(
                observations, initial_time
                # agent_ids=None → processes all agents
            )
        except Exception as e:
            logger.error(f"Failed to bootstrap initial decisions: {e}", exc_info=True)
            # Continue with normal runtime decision flow as fallback

    def run(self) -> dict[str, Any]:
        """
        Run the hybrid simulation.

        Returns:
            Dictionary with simulation results and statistics
        """
        logger.info("Starting hybrid Concordia + JuPedSim simulation")
        start_time = time.time()

        results = {
            "steps": 0,
            "sim_time": 0.0,
            "decisions_made": 0,
            "events_triggered": 0,
            "agents": {},
        }

        try:
            # Main simulation loop with progress bar
            with Progress(
                SpinnerColumn(),
                TextColumn("[bold blue]Simulating:"),
                BarColumn(complete_style="green", finished_style="bold green"),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TextColumn("•"),
                TextColumn("Step {task.completed}/{task.total}"),
                TextColumn("•"),
                TimeElapsedColumn(),
                TextColumn("•"),
                TimeRemainingColumn(),
            ) as progress:
                task = progress.add_task("simulation", total=self.max_steps)

                for step in range(self.max_steps):
                    step_start = time.perf_counter()
                    self.current_step = step
                    force_immediate_decision_cycle = False

                    # Advance JuPedSim simulation
                    with self.perf_timer.measure("jupedsim_step"):
                        if not self._step_jupedsim():
                            logger.info("JuPedSim simulation complete")
                            break

                    self.current_sim_time = step * self.jps_sim.dt

                    # Check for agents who have exited and remove them
                    self.exit_tracker.check_exited_agents(self.current_sim_time, self.current_step)

                    # Board agents who have explicitly committed to boarding this
                    # train (destination == exit_name).  Agents merely waiting on the
                    # platform or routing to an escalator are left untouched.
                    # We add boarded agents to exited_agents here so that if
                    # exit_tracker also sees them disappear next step it won't
                    # double-count them.
                    if (
                        hasattr(self.jps_sim, "board_agents_on_platform")
                        and self.event_manager.active_train_exits
                    ):
                        for _exit_name in list(self.event_manager.active_train_exits):
                            for _cid in self.jps_sim.board_agents_on_platform(
                                _exit_name,
                                agent_destinations=self.agent_destinations,
                            ):
                                if _cid not in self.exited_agents:
                                    self.exited_agents.add(_cid)
                                    self.agent_destinations[_cid] = _exit_name
                                    self.decision_processor.agent_goals.pop(_cid, None)

                    # Record population snapshot every simulation minute
                    self.population_monitor.record_snapshot(
                        self.current_sim_time, self.exited_agents
                    )

                    # Drain the recently-transferred set.  Transferred agents are
                    # given a temporary destination so they keep moving.
                    # Clear all stale route commitments so each agent makes a
                    # fresh, level-informed decision when they next get a turn.
                    if hasattr(self.jps_sim, "consume_recently_transferred_agents"):
                        transferred_agents = self.jps_sim.consume_recently_transferred_agents()
                        if transferred_agents:
                            immediate_transfer_redecision = bool(
                                self.performance_config.get(
                                    "immediate_redecision_on_transfer", True
                                )
                            )
                            logger.info(
                                f"Transferred agents queued for immediate decision cycle: "
                                f"{transferred_agents}"
                            )
                            for _tid in transferred_agents:
                                self.agent_destinations.pop(_tid, None)
                                self.decision_processor.agent_goals.pop(_tid, None)
                                self.decision_processor.prompt_cache.clear_agent(_tid)
                                self._pending_immediate_decisions.add(_tid)
                            if immediate_transfer_redecision:
                                force_immediate_decision_cycle = True
                            else:
                                logger.info(
                                    "Deferring transferred-agent re-decisions to next "
                                    "scheduled cycle for better batching "
                                    "(performance.immediate_redecision_on_transfer=false)"
                                )

                    # Consume agents that were bounced back from a blocked escalator.
                    # Clear their stale route commitments and schedule an immediate
                    # re-decision so they pick an unblocked exit next cycle.
                    if hasattr(self.jps_sim, "agents_needing_redecision") and self.jps_sim.agents_needing_redecision:
                        bounced = set(self.jps_sim.agents_needing_redecision)
                        self.jps_sim.agents_needing_redecision.clear()
                        logger.info(
                            f"Bounced agents re-deciding after blocked escalator: {bounced}"
                        )
                        for _bid in bounced:
                            self.agent_destinations.pop(_bid, None)
                            self.decision_processor.agent_goals.pop(_bid, None)
                            self.decision_processor.prompt_cache.clear_agent(_bid)
                            self._pending_immediate_decisions.add(_bid)
                        force_immediate_decision_cycle = True
                    # decisions, meaning the alarm was missed for the entire
                    # decision cycle that coincided with the alarm time).
                    with self.perf_timer.measure("event_checking"):
                        new_event_fired = self.event_manager.check_and_trigger_events(
                            self.current_sim_time,
                            self.concordia_agents,
                            message_system=self.message_system,
                            exited_agents=self.exited_agents,
                            zone_id_for_agent_fn=self._get_zone_id_for_agent,
                        )
                    fired_event_types = set(
                        getattr(self.event_manager, "last_fired_event_types", set())
                    )
                    critical_event_fired = bool(
                        fired_event_types.intersection({"block_exit", "train_departure"})
                    )

                    # When a train departs its exits are removed from active_train_exits.
                    # Clear any agent destination commitments that now point at a
                    # closed train exit so those agents make a fresh decision.
                    active_train_exits = self.event_manager.active_train_exits
                    stranded = [
                        aid for aid, dest in self.agent_destinations.items()
                        if dest.startswith("train_platform_")
                        and dest not in active_train_exits
                        and aid not in self.exited_agents
                    ]
                    if stranded:
                        for aid in stranded:
                            self.agent_destinations.pop(aid, None)
                            self.decision_processor.agent_goals.pop(aid, None)
                            self.decision_processor.prompt_cache.clear_agent(aid)
                        new_event_fired = True
                        logger.info(
                            f"Train departed — cleared stale destinations for "
                            f"{len(stranded)} stranded agent(s); forced decision cycle."
                        )

                    # Notify director systems when any event fires so that
                    # activate_on_event systems begin acting.
                    if new_event_fired:
                        for system in self._staff_systems:
                            system.notify_event_fired()

                    # Check if it's time for Concordia decisions (normal schedule) or
                    # if a critical event just fired (immediate all-agent override).
                    should_decide = self._should_make_decisions()
                    should_run_decisions = (
                        should_decide or force_immediate_decision_cycle or critical_event_fired
                    )
                    if new_event_fired and not critical_event_fired and not should_run_decisions:
                        logger.info(
                            "Informational event(s) fired (%s) — deferring broad "
                            "re-decisions to scheduled staggered cycle",
                            ", ".join(sorted(fired_event_types)) or "unknown",
                        )
                    if should_run_decisions:
                        with self.perf_timer.measure("agent_decisions_total"):
                            if critical_event_fired:
                                # Critical events bypass grouping so everyone can
                                # re-route promptly (e.g., blocked exits/closures).
                                current_group = None  # None → process all agents
                                logger.info(
                                    "Critical event(s) fired (%s) — triggering "
                                    "immediate all-agent decision cycle",
                                    ", ".join(sorted(fired_event_types)),
                                )
                            elif force_immediate_decision_cycle:
                                # Targeted immediate cycle: process only pending
                                # out-of-group agents (e.g. transferred/bounced)
                                # without disturbing the normal staggered cadence.
                                current_group = []
                                logger.info(
                                    "Immediate targeted decision cycle for pending "
                                    "transferred/re-routed agents"
                                )
                            else:
                                # Normal scheduled cycle: rotate through groups.
                                current_group = (
                                    self._agent_groups[self._current_group_index]
                                    if self._decision_groups > 1
                                    else None
                                )

                            # Merge any agents awaiting an out-of-group decision
                            # (e.g. recently transferred) into the current batch.
                            if self._pending_immediate_decisions:
                                pending = {
                                    a for a in self._pending_immediate_decisions
                                    if a not in self.exited_agents
                                }
                                self._pending_immediate_decisions.clear()
                                if pending:
                                    if current_group is None:
                                        # All-agents cycle — pending are already included
                                        pass
                                    else:
                                        extras = pending - set(current_group)
                                        if extras:
                                            current_group = list(current_group) + list(extras)
                                            logger.info(
                                                f"Added {len(extras)} recently-transferred "
                                                f"agent(s) to current decision batch: {extras}"
                                            )

                            # Advance the group index only on normally-scheduled
                            # cycles so the regular staggered cadence is preserved.
                            if should_decide:
                                self._current_group_index = (
                                    self._current_group_index + 1
                                ) % self._decision_groups

                            # Remove agents who have since exited from the group list.
                            if current_group is not None:
                                current_group = [
                                    a for a in current_group if a not in self.exited_agents
                                ]

                            # Generate observations for all agents (even those not
                            # deciding this cycle — their state may be read by others).
                            with self.perf_timer.measure("generate_observations"):
                                observations = (
                                    self.observation_coordinator.generate_all_observations(
                                        self.current_sim_time
                                    )
                                )
                            # Process the current group's decisions in parallel
                            with self.perf_timer.measure("decision_processing"):
                                cycle_time = self.decision_processor.process_all_agents(
                                    observations,
                                    self.current_sim_time,
                                    agent_ids=current_group,
                                )
                                # Preserve global cadence on targeted immediate
                                # cycles; only update last_decision_time for
                                # normal schedule ticks or global event overrides.
                                if new_event_fired or should_decide:
                                    self.last_decision_time = cycle_time

                    # Track position history for video generation (every 0.5s)
                    if self.position_tracker and step % 10 == 0:
                        self.position_tracker.save_frame(
                            self.current_sim_time,
                            self.jps_sim.get_all_agent_positions(),
                            self.agent_decisions,
                            self.event_manager.blocked_exits,
                            active_train_exits=self.event_manager.active_train_exits,
                            agent_levels=(
                                dict(self.jps_sim.agent_levels)
                                if hasattr(self.jps_sim, "agent_levels")
                                else None
                            ),
                        )

                    # Lightweight positions sidecar — every 10 steps (0.5 s).
                    # Updates agent_positions and current_time for the live viewer
                    # without the cost of serialising the full decisions/messages dict.
                    if self.output_file and step % 10 == 0:
                        with self.perf_timer.measure("file_io"):
                            _pos_levels = (
                                dict(self.jps_sim.agent_levels)
                                if hasattr(self.jps_sim, "agent_levels")
                                else None
                            )
                            ResultsWriter.save_positions_only(
                                self.output_file,
                                self.jps_sim.get_all_agent_positions(),
                                self.current_sim_time,
                                agent_levels=_pos_levels,
                                blocked_exits=self.event_manager.blocked_exits,
                                agent_roles=self.agent_roles if self.agent_roles else None,
                                active_train_exits=self.event_manager.active_train_exits,
                            )

                    # Incremental results write — every _write_interval_steps steps (10s
                    # at dt=0.05s).  The write runs in a background thread so the main loop
                    # is not blocked by disk I/O.  We wait for the previous write to finish
                    # before submitting a new one to avoid concurrent writes to the same file.
                    if self.output_file and step % self._write_interval_steps == 0:
                        # Block only if the previous background write is still running
                        # (this should be negligible given the 10s gap between writes).
                        if self._pending_write is not None and not self._pending_write.done():
                            self._pending_write.result()

                        agent_levels = (
                            dict(self.jps_sim.agent_levels)
                            if hasattr(self.jps_sim, "agent_levels")
                            else None
                        )
                        # Snapshot mutable state that could change while the write runs.
                        snapshot_decisions = {
                            k: {"decisions": list(v["decisions"])}
                            for k, v in self.agent_decisions.items()
                        }
                        snapshot_positions = dict(self.jps_sim.get_all_agent_positions())
                        snapshot_events = list(self.event_manager.event_history)
                        snapshot_blocked = set(self.event_manager.blocked_exits)
                        snapshot_messages = list(self.message_system.message_history)
                        snapshot_time = self.current_sim_time

                        with self.perf_timer.measure("file_io"):
                            self._pending_write = self._io_executor.submit(
                                ResultsWriter.save_incremental,
                                self.output_file,
                                snapshot_decisions,
                                snapshot_positions,
                                snapshot_time,
                                snapshot_events,
                                snapshot_blocked,
                                snapshot_messages,
                                self.decision_interval,
                                self.max_steps,
                                len(self.concordia_agents),
                                agent_levels,
                            )

                    results["steps"] = step + 1
                    results["sim_time"] = self.current_sim_time

                    # Update progress bar
                    progress.update(task, advance=1)

                    # Pace simulation to real time for smooth visualization.
                    # Only active when a live viewer is running; disabled by
                    # default so headless runs finish as fast as possible.
                    if self.pace_to_realtime and self.jps_sim.dt > 0:
                        elapsed = time.perf_counter() - step_start
                        sleep_time = self.jps_sim.dt - elapsed
                        if sleep_time > 0:
                            time.sleep(sleep_time)

        except KeyboardInterrupt:
            logger.info("Simulation interrupted by user")
        except Exception as e:
            logger.error(f"Simulation error: {e}", exc_info=True)
        finally:
            # Drain any in-flight background write so results aren't truncated.
            if self._pending_write is not None:
                try:
                    self._pending_write.result(timeout=30)
                except Exception:
                    pass
            self._io_executor.shutdown(wait=False)

        # Compute final statistics
        elapsed_time = time.time() - start_time
        results["elapsed_time"] = elapsed_time
        results["decisions_made"] = sum(
            len(d.get("decisions", [])) for d in self.agent_decisions.values()
        )
        results["events_triggered"] = len(self.event_manager.event_history)

        logger.info(
            f"Simulation complete: {results['steps']} steps, "
            f"{results['sim_time']:.1f}s sim time, "
            f"{elapsed_time:.1f}s real time"
        )

        # Print performance profile
        print(self.perf_timer.report())

        # Print financial report
        print(FinancialReporter.generate_report(self.llm_provider, len(self.concordia_agents)))

        # Display and save population time series.
        # force=True ensures the final state is always recorded even when the
        # last periodic interval (e.g. t=300 s) falls just past the actual end
        # time (e.g. t=299.95 s) and the normal guard would skip it.
        self.population_monitor.record_snapshot(self.current_sim_time, self.exited_agents, force=True)
        self.population_monitor.display_summary()
        if self.output_file:
            self.population_monitor.save(self.output_file.parent)
        results["population_timeseries"] = self.population_monitor.to_dict()

        # Save position history if video generation is enabled
        if self.position_tracker and self.output_file:
            # Use .jsonl extension for the streaming format; viewers that expect
            # the legacy .json wrapper can still read via save_to_file().
            history_file = self.output_file.parent / f"{self.output_file.stem}_history.jsonl"
            self.position_tracker.save_to_file(history_file)
            results["position_history_file"] = str(history_file)

        return results

    def cleanup(self):
        """Save partial results when simulation is interrupted."""
        logger.warning("Cleaning up simulation state...")

        # Save position history if available
        if self.position_tracker and self.output_file:
            history_file = self.output_file.parent / f"{self.output_file.stem}_history.jsonl"
            self.position_tracker.save_to_file(history_file)
            logger.info(f"Position history saved to {history_file}")

        # Save partial decision results
        if self.output_file:
            # Get agent levels for multi-level simulations
            agent_levels = None
            if hasattr(self.jps_sim, "agent_levels"):
                agent_levels = self.jps_sim.agent_levels

            ResultsWriter.save_final_results(
                self.output_file,
                self.agent_decisions,
                self.jps_sim.get_all_agent_positions(),
                self.current_sim_time,
                self.event_manager.event_history,
                self.event_manager.blocked_exits,
                self.message_system.message_history,
                self.wait_events,
                self.decision_interval,
                self.max_steps,
                len(self.concordia_agents),
                self.perf_timer.report(),
                self.llm_provider,
                agent_levels,
                self.agent_roles if self.agent_roles else None,
            )
            logger.info(f"Partial results saved to {self.output_file}")

    def _step_jupedsim(self) -> bool:
        """
        Advance JuPedSim simulation by one timestep.

        Returns:
            True if simulation should continue, False if complete
        """
        try:
            # Keep jps_sim's blocked_exits in sync so the physics layer can
            # intercept agents that reach a blocked escalator exit on levels
            # where no geometry obstacle could be placed (e.g. level -1).
            if hasattr(self.jps_sim, "blocked_exits"):
                self.jps_sim.blocked_exits = set(self.event_manager.blocked_exits)
            return self.jps_sim.step()
        except Exception as e:
            logger.error(f"JuPedSim step error: {e}")
            return False

    def _should_make_decisions(self) -> bool:
        """Check if it's time for agents to make decisions."""
        return (self.current_sim_time - self.last_decision_time) >= self._group_decision_interval
