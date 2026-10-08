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

import contextlib
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

from evacusim.concordia.agent_builder import AgentBuilder
from evacusim.conventions import is_train_exit
from evacusim.coordination.observation_coordinator import ObservationCoordinator
from evacusim.coordination.simulation_state_queries import SimulationStateQueries
from evacusim.decision.action_executor import ActionExecutor
from evacusim.decision.decision_processor import DecisionProcessor
from evacusim.decision.situation import goal_is_train_oriented, zone_containing
from evacusim.jps.exit_tracker import ExitTracker
from evacusim.jps.simulation_interface import PedestrianSimulation
from evacusim.metrics.llm_cost_reporter import FinancialReporter
from evacusim.metrics.population_monitor import PopulationMonitor
from evacusim.metrics.results_writer import ResultsWriter
from evacusim.systems.director_system import DirectorSystem
from evacusim.systems.event_manager import EventManager
from evacusim.systems.messaging import MessageSystem
from evacusim.translation import ActionTranslator, ObservationGenerator
from evacusim.utils.logger import get_logger
from evacusim.utils.performance_monitor import PerformanceTimer
from evacusim.visualization.position_history import PositionHistoryTracker

logger = get_logger(__name__)


class SimulationError(RuntimeError):
    """The simulation failed; partial outputs were written before raising."""


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
            logger.info(
                f"[pre-spawn] System '{name}' set up ({len(system.agent_ids)} director agent(s))"
            )
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
        start_time_s: float = 0.0,
        output_file: Path | None = None,
        enable_video: bool = False,
        monitoring_config: dict[str, Any] | None = None,
        performance_config: dict[str, Any] | None = None,
        systems_config: dict[str, Any] | None = None,
        decision_prompt_template_path: str | None = None,
        pace_to_realtime: bool = False,
        pre_built_systems: list | None = None,
        pre_built_agent_roles: dict | None = None,
        decision_engine=None,
        spawn_controller=None,
        defer_initial_decisions: bool = False,
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
            start_time_s: Absolute simulation clock time at the first step.
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
            defer_initial_decisions: Delay bootstrap until the factory has
                prepared event state for a non-zero absolute start time.
        """
        self.jps_sim = jupedsim_simulation
        self.station_layout = station_layout
        self.model = language_model
        self.embedder = embedder
        self.decision_interval = decision_interval
        self.max_steps = max_steps
        self.start_time_s = max(0.0, float(start_time_s))
        self.output_file = output_file
        self.enable_video = enable_video
        self.pace_to_realtime = pace_to_realtime
        self.performance_config = performance_config or {}

        # Feature A: runtime (Poisson/timetable) passenger spawning. When a
        # spawn controller is provided, agents are inserted mid-run as their
        # arrival times are reached. Runtime spawning builds LLM-free
        # NoOpAgents, so it requires the rule-based engine (decision_engine set).
        self.spawn_controller = spawn_controller
        self.spawn_log: list[dict[str, Any]] = []
        if spawn_controller is not None and decision_engine is None:
            raise ValueError(
                "Runtime passenger spawning (calibration) requires the rule-based "
                "decision engine; it is LLM-free only. Set decision.engine: rule_based."
            )

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
        self._staff_systems: list[Any] = []
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
        self.action_translator = ActionTranslator(station_layout, self.jps_sim)
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

        # Build the agent registry. In LLM mode each agent is a Concordia entity
        # (memory bank + language model + sentence embedder). When a non-LLM
        # decision engine is injected we skip Concordia entirely — no embedder is
        # loaded and the language model is never called — and register lightweight
        # no-op agents that only need to absorb broadcast observations.
        self._decision_engine = decision_engine
        if decision_engine is None:
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
        else:
            from evacusim.coordination.noop_agent import NoOpAgent

            self.concordia_agents = {
                cfg["id"]: NoOpAgent(cfg["id"], cfg.get("name")) for cfg in agents_config
            }
            self.agent_injured = {cfg["id"] for cfg in agents_config if cfg.get("is_injured")}
            logger.info(
                "Non-LLM decision engine (%s) selected — skipped Concordia agent "
                "construction for %d agents (no embedder, no model calls).",
                type(decision_engine).__name__,
                len(self.concordia_agents),
            )

        # Tracking
        # Seeded below once group cadence is known.
        self.last_decision_time = self.start_time_s
        self.current_sim_time = self.start_time_s
        self.current_step = 0  # Track current simulation step for logging
        self.agent_decisions: dict[str, dict[str, Any]] = {}

        # Route changing tracking
        self.agent_destinations: dict[str, str] = {}  # agent_id -> current exit name

        # Track exited agents (those who have evacuated)
        self.exited_agents: set[str] = set()  # agent_ids who have reached exits

        # Event management
        self.event_manager = EventManager(station_layout, jupedsim_simulation)

        # Exit tracking with validation
        # Every agent that leaves, with the exit it used and the time it did so.
        # Serialised as exit_log.csv; nothing else persists this.
        self.exit_log: list[dict[str, Any]] = []

        self.exit_tracker = ExitTracker(
            concordia_agents=self.concordia_agents,
            exited_agents=self.exited_agents,
            agent_destinations=self.agent_destinations,
            jps_sim=jupedsim_simulation,
            station_layout=station_layout,  # For exit validation
            exit_validation_radius=15.0,  # Agents must be within 15m of exit
            exit_log=self.exit_log,
        )

        # Waiting and information seeking tracking
        self.wait_events: list[dict[str, Any]] = []  # Track all wait decisions with reasons

        # Agent-to-agent messaging
        self.message_system = MessageSystem(default_radius=10.0)
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
            agent_roles=self.agent_roles,
            pace_multipliers=self.performance_config.get("pace_multipliers", {}),
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
            perf_timer=self.perf_timer,
            jps_sim=self.jps_sim,
            agent_configs=agents_config,
            llm_semaphore_limit=int(self.performance_config.get("max_parallel_agents", 10)),
            per_agent_timeout_secs=self.performance_config.get("decision_timeout_seconds", 30.0),
            min_redecision_interval_secs=float(
                self.performance_config.get("min_redecision_interval_seconds", 0.0)
            ),
            wait_nudge_enabled=bool(self.performance_config.get("wait_nudge_enabled", False)),
            decision_prompt_template_path=decision_prompt_template_path,
            decision_engine=decision_engine,
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
        self._last_step_error: str | None = None

        # Write every 200 steps (10 s at dt=0.05 s) instead of every 10 steps (0.5 s).
        # This reduces I/O traffic by 20× while keeping the live viewer reasonably fresh.
        self._write_interval_steps: int = 200

        # Staggered decision groups (Opt 9c).
        # Agents are divided into N_GROUPS groups; only one group is processed each
        # decision cycle.  This spreads LLM requests evenly across time, reducing
        # peak concurrency and preventing Azure rate-limit bursts.
        # Configure via performance.decision_groups (default 3). Set to 1 for
        # all-at-once behaviour each decision tick.
        self._decision_groups: int = max(1, int(self.performance_config.get("decision_groups", 3)))
        agent_ids_sorted = sorted(self.concordia_agents.keys())
        n = len(agent_ids_sorted)
        self._agent_groups: list[list[str]] = [
            agent_ids_sorted[i :: self._decision_groups] for i in range(self._decision_groups)
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

        # Bootstrap: run a full all-agent decision cycle at the configured start
        # time so every agent has a destination before physics starts.
        self._bootstrap_initial_decisions_enabled = bool(
            self.performance_config.get("bootstrap_initial_decisions", True)
        )
        if self._bootstrap_initial_decisions_enabled and not defer_initial_decisions:
            self.last_decision_time = self.start_time_s - self._group_decision_interval
            self._bootstrap_initial_decisions()
        elif self._bootstrap_initial_decisions_enabled:
            logger.info("Deferring bootstrap decisions until start-time state is prepared.")
        else:
            logger.info(
                "Skipping t=0 all-agent bootstrap decisions; using staggered runtime decisions."
            )

    # ------------------------------------------------------------------
    # System management
    # ------------------------------------------------------------------

    def register_runtime_agent(self, cfg: dict[str, Any], position, level_id) -> bool:
        """Insert a runtime-spawned passenger into the simulation and pipeline.

        Adds the agent to JuPedSim, registers an LLM-free ``NoOpAgent`` in the
        shared ``concordia_agents`` map (so exit/observation/decision machinery
        picks it up on the next cycle), and hands its config to the decision
        processor.  The agent spawns holding at its spawn point (no implicit
        destination) so it doesn't move on the level's default street-exit
        journey before the rule-based engine assigns its real route on its
        first decision; ``get_default_exit()`` always returns the first exit
        registered for the level (``blackett_street`` here), so a spawn point
        placed inside — or near — that exit's own polygon would otherwise be
        detected as "arrived" and removed before ever getting a decision (this
        is why blackett_street-spawned boarders were vanishing at spawn while
        the same exit worked fine for alighters leaving through it).  Returns
        False (and logs) if physical insertion fails (e.g. the spawn point is
        occupied).
        """
        agent_id = cfg["id"]
        walking_speed = float(cfg.get("walking_speed", 1.34))
        try:
            if hasattr(self.jps_sim, "simulations"):
                self.jps_sim.add_agent(
                    agent_id,
                    position,
                    walking_speed=walking_speed,
                    level_id=str(level_id),
                    assign_default_destination=False,
                )
            else:
                self.jps_sim.add_agent(
                    agent_id,
                    position,
                    walking_speed=walking_speed,
                    assign_default_destination=False,
                )
        except Exception as e:
            # Expected occasionally (occupied point, jitter outside walkable);
            # the caller retries with a larger jitter, so log at debug.
            logger.debug(
                "Runtime spawn attempt failed for %s at %s (level %s): %s",
                agent_id,
                position,
                level_id,
                e,
            )
            return False

        from evacusim.coordination.noop_agent import NoOpAgent

        self.concordia_agents[agent_id] = NoOpAgent(agent_id, cfg.get("name"))
        self.agent_configs.append(cfg)
        self.decision_processor.register_agent(cfg)
        # The staggered decision groups are fixed at init from the initial
        # population (empty for a calibration run), so a runtime-spawned agent is
        # in no rotation group and would never receive a decision — it would only
        # follow the default-destination journey set at spawn, never the
        # rule-based goal routing (e.g. leave_by_train).  Flag it for an immediate
        # out-of-group decision so it routes correctly on the step it appears.
        self._pending_immediate_decisions.add(agent_id)
        self.spawn_log.append(
            {
                "id": agent_id,
                "source": cfg.get("spawn_source"),
                "location": cfg.get("spawn_location"),
                "time_s": cfg.get("spawn_time_s"),
                "level": str(level_id),
                "dest_exit": cfg.get("target"),
            }
        )
        logger.debug(
            "Runtime-spawned %s (%s) at %s level %s -> %s",
            agent_id,
            cfg.get("spawn_source"),
            position,
            level_id,
            cfg.get("target"),
        )
        return True

    _SPAWN_MAX_ATTEMPTS = 12  # jitter-retry budget per runtime arrival

    def _spawn_arrivals(self, current_sim_time: float) -> None:
        """Spawn any passengers whose scheduled arrival time has been reached.

        No-op unless a spawn controller was configured (calibration runs).
        """
        if self.spawn_controller is None:
            return
        for event in self.spawn_controller.pop_due(current_sim_time):
            cfg, position, level_id = self.spawn_controller.build_agent_cfg(event)
            if self.register_runtime_agent(cfg, position, level_id):
                continue
            # Retry with a progressively larger jitter disc so bursts of
            # simultaneous arrivals (and points that land near a wall) spread
            # out until a valid, non-colliding position is found.
            placed = False
            for attempt in range(1, self._SPAWN_MAX_ATTEMPTS):
                position, level_id = self.spawn_controller.jittered_position(event, attempt=attempt)
                cfg["start_position"] = position
                if self.register_runtime_agent(cfg, position, level_id):
                    placed = True
                    break
            if not placed:
                logger.warning(
                    "Dropped runtime arrival %s (source=%s, location=%s): no valid "
                    "spawn position after %d attempts.",
                    cfg["id"],
                    cfg.get("spawn_source"),
                    cfg.get("spawn_location"),
                    self._SPAWN_MAX_ATTEMPTS,
                )

    def _in_idle_gap(self) -> bool:
        """True when the station is empty and the next arrival is still ahead.

        Only meaningful for calibration runs (a spawn controller is present).
        In that state there is nothing to step — no live agents to move, board,
        or exit — so the runner can skip the per-step body and spin cheaply to
        the next scheduled arrival.  Guards against draining the schedule: once
        no arrivals remain, ``peek_next_time()`` is ``None`` and this returns
        ``False`` so the normal completion path runs.
        """
        if self.spawn_controller is None:
            return False
        live = len(self.concordia_agents) - len(self.exited_agents)
        if live > 0:
            return False
        nxt = self.spawn_controller.peek_next_time()
        return nxt is not None and nxt > self.current_sim_time

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
        """Let staff (e.g. RCIs, fire brigade) move and give directives."""
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
        """The smallest named zone the agent is in (e.g. ``platform_1``, not ``level_-1``).

        Used to address zone-specific PA announcements and staff directives.
        """
        pos = self.state_queries.get_agent_position(agent_id)
        if pos is None:
            return None
        return zone_containing(pos, getattr(self.action_translator, "zones_polygons", {}))

    def _bootstrap_initial_decisions(self) -> None:
        """Run one decision cycle at the configured time before the first step.

        At bootstrap we process ALL agents regardless of group so every agent
        has an initial journey before physics starts.
        """
        initial_time = self.start_time_s
        logger.info(f"Bootstrapping initial agent decisions at t={initial_time:.1f}s")
        observations = self.observation_coordinator.generate_all_observations(initial_time)
        self.last_decision_time = self.decision_processor.process_all_agents(
            observations, initial_time
        )

    # ------------------------------------------------------------------
    # The simulation loop
    # ------------------------------------------------------------------

    def run(self) -> dict[str, Any]:
        """
        Run the simulation to the end (or until everyone has left).

        Each step (see :meth:`_run_step`): spawn arrivals, advance the
        pedestrian physics, remove agents who left or boarded, queue agents
        who must re-decide, let staff act, fire scheduled events, run a
        decision cycle when one is due, and record outputs.

        Returns:
            Dictionary with simulation results and statistics

        Raises:
            SimulationError: The simulation failed. Partial outputs (population
                series, calibration report, position frames) are written first.
        """
        logger.info("Starting hybrid Concordia + JuPedSim simulation")
        start_time = time.time()
        self._failure: BaseException | None = None
        results = {
            "steps": 0,
            "sim_time": 0.0,
            "decisions_made": 0,
            "events_triggered": 0,
            "agents": {},
        }

        try:
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
                    outcome = self._run_step(step)
                    if outcome == "stop":
                        break
                    if outcome == "skip":
                        continue
                    results["steps"] = step + 1
                    results["sim_time"] = self.current_sim_time
                    progress.update(task, advance=1)
                    if outcome == "idle":
                        continue
                    self._pace_to_realtime(step_start)
        except KeyboardInterrupt:
            logger.info("Simulation interrupted by user")
        except Exception as e:
            logger.error(f"Simulation error: {e}", exc_info=True)
            self._failure = e
        finally:
            # Drain any in-flight background write so results aren't truncated.
            if self._pending_write is not None:
                with contextlib.suppress(Exception):
                    self._pending_write.result(timeout=30)
            self._io_executor.shutdown(wait=False)

        self._finish(results, start_time)

        failure = self._failure
        if failure is not None:
            if isinstance(failure, SimulationError):
                raise failure
            raise SimulationError(
                f"simulation failed at t={self.current_sim_time:.2f}s: {failure}"
            ) from failure
        return results

    def _run_step(self, step: int) -> str:
        """One simulation step.

        Returns:
            ``"ok"``; ``"idle"`` when the station is empty and the step was
            skipped; ``"skip"`` when physics has nobody to move but arrivals
            are still due (the step is not counted); ``"stop"`` to end the run.
        """
        self.current_step = step
        self.current_sim_time = self.start_time_s + step * self.jps_sim.dt

        # 1. Arrivals (calibration runs), before physics so an initially empty
        #    station still fills.
        self._spawn_arrivals(self.current_sim_time)
        if self._in_idle_gap():
            # Nobody to move until the next arrival: keep only the zero samples.
            self.population_monitor.record_snapshot(self.current_sim_time, self.exited_agents)
            return "idle"

        # 2. Pedestrian physics.
        physics = self._advance_physics()
        if physics != "ok":
            return physics

        # 3. Agents who left the station or boarded a train.
        self.exit_tracker.check_exited_agents(self.current_sim_time, self.current_step)
        self._board_trains()
        self.population_monitor.record_snapshot(self.current_sim_time, self.exited_agents)

        # 4. Agents whose situation changed and who must re-decide now.
        force_immediate_cycle = self._queue_transferred_agents()
        force_immediate_cycle = self._queue_bounced_agents() or force_immediate_cycle

        # 5. Staff move and give directives (every step, so directive timing is exact).
        self._step_systems(self.current_sim_time)

        # 6. Scheduled events (alarm, PA, trains, closures).
        new_event_fired, critical_event_fired, fired_event_types = self._fire_events()

        # 7. A decision cycle, when one is due.
        self._run_decision_cycle(
            new_event_fired, critical_event_fired, fired_event_types, force_immediate_cycle
        )

        # 8. Outputs.
        self._record_step(step)
        return "ok"

    def _advance_physics(self) -> str:
        """Step the pedestrian simulation: ``"ok"``, ``"skip"`` or ``"stop"``."""
        # The physics layer latches "complete" the first time it steps with an
        # empty population, which for a calibration run is t=0, before the first
        # arrival. While arrivals are still queued (or agents remain) clear that
        # latch so stepping resumes and spawned passengers actually move.
        if (
            self.spawn_controller is not None
            and getattr(self.jps_sim, "is_complete", False)
            and (
                self.spawn_controller.remaining > 0
                or len(self.concordia_agents) > len(self.exited_agents)
            )
        ):
            self.jps_sim.is_complete = False

        with self.perf_timer.measure("jupedsim_step"):
            if self._step_jupedsim():
                return "ok"
        if self._last_step_error:
            self._failure = SimulationError(
                f"physics step failed at t={self.current_sim_time:.2f}s: {self._last_step_error}"
            )
            logger.error(str(self._failure))
            return "stop"
        # No agents remain this step; keep looping if arrivals are still due.
        if self.spawn_controller is not None and self.spawn_controller.remaining > 0:
            return "skip"
        logger.info("JuPedSim simulation complete")
        return "stop"

    def _board_trains(self) -> None:
        """Board waiting passengers onto trains dwelling at their platform.

        Eligible are agents whose goal is a train and who are not routing away
        (no destination, or a train exit). Alighters (goal: leave the station)
        are never re-boarded onto the train they just left. Boarded agents are
        added to ``exited_agents`` so the exit tracker does not count them again.
        """
        if not (
            hasattr(self.jps_sim, "board_agents_on_platform")
            and self.event_manager.active_train_exits
        ):
            return
        boarder_ids = {
            aid
            for aid, goal in self.decision_processor.agent_goals.items()
            if aid not in self.exited_agents
            and goal_is_train_oriented(goal)
            and (
                not self.agent_destinations.get(aid, "")
                or is_train_exit(self.agent_destinations.get(aid, ""))
            )
        }
        for exit_name in sorted(self.event_manager.active_train_exits):
            for cid in self.jps_sim.board_agents_on_platform(
                exit_name,
                agent_destinations=self.agent_destinations,
                eligible_ids=boarder_ids,
            ):
                if cid in self.exited_agents:
                    continue
                self.exited_agents.add(cid)
                self.agent_destinations[cid] = exit_name
                self.decision_processor.agent_goals.pop(cid, None)
                self.exit_log.append(
                    {
                        "agent_id": cid,
                        "exit_name": exit_name,
                        "intended_exit": exit_name,
                        "exit_distance_m": "",
                        "time_s": round(self.current_sim_time, 2),
                        "level": getattr(self.jps_sim, "platform_level", "-1"),
                        "x": "",
                        "y": "",
                        "validated": True,
                    }
                )

    def _forget_route(self, agent_id: str) -> None:
        """Clear an agent's route commitment so its next decision is fresh."""
        self.agent_destinations.pop(agent_id, None)
        self.decision_processor.clear_goal_for_redecision(agent_id)
        self.decision_processor.reset_agent_decision(agent_id)

    def _queue_transferred_agents(self) -> bool:
        """Queue agents who just changed level for a fresh, level-aware decision.

        Returns True if an immediate decision cycle should run
        (``performance.immediate_redecision_on_transfer``).
        """
        if not hasattr(self.jps_sim, "consume_recently_transferred_agents"):
            return False
        transferred = self.jps_sim.consume_recently_transferred_agents()
        if not transferred:
            return False
        logger.info(f"Transferred agents queued for immediate decision cycle: {transferred}")
        for agent_id in transferred:
            # Defensive recovery for any observer that saw the source/destination
            # handoff gap as an exit.
            self.exited_agents.discard(agent_id)
            self._forget_route(agent_id)
            self._pending_immediate_decisions.add(agent_id)
        if bool(self.performance_config.get("immediate_redecision_on_transfer", True)):
            return True
        logger.info(
            "Deferring transferred-agent re-decisions to next scheduled cycle for better "
            "batching (performance.immediate_redecision_on_transfer=false)"
        )
        return False

    def _queue_bounced_agents(self) -> bool:
        """Queue agents turned back at a blocked corridor; True if any were."""
        if not getattr(self.jps_sim, "agents_needing_redecision", None):
            return False
        bounced = set(self.jps_sim.agents_needing_redecision)
        self.jps_sim.agents_needing_redecision.clear()
        logger.info(f"Blocked-corridor contacts queued for immediate re-decision: {bounced}")
        if hasattr(self.observation_coordinator, "remember_blocked_exits_for_agents"):
            self.observation_coordinator.remember_blocked_exits_for_agents(
                bounced, set(self.event_manager.blocked_exits)
            )
        for agent_id in bounced:
            self._forget_route(agent_id)
            self._pending_immediate_decisions.add(agent_id)
        return True

    def _fire_events(self) -> tuple[bool, bool, set[str]]:
        """Fire due events and handle departed trains.

        Returns ``(new_event_fired, critical_event_fired, fired_event_types)``.
        Blocked exits and departing trains are *critical*: everyone re-decides
        at once. Agents heading for a train that has left re-decide too.
        """
        with self.perf_timer.measure("event_checking"):
            new_event_fired = self.event_manager.check_and_trigger_events(
                self.current_sim_time,
                self.concordia_agents,
                message_system=self.message_system,
                exited_agents=self.exited_agents,
                zone_id_for_agent_fn=self._get_zone_id_for_agent,
            )
        fired_event_types = set(getattr(self.event_manager, "last_fired_event_types", set()))
        critical_event_fired = bool(
            fired_event_types.intersection({"block_exit", "train_departure"})
        )

        active_train_exits = self.event_manager.active_train_exits
        stranded = [
            aid
            for aid, dest in self.agent_destinations.items()
            if is_train_exit(dest)
            and dest not in active_train_exits
            and aid not in self.exited_agents
        ]
        if stranded:
            for aid in stranded:
                self._forget_route(aid)
            new_event_fired = True
            logger.info(
                f"Train departed — cleared stale destinations for "
                f"{len(stranded)} stranded agent(s); forced decision cycle."
            )

        # Staff systems activated "on_event" start acting once any event fires.
        if new_event_fired:
            for system in self._staff_systems:
                system.notify_event_fired()
        return new_event_fired, critical_event_fired, fired_event_types

    def _run_decision_cycle(
        self,
        new_event_fired: bool,
        critical_event_fired: bool,
        fired_event_types: set[str],
        force_immediate_cycle: bool,
    ) -> None:
        """Run a decision cycle if one is due, for the agents who should decide.

        - A critical event: everyone decides now.
        - Agents queued for an immediate decision (changed level, turned back):
          only they decide, without disturbing the staggered cadence.
        - Otherwise, on schedule: the next group of the staggered rotation
          (everyone, with ``performance.decision_groups: 1``).
        """
        should_decide = self._should_make_decisions()
        has_pending_immediate = bool(self._pending_immediate_decisions)
        if not (
            should_decide or force_immediate_cycle or critical_event_fired or has_pending_immediate
        ):
            if new_event_fired:
                logger.info(
                    "Informational event(s) fired (%s) — deferring broad "
                    "re-decisions to scheduled staggered cycle",
                    ", ".join(sorted(fired_event_types)) or "unknown",
                )
            return

        with self.perf_timer.measure("agent_decisions_total"):
            if critical_event_fired:
                current_group = None  # None → all agents
                logger.info(
                    "Critical event(s) fired (%s) — triggering immediate all-agent decision cycle",
                    ", ".join(sorted(fired_event_types)),
                )
            elif force_immediate_cycle or has_pending_immediate:
                current_group = []
                logger.info(
                    "Immediate targeted decision cycle for pending transferred/re-routed agents"
                )
            else:
                current_group = (
                    self._agent_groups[self._current_group_index]
                    if self._decision_groups > 1
                    else None
                )

            # Add agents awaiting an out-of-group decision to this batch.
            if self._pending_immediate_decisions:
                pending = {
                    a for a in self._pending_immediate_decisions if a not in self.exited_agents
                }
                self._pending_immediate_decisions.clear()
                if pending and current_group is not None:
                    extras = pending - set(current_group)
                    if extras:
                        current_group = list(current_group) + sorted(extras)
                        logger.info(
                            f"Added {len(extras)} recently-transferred "
                            f"agent(s) to current decision batch: {extras}"
                        )

            # Only scheduled cycles advance the staggered rotation.
            if should_decide:
                self._current_group_index = (self._current_group_index + 1) % self._decision_groups

            if current_group is not None:
                current_group = [a for a in current_group if a not in self.exited_agents]

            # Observations only for deciding agents (they still perceive everyone).
            with self.perf_timer.measure("generate_observations"):
                observations = self.observation_coordinator.generate_all_observations(
                    self.current_sim_time, agent_ids=current_group
                )
            with self.perf_timer.measure("decision_processing"):
                cycle_time = self.decision_processor.process_all_agents(
                    observations, self.current_sim_time, agent_ids=current_group
                )
                # Deferred agents (clearing an escalator) keep their current
                # route and are reconsidered on the normal cadence.
                self.decision_processor.consume_deferred_escalator_agents()
                # Targeted cycles keep the global cadence.
                if new_event_fired or should_decide:
                    self.last_decision_time = cycle_time

    def _record_step(self, step: int) -> None:
        """Position frames, the live-viewer sidecar, and periodic result writes."""
        agent_levels = (
            dict(self.jps_sim.agent_levels) if hasattr(self.jps_sim, "agent_levels") else None
        )
        if self.position_tracker and step % 10 == 0:
            self.position_tracker.save_frame(
                self.current_sim_time,
                self.jps_sim.get_all_agent_positions(),
                self.agent_decisions,
                self.event_manager.blocked_exits,
                active_train_exits=self.event_manager.active_train_exits,
                agent_levels=agent_levels,
                escalators=(
                    self.jps_sim.escalator_system.frame_snapshot()
                    if hasattr(self.jps_sim, "escalator_system")
                    else None
                ),
            )

        # Lightweight positions sidecar for the live viewer.
        if self.output_file and step % 10 == 0:
            with self.perf_timer.measure("file_io"):
                ResultsWriter.save_positions_only(
                    self.output_file,
                    self.jps_sim.get_all_agent_positions(),
                    self.current_sim_time,
                    agent_levels=agent_levels,
                    blocked_exits=self.event_manager.blocked_exits,
                    agent_roles=self.agent_roles if self.agent_roles else None,
                    active_train_exits=self.event_manager.active_train_exits,
                )

        # Incremental results, written in a background thread; wait for the
        # previous write first so two never touch the file at once.
        if self.output_file and step % self._write_interval_steps == 0:
            if self._pending_write is not None and not self._pending_write.done():
                self._pending_write.result()
            snapshot_decisions = {
                k: {"decisions": list(v["decisions"])} for k, v in self.agent_decisions.items()
            }
            with self.perf_timer.measure("file_io"):
                self._pending_write = self._io_executor.submit(
                    ResultsWriter.save_incremental,
                    self.output_file,
                    snapshot_decisions,
                    dict(self.jps_sim.get_all_agent_positions()),
                    self.current_sim_time,
                    list(self.event_manager.event_history),
                    set(self.event_manager.blocked_exits),
                    list(self.message_system.message_history),
                    self.decision_interval,
                    self.max_steps,
                    len(self.concordia_agents),
                    agent_levels,
                )

    def _pace_to_realtime(self, step_start: float) -> None:
        """With a live viewer, slow the run to real time for smooth display."""
        if self.pace_to_realtime and self.jps_sim.dt > 0:
            sleep_time = self.jps_sim.dt - (time.perf_counter() - step_start)
            if sleep_time > 0:
                time.sleep(sleep_time)

    def _finish(self, results: dict[str, Any], start_time: float) -> None:
        """Final statistics, reports and end-of-run outputs."""
        elapsed_time = time.time() - start_time
        results["elapsed_time"] = elapsed_time
        results["decisions_made"] = sum(
            len(d.get("decisions", [])) for d in self.agent_decisions.values()
        )
        results["events_triggered"] = len(self.event_manager.event_history)

        logger.info(
            f"Simulation {'FAILED' if self._failure else 'complete'}: {results['steps']} steps, "
            f"{results['sim_time']:.1f}s sim time, "
            f"{elapsed_time:.1f}s real time"
        )
        print(self.perf_timer.report())
        print(FinancialReporter.generate_report(self.llm_provider, len(self.concordia_agents)))

        # force=True records the final state even when the last periodic
        # interval falls just past the end time.
        self.population_monitor.record_snapshot(
            self.current_sim_time, self.exited_agents, force=True
        )
        self.population_monitor.display_summary()
        if self.output_file:
            self.population_monitor.save(self.output_file.parent)
        results["population_timeseries"] = self.population_monitor.to_dict()

        if self.spawn_controller is not None and self.output_file is not None:
            try:
                from evacusim.calibration.calibration_report import write_calibration_report

                write_calibration_report(
                    getattr(self.spawn_controller, "expected_intervals", []) or [],
                    self.spawn_log,
                    self.population_monitor,
                    self.output_file.parent,
                )
            except Exception as e:
                logger.error(f"Failed to write calibration report: {e}", exc_info=True)

        if self.position_tracker and self.output_file:
            history_file = self.output_file.parent / f"{self.output_file.stem}_history.jsonl"
            self.position_tracker.save_to_file(history_file)
            results["position_history_file"] = str(history_file)

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
                exit_log=self.exit_log,
                spawn_log=self.spawn_log,
                escalator_system=getattr(self.jps_sim, "escalator_system", None),
            )
            logger.info(f"Partial results saved to {self.output_file}")

    def _step_jupedsim(self) -> bool:
        """
        Advance JuPedSim simulation by one timestep.

        Returns:
            True if simulation should continue, False if complete
        """
        try:
            self._last_step_error = None
            # Keep jps_sim's blocked_exits in sync so the physics layer can
            # intercept agents that reach a blocked escalator exit on levels
            # where no geometry obstacle could be placed (e.g. level -1).
            if hasattr(self.jps_sim, "blocked_exits"):
                self.jps_sim.blocked_exits = set(self.event_manager.blocked_exits)
            return self.jps_sim.step()
        except Exception as e:
            self._last_step_error = str(e)
            logger.error(f"JuPedSim step error: {e}")
            return False

    def _should_make_decisions(self) -> bool:
        """Check if it's time for agents to make decisions."""
        return (self.current_sim_time - self.last_decision_time) >= self._group_decision_interval
