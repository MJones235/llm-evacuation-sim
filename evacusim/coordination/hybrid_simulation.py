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
        test_scenarios: dict[str, Any] | None = None,
        enable_video: bool = False,
        monitoring_config: dict[str, Any] | None = None,
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
            test_scenarios: Test scenario configuration
            enable_video: Whether to track position history for video generation
            monitoring_config: Optional monitoring configuration dict with keys
                ``interval_seconds`` and ``zones`` (list of zone spec dicts).
                If ``None``, PopulationMonitor defaults are used.
        """
        self.jps_sim = jupedsim_simulation
        self.station_layout = station_layout
        self.model = language_model
        self.embedder = embedder
        self.decision_interval = decision_interval
        self.max_steps = max_steps
        self.output_file = output_file
        self.test_scenarios = test_scenarios or {}
        self.enable_video = enable_video

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
        self.last_decision_time = (
            -decision_interval
        )  # Start negative so first decision happens immediately
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
        self.event_manager.setup_test_scenario(test_scenarios)

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
            test_scenarios=test_scenarios or {},
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
            test_scenarios=test_scenarios or {},
            jps_sim=self.jps_sim,
        )

        # Position history tracker for video generation
        self.position_tracker = None
        if self.enable_video:
            self.position_tracker = PositionHistoryTracker(save_interval=0.5)
            logger.info("Position history tracking enabled for video generation")

        # Population monitor — records zone occupancy at configured intervals
        monitoring_config = monitoring_config or {}
        self.population_monitor = PopulationMonitor(
            jupedsim_simulation,
            zone_specs=monitoring_config.get("zones"),  # None → use defaults
            interval_seconds=monitoring_config.get("interval_seconds", 60.0),
        )

        # Bootstrap decisions at t=0 so agents choose initial journeys before first sim step
        self._bootstrap_initial_decisions()

    def _bootstrap_initial_decisions(self) -> None:
        """Run one decision cycle at t=0 before the first JuPedSim step."""
        try:
            logger.info("Bootstrapping initial agent decisions at t=0.0s")
            initial_time = 0.0
            observations = self.observation_coordinator.generate_all_observations(initial_time)
            self.last_decision_time = self.decision_processor.process_all_agents(
                observations, initial_time
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

                    # Advance JuPedSim simulation
                    with self.perf_timer.measure("jupedsim_step"):
                        if not self._step_jupedsim():
                            logger.info("JuPedSim simulation complete")
                            break

                    self.current_sim_time = step * self.jps_sim.dt

                    # Check for agents who have exited and remove them
                    self.exit_tracker.check_exited_agents(self.current_sim_time, self.current_step)

                    # Record population snapshot every simulation minute
                    self.population_monitor.record_snapshot(
                        self.current_sim_time, self.exited_agents
                    )

                    # Drain the recently-transferred set.  Transferred agents are
                    # given a temporary destination so they keep moving
                    if hasattr(self.jps_sim, "consume_recently_transferred_agents"):
                        transferred_agents = self.jps_sim.consume_recently_transferred_agents()
                        if transferred_agents:
                            logger.info(
                                f"Transferred agents will decide at next decision cycle: "
                                f"{transferred_agents}"
                            )

                    # Check if it's time for Concordia decisions
                    if self._should_make_decisions():
                        with self.perf_timer.measure("agent_decisions_total"):
                            # Generate observations for all agents
                            with self.perf_timer.measure("generate_observations"):
                                observations = (
                                    self.observation_coordinator.generate_all_observations(
                                        self.current_sim_time
                                    )
                                )
                            # Process all agent decisions in parallel
                            with self.perf_timer.measure("decision_processing"):
                                self.last_decision_time = (
                                    self.decision_processor.process_all_agents(
                                        observations, self.current_sim_time
                                    )
                                )

                    # Check for events
                    with self.perf_timer.measure("event_checking"):
                        self.event_manager.check_and_trigger_events(self.current_sim_time)

                    # Track position history for video generation (every 0.5s)
                    if self.position_tracker and step % 10 == 0:
                        self.position_tracker.save_frame(
                            self.current_sim_time,
                            self.jps_sim.get_all_agent_positions(),
                            self.agent_decisions,
                            self.event_manager.blocked_exits,
                        )

                    # Save positions every 10 steps (0.5s) for smooth visualization
                    # Writing every single step (0.05s) is too slow for file I/O
                    if self.output_file and step % 10 == 0:
                        with self.perf_timer.measure("file_io"):
                            # Get agent levels for multi-level simulations
                            agent_levels = None
                            if hasattr(self.jps_sim, "agent_levels"):
                                agent_levels = self.jps_sim.agent_levels

                            ResultsWriter.save_incremental(
                                self.output_file,
                                self.agent_decisions,
                                self.jps_sim.get_all_agent_positions(),
                                self.current_sim_time,
                                self.event_manager.event_history,
                                self.event_manager.blocked_exits,
                                self.message_system.message_history,
                                self.decision_interval,
                                self.max_steps,
                                len(self.concordia_agents),
                                agent_levels,
                            )

                    results["steps"] = step + 1
                    results["sim_time"] = self.current_sim_time

                    # Update progress bar
                    progress.update(task, advance=1)

                    # Pace simulation to real time for smooth visualization
                    if self.jps_sim.dt > 0:
                        elapsed = time.perf_counter() - step_start
                        sleep_time = self.jps_sim.dt - elapsed
                        if sleep_time > 0:
                            time.sleep(sleep_time)

        except KeyboardInterrupt:
            logger.info("Simulation interrupted by user")
        except Exception as e:
            logger.error(f"Simulation error: {e}", exc_info=True)

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

        # Display and save population time series
        self.population_monitor.record_snapshot(self.current_sim_time, self.exited_agents)
        self.population_monitor.display_summary()
        if self.output_file:
            self.population_monitor.save(self.output_file.parent)
        results["population_timeseries"] = self.population_monitor.to_dict()

        # Save position history if video generation is enabled
        if self.position_tracker and self.output_file:
            history_file = self.output_file.parent / f"{self.output_file.stem}_history.json"
            self.position_tracker.save_to_file(history_file)
            results["position_history_file"] = str(history_file)

        return results

    def cleanup(self):
        """Save partial results when simulation is interrupted."""
        logger.warning("Cleaning up simulation state...")

        # Save position history if available
        if self.position_tracker and self.output_file:
            history_file = self.output_file.parent / f"{self.output_file.stem}_history.json"
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
            )
            logger.info(f"Partial results saved to {self.output_file}")

    def _step_jupedsim(self) -> bool:
        """
        Advance JuPedSim simulation by one timestep.

        Returns:
            True if simulation should continue, False if complete
        """
        try:
            # TODO: This works with MockJuPedSim, verify with real JuPedSim
            return self.jps_sim.step()
        except Exception as e:
            logger.error(f"JuPedSim step error: {e}")
            return False

    def _should_make_decisions(self) -> bool:
        """Check if it's time for agents to make decisions."""
        return (self.current_sim_time - self.last_decision_time) >= self.decision_interval
