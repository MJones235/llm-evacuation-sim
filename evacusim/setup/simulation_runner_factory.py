"""Build the simulation runner from a run's parameters.

This module is responsible for:
- Creating and configuring the HybridSimulationRunner
- Selecting the decision engine
- Building runtime passenger spawning for calibration runs
- Scheduling the scenario's events
"""

import logging
from pathlib import Path

from evacusim.config.schema import (
    CalibrationConfig,
    DecisionConfig,
    Event,
    RuleBasedDecisionConfig,
    RunConfig,
    as_dict,
)
from evacusim.coordination.hybrid_simulation import HybridSimulationRunner
from evacusim.utils.logger import get_logger, setup_logger

logger = get_logger(__name__)


class SimulationRunnerFactory:
    """Handles creation and configuration of simulation runners."""

    @staticmethod
    def create_runner(
        jps_sim,
        agents_config: list,
        station_layout: dict,
        model,
        embedder,
        decisions_file: Path,
        params: RunConfig,
        pace_to_realtime: bool = False,
        pre_built_systems: list | None = None,
        pre_built_agent_roles: dict | None = None,
    ) -> HybridSimulationRunner:
        """
        Create and configure a HybridSimulationRunner.

        Args:
            jps_sim: JuPedSim simulation instance
            agents_config: List of agent configuration dictionaries
            station_layout: Station layout dictionary
            model: Language model instance
            embedder: Sentence embedder function
            decisions_file: Path to decisions output file
            params: The run's parameters

        Returns:
            Configured HybridSimulationRunner ready to run

        Raises:
            Exception: If runner initialization fails
        """
        start_time_s = params.simulation.start_time_s

        # The LLM engine is the default; a rule-based (LLM-free) engine builds
        # no Concordia agents or embedder.
        decision_engine = SimulationRunnerFactory._build_decision_engine(params.decision)

        # Calibration runs spawn passengers at runtime from usage and timetable
        # data (normal operations, no evacuation).
        spawn_controller, calibration_timetable = SimulationRunnerFactory._build_calibration(
            params.calibration
        )

        logger.info("Creating HybridSimulationRunner...")

        # Persist logs alongside run artifacts so transfer/discharge traces can
        # be inspected after the run without relying on terminal output.
        # DEBUG-level file logging emits several lines per physics step (zone
        # transfers, escalator state, etc.), which is useful when debugging a
        # specific run but adds real per-step formatting/IO overhead over a long
        # run. Default to INFO; opt back into DEBUG via
        # performance.file_log_level in the experiment config when needed.
        log_file = decisions_file.parent / "simulation.log"
        file_log_level = logging.getLevelName(params.performance.file_log_level)
        setup_logger(
            log_file=log_file,
            console_level=logging.INFO,
            file_level=file_log_level,
        )
        logger.info(
            f"Per-run log file: {log_file} (file_level={logging.getLevelName(file_log_level)})"
        )

        try:
            runner = HybridSimulationRunner(
                jupedsim_simulation=jps_sim,
                agents_config=agents_config,
                station_layout=station_layout,
                language_model=model,
                embedder=embedder,
                decision_interval=params.simulation.decision_interval,
                max_steps=params.simulation.max_iterations,
                start_time_s=start_time_s,
                output_file=decisions_file,
                enable_video=params.video.enabled,
                monitoring_config=as_dict(params.monitoring),
                performance_config=as_dict(params.performance),
                systems_config={name: as_dict(cfg) for name, cfg in params.systems.items()},
                decision_prompt_template_path=params.prompts.decision_prompt_template_path,
                pace_to_realtime=pace_to_realtime,
                pre_built_systems=pre_built_systems,
                pre_built_agent_roles=pre_built_agent_roles,
                decision_engine=decision_engine,
                spawn_controller=spawn_controller,
                defer_initial_decisions=start_time_s > 0,
            )
            logger.info("HybridSimulationRunner initialized")
        except Exception as e:
            logger.error(f"FATAL ERROR during HybridSimulationRunner initialization: {e}")
            import traceback

            traceback.print_exc()
            raise

        # Configure events
        SimulationRunnerFactory._load_events(runner, params.events)
        SimulationRunnerFactory._load_calibration_train_events(runner, calibration_timetable)
        discarded = 0
        if runner.spawn_controller is not None and start_time_s > 0:
            discarded = runner.spawn_controller.discard_before(start_time_s)
        if start_time_s > 0:
            runner.event_manager.prepare_for_start_time(start_time_s)
            if runner._bootstrap_initial_decisions_enabled:
                runner.last_decision_time = start_time_s - runner._group_decision_interval
                runner._bootstrap_initial_decisions()
            logger.info(
                "Simulation starts at %.1fs; discarded %d earlier passenger arrivals",
                start_time_s,
                discarded,
            )

        return runner

    @staticmethod
    def _build_decision_engine(decision: DecisionConfig):
        """Construct the decision engine selected by the ``decision`` section.

        Returns ``None`` for the LLM engine (the DecisionProcessor then builds
        its own LLMDecisionEngine), or a RuleBasedDecisionEngine for an
        LLM-free run.
        """
        if not isinstance(decision, RuleBasedDecisionConfig):
            return None

        from evacusim.decision.rule_based_decision_engine import RuleBasedDecisionEngine

        weights = decision.rule_weights
        logger.info("Decision engine: rule_based (LLM-free); weights=%s", as_dict(weights))
        return RuleBasedDecisionEngine(
            w_proximity=weights.proximity,
            w_busyness=weights.busyness,
            w_familiarity=weights.familiarity,
            w_visibility=weights.visibility,
            crowd_radius_m=decision.crowd_radius_m,
        )

    @staticmethod
    def _build_calibration(calibration: CalibrationConfig | None):
        """Build the runtime spawn controller for a calibration run.

        Returns ``(spawn_controller, timetable)``. When calibration is absent
        or disabled, returns ``(None, [])`` and the run behaves normally.
        Loads the usage and timetable CSVs, builds a seeded Poisson arrival
        schedule, and wraps it in a :class:`RuntimeSpawnController`.
        """
        if calibration is None or not calibration.enabled:
            return None, []

        from evacusim.calibration.poisson_scheduler import build_arrival_schedule
        from evacusim.calibration.spawn_controller import RuntimeSpawnController
        from evacusim.calibration.usage_data import load_entrance_usage, load_timetable

        intervals = load_entrance_usage(calibration.entrance_usage_csv)
        timetable = load_timetable(calibration.timetable_csv) if calibration.timetable_csv else []

        spawn_points = {name: as_dict(point) for name, point in calibration.spawn_points.items()}
        spawn_cfg = {
            "entrance_level": calibration.entrance_level,
            "entrance_dest_exits": calibration.entrance_dest_exits,
            "platform_level": calibration.platform_level,
            "platform_exit": calibration.platform_exit,
            "train_door_counts": {
                platform: len(point.door_points)
                for platform, point in calibration.spawn_points.items()
                if platform.isdigit() and point.door_points
            },
            "train_alighting_duration_s": calibration.train_alighting_duration_s,
        }
        schedule = build_arrival_schedule(intervals, timetable, spawn_cfg, seed=calibration.seed)

        controller = RuntimeSpawnController(
            schedule,
            spawn_points,
            seed=calibration.seed,
            jitter_m=calibration.spawn_jitter_m,
            train_door_jitter_m=calibration.train_door_jitter_m,
            walking_speed_mean=calibration.walking_speed_mean,
            walking_speed_std=calibration.walking_speed_std,
            walking_speed_min=calibration.walking_speed_min,
            walking_speed_max=calibration.walking_speed_max,
            knowledge_profile=calibration.knowledge_profile,
        )
        # Retained for the end-of-run calibration report (expected vs realised).
        controller.expected_intervals = intervals

        logger.info(
            "Calibration enabled: %d scheduled arrivals "
            "(%d entrance intervals, %d trains); seed=%d",
            len(schedule),
            len(intervals),
            len(timetable),
            calibration.seed,
        )
        return controller, timetable

    @staticmethod
    def _load_calibration_train_events(runner, timetable) -> None:
        """Map each timetable train arrival to a ``train_arrival`` event.

        Reuses the existing EventManager mechanism so platform boarding exits
        activate when trains arrive (entrance passengers can then board).
        """
        for tr in timetable:
            runner.event_manager.scheduled_events.append(
                {
                    "time": float(tr.arrival_s),
                    "type": "train_arrival",
                    "platforms": [tr.platform],
                    "dwell_seconds": float(tr.dwell_s),
                    "message": f"A train has arrived at platform {tr.platform}.",
                    "_fired": False,
                }
            )
        if timetable:
            logger.info(
                "Calibration: mapped %d train arrivals to train_arrival events",
                len(timetable),
            )

    @staticmethod
    def _load_events(runner: HybridSimulationRunner, events: list[Event]) -> None:
        """Schedule the scenario's events on the runner's EventManager."""
        for event in events:
            runner.event_manager.scheduled_events.append({**as_dict(event), "_fired": False})

        if events:
            logger.info(f"Loaded {len(events)} events from configuration")
        else:
            logger.warning("No events defined in configuration")
