"""
Simulation runner factory for Station Concordia simulations.

This module is responsible for:
- Creating and configuring HybridSimulationRunner instances
- Loading events from configuration
- Setting up test scenarios (blocked exits, etc.)
- Configuring runner parameters
"""

from pathlib import Path
import logging

from evacusim.utils.logger import get_logger, setup_logger
from evacusim.coordination.hybrid_simulation import HybridSimulationRunner

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
        config: dict,
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
            config: Full configuration dictionary

        Returns:
            Configured HybridSimulationRunner ready to run

        Raises:
            Exception: If runner initialization fails
        """
        sim_config = config.get("simulation", {})
        max_steps = sim_config.get("max_iterations", 200)
        decision_interval = sim_config.get("decision_interval", 5.0)
        start_time_s = float(sim_config.get("start_time_s", 0.0))

        # Video generation settings
        video_config = config.get("video", {})
        enable_video = video_config.get("enabled", False)

        # Monitoring settings
        monitoring_config = config.get("monitoring", {})

        # Performance settings
        performance_config = config.get("performance", {})

        # Rule-based director systems (staff, firefighters, etc.)
        systems_config = config.get("systems", {})
        prompts_config = config.get("prompts", {})
        decision_prompt_template_path = prompts_config.get("decision_prompt_template_path")

        # Select the decision engine. Default is the Concordia/LLM engine; a
        # rule-based (LLM-free) engine can be requested via the ``decision``
        # config section, in which case no Concordia agents or embedder are built.
        decision_engine = SimulationRunnerFactory._build_decision_engine(config)

        # Feature A: optional runtime passenger spawning driven by usage +
        # timetable CSVs (calibration under non-evacuation conditions).
        spawn_controller, calibration_timetable = (
            SimulationRunnerFactory._build_calibration(config)
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
        file_log_level_name = str(performance_config.get("file_log_level", "INFO")).upper()
        file_log_level = logging.getLevelName(file_log_level_name)
        if not isinstance(file_log_level, int):
            logger.warning(
                f"Invalid performance.file_log_level {file_log_level_name!r}; defaulting to INFO"
            )
            file_log_level = logging.INFO
        setup_logger(
            log_file=log_file,
            console_level=logging.INFO,
            file_level=file_log_level,
        )
        logger.info(f"Per-run log file: {log_file} (file_level={logging.getLevelName(file_log_level)})")

        try:
            runner = HybridSimulationRunner(
                jupedsim_simulation=jps_sim,
                agents_config=agents_config,
                station_layout=station_layout,
                language_model=model,
                embedder=embedder,
                decision_interval=decision_interval,
                max_steps=max_steps,
                start_time_s=start_time_s,
                output_file=decisions_file,
                enable_video=enable_video,
                monitoring_config=monitoring_config,
                performance_config=performance_config,
                systems_config=systems_config,
                decision_prompt_template_path=decision_prompt_template_path,
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
        SimulationRunnerFactory._load_events(runner, config)
        SimulationRunnerFactory._load_calibration_train_events(runner, calibration_timetable)
        discarded = 0
        if runner.spawn_controller is not None and start_time_s > 0:
            discarded = runner.spawn_controller.discard_before(start_time_s)
        if start_time_s > 0:
            runner.event_manager.prepare_for_start_time(start_time_s)
            if runner._bootstrap_initial_decisions_enabled:
                runner.last_decision_time = (
                    start_time_s - runner._group_decision_interval
                )
                runner._bootstrap_initial_decisions()
            logger.info(
                "Simulation starts at %.1fs; discarded %d earlier passenger arrivals",
                start_time_s,
                discarded,
            )

        return runner

    @staticmethod
    def _build_decision_engine(config: dict):
        """Construct the decision engine named by ``config["decision"]``.

        Returns ``None`` for the default LLM engine (the DecisionProcessor then
        builds its own LLMDecisionEngine), or a RuleBasedDecisionEngine for an
        LLM-free run.
        """
        decision_config = config.get("decision", {}) or {}
        engine_name = str(decision_config.get("engine", "llm")).lower()
        if engine_name in ("rule_based", "rule", "rules"):
            from evacusim.decision.rule_based_decision_engine import (
                RuleBasedDecisionEngine,
            )

            weights = decision_config.get("rule_weights", {}) or {}
            engine = RuleBasedDecisionEngine(
                w_proximity=float(weights.get("proximity", 0.5)),
                w_busyness=float(weights.get("busyness", 0.3)),
                w_familiarity=float(weights.get("familiarity", 0.2)),
                w_visibility=float(weights.get("visibility", 0.0)),
                crowd_radius_m=float(decision_config.get("crowd_radius_m", 5.0)),
            )
            logger.info(
                "Decision engine: rule_based (LLM-free); weights=%s",
                weights or "defaults",
            )
            return engine
        if engine_name not in ("llm", "concordia", "default"):
            logger.warning(
                "Unknown decision.engine '%s'; defaulting to the LLM engine.",
                engine_name,
            )
        return None

    @staticmethod
    def _build_calibration(config: dict):
        """Build the runtime spawn controller for a calibration run.

        Returns ``(spawn_controller, timetable)``.  When ``calibration.enabled``
        is false/absent, returns ``(None, [])`` and the run behaves normally.
        Loads the usage + timetable CSVs, builds a seeded Poisson arrival
        schedule, and wraps it in a :class:`RuntimeSpawnController`.
        """
        calibration = config.get("calibration") or {}
        if not calibration.get("enabled", False):
            return None, []

        from evacusim.calibration.usage_data import (
            load_entrance_usage,
            load_timetable,
        )
        from evacusim.calibration.poisson_scheduler import build_arrival_schedule
        from evacusim.calibration.spawn_controller import RuntimeSpawnController

        intervals = load_entrance_usage(calibration["entrance_usage_csv"])
        timetable = (
            load_timetable(calibration["timetable_csv"])
            if calibration.get("timetable_csv")
            else []
        )

        spawn_points = calibration["spawn_points"]
        spawn_cfg = {
            "entrance_level": str(calibration.get("entrance_level", "0")),
            "entrance_dest_exits": calibration.get("entrance_dest_exits", []),
            "platform_level": str(calibration.get("platform_level", "-1")),
            "platform_exit": calibration.get("platform_exit", ""),
            "train_door_counts": {
                str(platform): len(point.get("door_points", []))
                for platform, point in spawn_points.items()
                if str(platform).isdigit() and point.get("door_points")
            },
            "train_alighting_duration_s": calibration.get(
                "train_alighting_duration_s", 12.0
            ),
        }
        seed = int(calibration.get("seed", 0))
        schedule = build_arrival_schedule(intervals, timetable, spawn_cfg, seed=seed)

        controller = RuntimeSpawnController(
            schedule,
            spawn_points,
            seed=seed,
            jitter_m=float(calibration.get("spawn_jitter_m", 0.5)),
            train_door_jitter_m=float(calibration.get("train_door_jitter_m", 0.3)),
            walking_speed_mean=float(calibration.get("walking_speed_mean", 1.34)),
            walking_speed_std=float(calibration.get("walking_speed_std", 0.0)),
            walking_speed_min=float(calibration.get("walking_speed_min", 0.3)),
            walking_speed_max=float(calibration.get("walking_speed_max", 2.2)),
            knowledge_profile=calibration.get("knowledge_profile", "novice"),
        )
        # Retained for the end-of-run calibration report (expected vs realised).
        controller.expected_intervals = intervals

        logger.info(
            "Calibration enabled: %d scheduled arrivals "
            "(%d entrance intervals, %d trains); seed=%d",
            len(schedule), len(intervals), len(timetable), seed,
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
    def _load_events(runner: HybridSimulationRunner, config: dict) -> None:
        """
        Load events from configuration into the runner.

        Args:
            runner: Simulation runner instance
            config: Configuration dictionary
        """
        events_config = config.get("events", [])
        if events_config is None:
            events_config = []
        elif not isinstance(events_config, list):
            logger.warning(
                "Invalid 'events' config type %s; expected list. Ignoring events.",
                type(events_config).__name__,
            )
            events_config = []
        for event in events_config:
            # Preserve all event fields (type, pa_announcement, zone_messages, etc.)
            # so that the EventManager can handle PA announcements, zone routing, etc.
            record = {k: v for k, v in event.items() if k != "_fired"}
            record.setdefault("_fired", False)
            runner.event_manager.scheduled_events.append(record)

        if events_config:
            logger.info(f"Loaded {len(events_config)} events from configuration")
        else:
            logger.warning("No events defined in configuration")
