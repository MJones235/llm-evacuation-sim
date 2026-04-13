"""
Simulation runner factory for Station Concordia simulations.

This module is responsible for:
- Creating and configuring HybridSimulationRunner instances
- Loading events from configuration
- Setting up test scenarios (blocked exits, etc.)
- Configuring runner parameters
"""

from pathlib import Path

from evacusim.utils.logger import get_logger
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

        # Video generation settings
        video_config = config.get("video", {})
        enable_video = video_config.get("enabled", False)

        # Monitoring settings
        monitoring_config = config.get("monitoring", {})

        logger.info("Creating HybridSimulationRunner...")

        try:
            runner = HybridSimulationRunner(
                jupedsim_simulation=jps_sim,
                agents_config=agents_config,
                station_layout=station_layout,
                language_model=model,
                embedder=embedder,
                decision_interval=decision_interval,
                max_steps=max_steps,
                output_file=decisions_file,
                enable_video=enable_video,
                monitoring_config=monitoring_config,
            )
            logger.info("HybridSimulationRunner initialized")
        except Exception as e:
            logger.error(f"FATAL ERROR during HybridSimulationRunner initialization: {e}")
            import traceback

            traceback.print_exc()
            raise

        # Configure events
        SimulationRunnerFactory._load_events(runner, config)

        return runner

    @staticmethod
    def _load_events(runner: HybridSimulationRunner, config: dict) -> None:
        """
        Load events from configuration into the runner.

        Args:
            runner: Simulation runner instance
            config: Configuration dictionary
        """
        events_config = config.get("events", [])
        for event in events_config:
            runner.event_manager.scheduled_events.append(
                {
                    "time": event.get("time", 0.0),
                    "message": event.get("message", ""),
                    "_fired": False,
                }
            )

        if events_config:
            logger.info(f"Loaded {len(events_config)} events from configuration")
        else:
            logger.warning("No events defined in configuration")
