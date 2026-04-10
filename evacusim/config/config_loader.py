"""
Configuration loading and validation for Station Concordia simulations.

This module is responsible for:
- Loading configuration from YAML files
- Applying command-line overrides
- Validating configuration structure
"""

from pathlib import Path
from typing import Any

import yaml

from evacusim.utils.logger import get_logger

logger = get_logger(__name__)


class ConfigLoader:
    """Handles loading and validation of simulation configuration."""

    @staticmethod
    def load_and_validate(
        config_path: str,
        agents: int | None = None,
        max_steps: int | None = None,
        output_dir: str | None = None,
    ) -> dict[str, Any]:
        """
        Load, validate, and apply overrides to configuration in one step.

        This is the primary entry point for configuration loading.

        Args:
            config_path: Path to the YAML configuration file
            agents: Number of agents (overrides config if provided)
            max_steps: Maximum simulation steps (overrides config if provided)
            output_dir: Output directory (overrides config if provided)

        Returns:
            Dictionary containing the loaded and validated configuration

        Raises:
            FileNotFoundError: If the configuration file doesn't exist
            yaml.YAMLError: If the YAML file is malformed
            ValueError: If required configuration is missing
        """
        # Load base configuration
        config = ConfigLoader.load_config(config_path)

        # Apply CLI overrides
        config = ConfigLoader.apply_cli_overrides(config, agents, max_steps, output_dir)

        # Validate
        ConfigLoader.validate_config(config)

        return config

    @staticmethod
    def load_config(config_path: str) -> dict[str, Any]:
        """
        Load configuration from YAML file, resolving any ``extends`` inheritance.

        If the YAML file contains a top-level ``extends`` key pointing to another
        YAML file (relative to the config file's own directory), the base config is
        loaded first and then deep-merged with the child config so that child values
        take precedence.  The ``extends`` key is removed from the returned dict.

        Args:
            config_path: Path to the YAML configuration file

        Returns:
            Dictionary containing the loaded (and merged) configuration

        Raises:
            FileNotFoundError: If the configuration file doesn't exist
            yaml.YAMLError: If the YAML file is malformed
        """
        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(config_file) as f:
            config = yaml.safe_load(f) or {}

        # Resolve optional inheritance
        extends = config.pop("extends", None)
        if extends:
            base_path = (config_file.parent / extends).resolve()
            base_config = ConfigLoader.load_config(str(base_path))
            config = ConfigLoader._deep_merge(base_config, config)
            logger.info(f"Merged {config_path} on top of {base_path}")
        else:
            logger.info(f"Loaded configuration from {config_path}")

        return config

    @staticmethod
    def _deep_merge(base: dict, override: dict) -> dict:
        """Recursively merge *override* into *base*, returning a new dict."""
        result = dict(base)
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = ConfigLoader._deep_merge(result[key], value)
            else:
                result[key] = value
        return result

    @staticmethod
    def apply_cli_overrides(
        config: dict[str, Any],
        agents: int | None = None,
        max_steps: int | None = None,
        output_dir: str | None = None,
    ) -> dict[str, Any]:
        """
        Apply command-line argument overrides to configuration.

        Args:
            config: Base configuration dictionary
            agents: Number of agents (overrides config if provided)
            max_steps: Maximum simulation steps (overrides config if provided)
            output_dir: Output directory (overrides config if provided)

        Returns:
            Modified configuration dictionary
        """
        if agents is not None:
            config.setdefault("agents", {})["count"] = agents
            logger.info(f"Override: agents count = {agents}")

        if max_steps is not None:
            config.setdefault("simulation", {})["max_iterations"] = max_steps
            logger.info(f"Override: max_iterations = {max_steps}")

        if output_dir is not None:
            config.setdefault("output", {})["directory"] = output_dir
            logger.info(f"Override: output directory = {output_dir}")

        return config

    @staticmethod
    def validate_config(config: dict[str, Any]) -> None:
        """
        Validate that required configuration sections exist.

        Args:
            config: Configuration dictionary to validate

        Raises:
            ValueError: If required configuration is missing
        """
        required_sections = ["agents", "simulation"]
        for section in required_sections:
            if section not in config:
                raise ValueError(f"Required configuration section '{section}' is missing")

        # Validate agents section
        agents_config = config["agents"]
        if "count" not in agents_config:
            raise ValueError("agents.count is required in configuration")

        ConfigLoader._validate_knowledge_profiles(config)

        # Validate spawn_schedule (optional)
        if "spawn_schedule" in agents_config:
            schedule = agents_config["spawn_schedule"]
            if isinstance(schedule, dict) and schedule.get("enabled", False):
                if not isinstance(agents_config.get("spawn_schedule"), dict):
                    raise ValueError("agents.spawn_schedule must be a dictionary when enabled")
                logger.info("Spawn schedule enabled: agents will be spawned according to schedule")
            elif isinstance(schedule, list):
                # List format for backwards compatibility
                logger.info(f"Spawn schedule configured with {len(schedule)} spawn events")

        # Validate simulation section
        sim_config = config["simulation"]
        if "network_path" not in sim_config:
            raise ValueError("simulation.network_path is required in configuration")

        # Validate multi_level (optional)
        if "multi_level" in sim_config:
            multi_level = sim_config["multi_level"]
            if isinstance(multi_level, bool) and multi_level:
                # Boolean format: just enable/disable
                levels = sim_config.get("levels", ["0", "-1"])
                logger.info(f"Multi-level simulation enabled: {levels}")
            elif isinstance(multi_level, dict):
                # Dictionary format (legacy support)
                if multi_level.get("enabled", False):
                    if "levels" not in multi_level:
                        raise ValueError("simulation.multi_level.levels required when enabled")
                    logger.info(f"Multi-level simulation enabled: {multi_level['levels']}")

        logger.debug("Configuration validation passed")

    @staticmethod
    def _validate_knowledge_profiles(config: dict[str, Any]) -> None:
        """Validate required static environment knowledge schema."""
        station_config = config.get("station")
        if not isinstance(station_config, dict):
            raise ValueError("station section is required and must be a dictionary")

        knowledge_config = station_config.get("knowledge")
        if not isinstance(knowledge_config, dict):
            raise ValueError("station.knowledge is required and must be a dictionary")

        base_memories = knowledge_config.get("base_memories")
        if not isinstance(base_memories, list) or not base_memories:
            raise ValueError("station.knowledge.base_memories must be a non-empty list")
        for memory in base_memories:
            if not isinstance(memory, str) or not memory.strip():
                raise ValueError("station.knowledge.base_memories must contain non-empty strings")

        profile_memories = knowledge_config.get("profiles")
        if not isinstance(profile_memories, dict) or not profile_memories:
            raise ValueError("station.knowledge.profiles must be a non-empty dictionary")

        for profile_name, memories in profile_memories.items():
            if not isinstance(profile_name, str) or not profile_name.strip():
                raise ValueError("station.knowledge.profiles keys must be non-empty strings")
            if not isinstance(memories, list) or not memories:
                raise ValueError(
                    f"station.knowledge.profiles.{profile_name} must be a non-empty list"
                )
            for memory in memories:
                if not isinstance(memory, str) or not memory.strip():
                    raise ValueError(
                        f"station.knowledge.profiles.{profile_name} must contain non-empty strings"
                    )

        location_memories = knowledge_config.get("location_memories", [])
        if not isinstance(location_memories, list):
            raise ValueError("station.knowledge.location_memories must be a list when provided")

        for idx, rule in enumerate(location_memories):
            if not isinstance(rule, dict):
                raise ValueError(f"station.knowledge.location_memories[{idx}] must be a dictionary")

            condition = rule.get("when", {})
            if not isinstance(condition, dict):
                raise ValueError(
                    f"station.knowledge.location_memories[{idx}].when must be a dictionary"
                )

            for key in ("profiles", "level_ids", "zones"):
                value = condition.get(key)
                if value is None:
                    continue
                if not isinstance(value, list) or not all(
                    isinstance(item, str | int) for item in value
                ):
                    raise ValueError(
                        f"station.knowledge.location_memories[{idx}].when.{key} must be a list of strings"
                    )

            memories = rule.get("memories")
            if not isinstance(memories, list) or not memories:
                raise ValueError(
                    f"station.knowledge.location_memories[{idx}].memories must be a non-empty list"
                )
            for memory in memories:
                if not isinstance(memory, str) or not memory.strip():
                    raise ValueError(
                        f"station.knowledge.location_memories[{idx}].memories must contain non-empty strings"
                    )

        agents_profiles = config["agents"].get("knowledge_profiles")
        if not isinstance(agents_profiles, dict) or not agents_profiles:
            raise ValueError("agents.knowledge_profiles must be a non-empty dictionary")

        total_weight = 0.0
        for profile_name, weight in agents_profiles.items():
            if profile_name not in profile_memories:
                raise ValueError(
                    f"agents.knowledge_profiles contains unknown profile '{profile_name}'"
                )
            if not isinstance(weight, int | float) or weight <= 0:
                raise ValueError(
                    f"agents.knowledge_profiles.{profile_name} must be a positive number"
                )
            total_weight += float(weight)

        if total_weight <= 0:
            raise ValueError("agents.knowledge_profiles weights must sum to a positive value")
