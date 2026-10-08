"""Load a run's YAML configuration and validate it against the parameter schema.

Flow::

    YAML file ──load_config──▶ merged dict ──apply_cli_overrides──▶ dict
        (follows ``extends:``)                                        │
                                                                      ▼
                                            validate_config ──▶ RunConfig (typed)

The parameters themselves (names, types, defaults, meaning) are defined once,
in :mod:`evacusim.config.schema`.
"""

from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from evacusim.config.schema import RunConfig
from evacusim.utils.logger import get_logger

logger = get_logger(__name__)


class ConfigError(ValueError):
    """A configuration file does not match the parameter schema."""


class ConfigLoader:
    """Loads, overrides and validates run configuration files."""

    @staticmethod
    def load_and_validate(
        config_path: str,
        agents: int | None = None,
        max_steps: int | None = None,
        output_dir: str | None = None,
    ) -> dict[str, Any]:
        """Load a config file, apply command-line overrides and validate it.

        Args:
            config_path: Path to the YAML configuration file.
            agents: Overrides ``agents.count``.
            max_steps: Overrides ``simulation.max_iterations``.
            output_dir: Overrides ``output.directory``.

        Returns:
            The merged configuration as a plain dict, exactly as written in the
            YAML files (defaults are *not* filled in). It is guaranteed to
            validate against :class:`~evacusim.config.schema.RunConfig`.

        Raises:
            FileNotFoundError: The file (or a file it extends) does not exist.
            ConfigError: The configuration does not match the schema.
        """
        config = ConfigLoader.load_config(config_path)
        config = ConfigLoader.apply_cli_overrides(config, agents, max_steps, output_dir)
        ConfigLoader.validate_config(config)
        return config

    @staticmethod
    def load_config(config_path: str) -> dict[str, Any]:
        """Read a YAML config file, resolving ``extends:`` inheritance.

        A top-level ``extends: <path>`` (relative to the file's directory)
        names a base file. The base is loaded first, then this file is
        deep-merged on top so its values win. An explicitly empty mapping
        (e.g. ``systems: {}``) clears the inherited one.

        Raises:
            FileNotFoundError: The file (or a file it extends) does not exist.
        """
        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(config_file) as f:
            config = yaml.safe_load(f) or {}

        extends = config.pop("extends", None)
        if extends:
            base_path = (config_file.parent / extends).resolve()
            config = ConfigLoader._deep_merge(ConfigLoader.load_config(str(base_path)), config)
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
                result[key] = ConfigLoader._deep_merge(result[key], value) if value else {}
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
        """Apply command-line overrides in place and return the config."""
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
    def validate_config(config: dict[str, Any]) -> RunConfig:
        """Validate a merged config dict against the parameter schema.

        Returns:
            The typed parameters, with every default filled in.

        Raises:
            ConfigError: Listing every problem as ``section.key: message``.
        """
        try:
            run_config = RunConfig.model_validate(config)
        except ValidationError as exc:
            problems = "\n".join(
                f"  {'.'.join(str(part) for part in err['loc']) or '(root)'}: {err['msg']}"
                for err in exc.errors()
            )
            raise ConfigError(f"Invalid configuration:\n{problems}") from None
        logger.debug("Configuration validation passed")
        return run_config
