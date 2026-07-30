# Function Reference: evacusim/config/config_loader.py

- Source file: [evacusim/config/config_loader.py](../../../evacusim/config/config_loader.py)
- Top-level classes: 1
- Top-level functions: 0

## Module Summary

Configuration loading and validation for Station Concordia simulations.

## Classes and Methods

### Class ConfigLoader

- Location: [evacusim/config/config_loader.py#L20](../../../evacusim/config/config_loader.py#L20)
- Summary: Handles loading and validation of simulation configuration.
- Method count: 6

#### load_and_validate

- Signature: def load_and_validate(config_path, agents=None, max_steps=None, output_dir=None) -> dict[str, Any]
- Location: [evacusim/config/config_loader.py#L24](../../../evacusim/config/config_loader.py#L24)
- Summary: Load, validate, and apply overrides to configuration in one step.
- Key calls: ConfigLoader.load_config, ConfigLoader.apply_cli_overrides, ConfigLoader.validate_config

#### load_config

- Signature: def load_config(config_path) -> dict[str, Any]
- Location: [evacusim/config/config_loader.py#L61](../../../evacusim/config/config_loader.py#L61)
- Summary: Load configuration from YAML file, resolving any ``extends`` inheritance.
- Key calls: Path, config.pop, config_file.exists, FileNotFoundError, open, resolve, ConfigLoader.load_config, ConfigLoader._deep_merge, logger.info, yaml.safe_load

#### _deep_merge

- Signature: def _deep_merge(base, override) -> dict
- Location: [evacusim/config/config_loader.py#L100](../../../evacusim/config/config_loader.py#L100)
- Summary: Recursively merge *override* into *base*, returning a new dict.
- Key calls: dict, override.items, isinstance, ConfigLoader._deep_merge

#### apply_cli_overrides

- Signature: def apply_cli_overrides(config, agents=None, max_steps=None, output_dir=None) -> dict[str, Any]
- Location: [evacusim/config/config_loader.py#L111](../../../evacusim/config/config_loader.py#L111)
- Summary: Apply command-line argument overrides to configuration.
- Key calls: logger.info, config.setdefault

#### validate_config

- Signature: def validate_config(config) -> None
- Location: [evacusim/config/config_loader.py#L144](../../../evacusim/config/config_loader.py#L144)
- Summary: Validate that required configuration sections exist.
- Key calls: ConfigLoader._validate_knowledge_profiles, logger.debug, ValueError, isinstance, schedule.get, logger.info, sim_config.get, multi_level.get, agents_config.get, len

#### _validate_knowledge_profiles

- Signature: def _validate_knowledge_profiles(config) -> None
- Location: [evacusim/config/config_loader.py#L199](../../../evacusim/config/config_loader.py#L199)
- Summary: Validate required static environment knowledge schema.
- Key calls: config.get, station_config.get, knowledge_config.get, profile_memories.items, enumerate, get, agents_profiles.items, isinstance, ValueError, rule.get

## Coverage Notes

- Nested helper functions documented in this file: 0
