# Function Reference: evacusim/setup/output_manager.py

- Source file: [evacusim/setup/output_manager.py](../../../evacusim/setup/output_manager.py)
- Top-level classes: 1
- Top-level functions: 0

## Module Summary

Output directory management for Station Concordia simulations.

## Classes and Methods

### Class OutputManager

- Location: [evacusim/setup/output_manager.py#L19](../../../evacusim/setup/output_manager.py#L19)
- Summary: Handles output directory and file management for simulation runs.
- Method count: 1

#### setup_output_directory

- Signature: def setup_output_directory(config) -> tuple[str, Path, Path]
- Location: [evacusim/setup/output_manager.py#L23](../../../evacusim/setup/output_manager.py#L23)
- Summary: Setup output directory structure for a simulation run.
- Key calls: strftime, config.get, Path, output_dir.mkdir, str, logger.info, output_config.get, datetime.now

## Coverage Notes

- Nested helper functions documented in this file: 0
