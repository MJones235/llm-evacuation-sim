# Function Reference: evacusim/utils/logger.py

- Source file: [evacusim/utils/logger.py](../../../evacusim/utils/logger.py)
- Top-level classes: 0
- Top-level functions: 2

## Module Summary

Logging configuration for JuPedSim station simulation.

## Top-Level Functions

### setup_logger

- Signature: def setup_logger(name='station_jupedsim', log_file=None, console_level=logging.INFO, file_level=logging.DEBUG) -> logging.Logger
- Location: [evacusim/utils/logger.py#L16](../../../evacusim/utils/logger.py#L16)
- Summary: Setup and configure logger with console and optional file handlers.
- Key calls: logging.getLogger, root_logger.setLevel, root_logger.handlers.clear, logger.setLevel, logger.handlers.clear, logging.StreamHandler, console_handler.setLevel, logging.Formatter, console_handler.setFormatter, logger.addHandler

### get_logger

- Signature: def get_logger(name='station_jupedsim') -> logging.Logger
- Location: [evacusim/utils/logger.py#L71](../../../evacusim/utils/logger.py#L71)
- Summary: Get existing logger instance.
- Key calls: logging.getLogger

## Coverage Notes

- Nested helper functions documented in this file: 0
