"""
Logging configuration for JuPedSim station simulation.

Provides centralized logging setup with multiple handlers:
- Console output for important messages
- File output for detailed debugging
- Configurable log levels
"""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str = "station_jupedsim",
    log_file: Optional[Path] = None,
    console_level: int = logging.INFO,
    file_level: int = logging.DEBUG,
) -> logging.Logger:
    """
    Setup and configure logger with console and optional file handlers.

    Configures BOTH the named logger AND the root logger so that all child
    loggers (e.g., scenarios.common.llm.azure_provider) inherit the file handler.

    Args:
        name: Logger name
        log_file: Optional path to log file
        console_level: Logging level for console output (default: INFO)
        file_level: Logging level for file output (default: DEBUG)

    Returns:
        Configured logger instance
    """
    # Configure root logger to capture all logs
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.handlers.clear()

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)  # Capture everything, handlers will filter

    # Clear any existing handlers
    logger.handlers.clear()

    # Console handler - INFO and above
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level)
    console_formatter = logging.Formatter("%(levelname)s: %(message)s")
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    root_logger.addHandler(console_handler)

    # File handler - DEBUG and above (if log file specified)
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, mode="w")
        file_handler.setLevel(file_level)
        file_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        root_logger.addHandler(file_handler)

    return logger


def get_logger(name: str = "station_jupedsim") -> logging.Logger:
    """
    Get existing logger instance.

    Args:
        name: Logger name

    Returns:
        Logger instance
    """
    return logging.getLogger(name)
