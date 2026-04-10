"""
Speed conversion utilities for agent movement.

This module provides utilities for converting human-readable speed descriptors
(e.g., "slow_walk", "jog") into realistic m/s values with natural variation.
"""

import numpy as np


def convert_speed_to_ms(speed_str: str | None) -> float | None:
    """
    Convert speed string to m/s value sampled from normal distribution.

    Args:
        speed_str: Speed descriptor (slow_walk, normal_walk, brisk_walk, jog, run)

    Returns:
        Speed in m/s sampled from appropriate distribution, or None if not specified

    Speed distributions are based on pedestrian dynamics research:
    - slow_walk: 0.5 m/s (±0.1) - seeking information, cautious movement
    - normal_walk: 1.0 m/s (±0.2) - regular walking pace
    - brisk_walk: 1.5 m/s (±0.25) - hurried but not panicked
    - jog: 2.0 m/s (±0.3) - light running
    - run: 2.5 m/s (±0.35) - fast evacuation pace
    """
    if not speed_str:
        return None

    # Define mean and std for each speed category
    # Based on pedestrian dynamics research, with variation for different urgency levels
    speed_distributions = {
        "slow_walk": {"mean": 0.5, "std": 0.10, "min": 0.3, "max": 0.8},
        "normal_walk": {"mean": 1.0, "std": 0.20, "min": 0.6, "max": 1.4},
        "brisk_walk": {"mean": 1.5, "std": 0.25, "min": 1.0, "max": 2.0},
        "jog": {"mean": 2.0, "std": 0.30, "min": 1.5, "max": 2.5},
        "run": {"mean": 2.5, "std": 0.35, "min": 2.0, "max": 3.0},
    }

    dist = speed_distributions.get(speed_str.lower())
    if not dist:
        return None

    # Sample from normal distribution and clip to min/max
    speed = np.random.normal(dist["mean"], dist["std"])
    return max(dist["min"], min(dist["max"], speed))
