"""
Walking speed utilities for pedestrian simulations.
"""

import numpy as np


def sample_walking_speed(
    mean: float = 1.34, std: float = 0.37, min_speed: float = 0.5, max_speed: float = 2.5
) -> float:
    """
    Sample walking speed from normal distribution.

    Based on pedestrian dynamics research (Weidmann, 1993).
    Default values: mean=1.34 m/s, std=0.37 m/s

    Args:
        mean: Mean walking speed (m/s)
        std: Standard deviation (m/s)
        min_speed: Minimum allowed speed
        max_speed: Maximum allowed speed

    Returns:
        Walking speed in m/s
    """
    speed = np.random.normal(mean, std)
    return max(min_speed, min(max_speed, speed))


if __name__ == "__main__":
    # Test walking speed distribution
    print("Testing walking speed distribution:")
    speeds = [sample_walking_speed() for _ in range(1000)]
    print(f"  Mean: {np.mean(speeds):.3f} m/s")
    print(f"  Std: {np.std(speeds):.3f} m/s")
    print(f"  Min: {np.min(speeds):.3f} m/s")
    print(f"  Max: {np.max(speeds):.3f} m/s")
