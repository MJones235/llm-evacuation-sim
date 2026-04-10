"""
Position history tracker for video generation.

Stores agent positions at regular intervals during simulation
to enable smooth video playback without LLM response delays.
"""

from pathlib import Path
from typing import Any

from evacusim.utils.logger import get_logger

logger = get_logger(__name__)


class PositionHistoryTracker:
    """
    Tracks agent positions over time for video generation.

    Stores positions at regular intervals (e.g., every 0.5s) to create
    smooth videos that show simulation progression at constant speed.
    """

    def __init__(self, save_interval: float = 0.5):
        """
        Initialize position history tracker.

        Args:
            save_interval: Time interval between position saves (seconds)
        """
        self.save_interval = save_interval
        self.last_save_time = -save_interval  # Save immediately on first call
        self.position_history: list[dict[str, Any]] = []

    def should_save(self, current_time: float) -> bool:
        """
        Check if positions should be saved at current time.

        Args:
            current_time: Current simulation time (seconds)

        Returns:
            True if it's time to save positions
        """
        return (current_time - self.last_save_time) >= self.save_interval

    def save_frame(
        self,
        current_time: float,
        agent_positions: dict[str, tuple[float, float]],
        agent_decisions: dict[str, Any],
        blocked_exits: set[str],
    ) -> None:
        """
        Save a frame of agent positions and state.

        Args:
            current_time: Current simulation time
            agent_positions: Current agent positions
            agent_decisions: Agent decision history
            blocked_exits: Currently blocked exits
        """
        if not self.should_save(current_time):
            return

        # Extract current agent states from decisions
        agent_states = {}
        for agent_id, data in agent_decisions.items():
            if "decisions" in data and data["decisions"]:
                latest = data["decisions"][-1]
                translated = latest.get("translated", {})
                agent_states[agent_id] = {
                    "action_type": translated.get("action_type", "unknown"),
                    "wait_reason": translated.get("wait_reason"),
                    "destination": translated.get("destination"),
                }

        frame = {
            "time": current_time,
            "positions": dict(agent_positions),  # Copy to avoid mutation
            "agent_states": agent_states,
            "blocked_exits": list(blocked_exits),
        }

        self.position_history.append(frame)
        self.last_save_time = current_time

    def get_history(self) -> list[dict[str, Any]]:
        """
        Get complete position history.

        Returns:
            List of frame dicts with time, positions, and states
        """
        return self.position_history

    def save_to_file(self, output_path: Path) -> None:
        """
        Save position history to JSON file.

        Args:
            output_path: Path to output file
        """
        import json

        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save position history separate from main results for efficiency
        data = {"position_history": self.position_history}

        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Saved {len(self.position_history)} position frames to {output_path}")
