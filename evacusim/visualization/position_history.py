"""
Position history tracker for video generation.

Stores agent positions at regular intervals during simulation
to enable smooth video playback without LLM response delays.

Streaming mode: when an output_path is provided at construction time, frames
are appended immediately to a newline-delimited JSON file (.jsonl) rather than
accumulated in memory.  This keeps RAM usage constant for long simulations and
avoids losing history on crashes.
"""

import json
from pathlib import Path
from typing import Any

from evacusim.utils.logger import get_logger

logger = get_logger(__name__)


class PositionHistoryTracker:
    """
    Tracks agent positions over time for video generation.

    Stores positions at regular intervals (e.g., every 0.5s) to create
    smooth videos that show simulation progression at constant speed.

    If *streaming_path* is provided, frames are written directly to that
    ``.jsonl`` file instead of being buffered in memory.
    """

    def __init__(self, save_interval: float = 0.5, streaming_path: Path | None = None):
        """
        Initialize position history tracker.

        Args:
            save_interval: Time interval between position saves (seconds)
            streaming_path: Optional path for streaming writes.  When set, every
                frame is appended to the file immediately and ``position_history``
                is kept empty to avoid memory growth.  If None, behaviour is the
                same as before: accumulate in memory and write at end.
        """
        self.save_interval = save_interval
        self.last_save_time = -save_interval  # Save immediately on first call
        self.position_history: list[dict[str, Any]] = []

        self._streaming_path = streaming_path
        self._stream_file = None
        if streaming_path is not None:
            streaming_path.parent.mkdir(parents=True, exist_ok=True)
            # Open for appending so partial runs can be resumed if needed.
            self._stream_file = open(streaming_path, "a", buffering=1)  # line-buffered
            logger.info(f"Streaming position history to {streaming_path}")

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

        if self._stream_file is not None:
            # Streaming mode: write one JSON line per frame, no in-memory accumulation.
            self._stream_file.write(json.dumps(frame) + "\n")
        else:
            # In-memory mode (legacy behaviour).
            self.position_history.append(frame)

        self.last_save_time = current_time

    def close(self) -> None:
        """Flush and close the streaming file (no-op in in-memory mode)."""
        if self._stream_file is not None:
            self._stream_file.flush()
            self._stream_file.close()
            self._stream_file = None

    def get_history(self) -> list[dict[str, Any]]:
        """
        Get complete position history.

        Returns:
            List of frame dicts with time, positions, and states
        """
        return self.position_history

    def save_to_file(self, output_path: Path) -> None:
        """
        Save position history to file.

        In streaming mode the data is already on disk; this method closes the
        stream file and, if *output_path* differs from the streaming path, copies
        it, otherwise it is a no-op (the data is already at the right location).

        In in-memory mode it behaves as before: write the full history to a JSON
        file (legacy format, wrapped in ``{"position_history": [...]}``) unless
        the extension is ``.jsonl``, in which case each frame is written as one
        line.

        Args:
            output_path: Destination file path
        """
        self.close()

        if self._streaming_path is not None:
            # Streaming mode: data already written; move/copy to requested path.
            if output_path.resolve() != self._streaming_path.resolve():
                import shutil
                shutil.copy2(self._streaming_path, output_path)
            logger.info(
                f"Position history stream closed: {len(self.position_history)} in-memory "
                f"frames + streamed data at {output_path}"
            )
            return

        # In-memory mode: write to file.
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if output_path.suffix == ".jsonl":
            with open(output_path, "w") as f:
                for frame in self.position_history:
                    f.write(json.dumps(frame) + "\n")
        else:
            data = {"position_history": self.position_history}
            with open(output_path, "w") as f:
                json.dump(data, f, indent=2)

        logger.info(f"Saved {len(self.position_history)} position frames to {output_path}")
