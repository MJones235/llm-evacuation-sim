# Function Reference: evacusim/visualization/position_history.py

- Source file: [evacusim/visualization/position_history.py](../../../evacusim/visualization/position_history.py)
- Top-level classes: 1
- Top-level functions: 0

## Module Summary

Position history tracker for video generation.

## Classes and Methods

### Class PositionHistoryTracker

- Location: [evacusim/visualization/position_history.py#L22](../../../evacusim/visualization/position_history.py#L22)
- Summary: Tracks agent positions over time for video generation.
- Method count: 6

#### __init__

- Signature: def __init__(self, save_interval=0.5, streaming_path=None)
- Location: [evacusim/visualization/position_history.py#L33](../../../evacusim/visualization/position_history.py#L33)
- Summary: Initialize position history tracker.
- Key calls: streaming_path.parent.mkdir, open, logger.info

#### should_save

- Signature: def should_save(self, current_time) -> bool
- Location: [evacusim/visualization/position_history.py#L56](../../../evacusim/visualization/position_history.py#L56)
- Summary: Check if positions should be saved at current time.

#### save_frame

- Signature: def save_frame(self, current_time, agent_positions, agent_decisions, blocked_exits, active_train_exits=None) -> None
- Location: [evacusim/visualization/position_history.py#L68](../../../evacusim/visualization/position_history.py#L68)
- Summary: Save a frame of agent positions and state.
- Key calls: agent_decisions.items, self.should_save, dict, list, self._stream_file.write, self.position_history.append, latest.get, translated.get, json.dumps

#### close

- Signature: def close(self) -> None
- Location: [evacusim/visualization/position_history.py#L118](../../../evacusim/visualization/position_history.py#L118)
- Summary: Flush and close the streaming file (no-op in in-memory mode).
- Key calls: self._stream_file.flush, self._stream_file.close

#### get_history

- Signature: def get_history(self) -> list[dict[str, Any]]
- Location: [evacusim/visualization/position_history.py#L125](../../../evacusim/visualization/position_history.py#L125)
- Summary: Get complete position history.

#### save_to_file

- Signature: def save_to_file(self, output_path) -> None
- Location: [evacusim/visualization/position_history.py#L134](../../../evacusim/visualization/position_history.py#L134)
- Summary: Save position history to file.
- Key calls: self.close, output_path.parent.mkdir, logger.info, output_path.resolve, self._streaming_path.resolve, shutil.copy2, open, json.dump, f.write, len

## Coverage Notes

- Nested helper functions documented in this file: 0
