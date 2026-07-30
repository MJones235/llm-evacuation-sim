# Function Reference: evacusim/visualization/view_concordia_live.py

- Source file: [evacusim/visualization/view_concordia_live.py](../../../evacusim/visualization/view_concordia_live.py)
- Top-level classes: 1
- Top-level functions: 1

## Module Summary

Real-time viewer for Station Concordia simulation progress.

## Top-Level Functions

### main

- Signature: def main()
- Location: [evacusim/visualization/view_concordia_live.py#L216](../../../evacusim/visualization/view_concordia_live.py#L216)
- Summary: Main entry point.
- Key calls: argparse.ArgumentParser, parser.add_argument, parser.parse_args, ConcordiaViewer, viewer.watch

## Classes and Methods

### Class ConcordiaViewer

- Location: [evacusim/visualization/view_concordia_live.py#L27](../../../evacusim/visualization/view_concordia_live.py#L27)
- Summary: Real-time viewer for Concordia simulation.
- Method count: 5

#### __init__

- Signature: def __init__(self, output_file)
- Location: [evacusim/visualization/view_concordia_live.py#L30](../../../evacusim/visualization/view_concordia_live.py#L30)
- Summary: Initialize viewer.
- Key calls: Path, set, Console

#### parse_concordia_action

- Signature: def parse_concordia_action(self, action) -> dict
- Location: [evacusim/visualization/view_concordia_live.py#L39](../../../evacusim/visualization/view_concordia_live.py#L39)
- Summary: Parse Concordia's multi-question action format.
- Key calls: action.split, line.strip, line.startswith, strip, line.replace

#### format_decision_rich

- Signature: def format_decision_rich(self, agent_id, decision) -> Panel
- Location: [evacusim/visualization/view_concordia_live.py#L85](../../../evacusim/visualization/view_concordia_live.py#L85)
- Summary: Format a decision using rich formatting.
- Key calls: decision.get, content.append, reasoning.get, Panel, join, translated.get

#### format_decision_simple

- Signature: def format_decision_simple(self, agent_id, decision) -> str
- Location: [evacusim/visualization/view_concordia_live.py#L135](../../../evacusim/visualization/view_concordia_live.py#L135)
- Summary: Format a decision using simple text.
- Key calls: decision.get, output.append, reasoning.get, join, translated.get

#### watch

- Signature: def watch(self)
- Location: [evacusim/visualization/view_concordia_live.py#L169](../../../evacusim/visualization/view_concordia_live.py#L169)
- Summary: Watch the output file for changes.
- Key calls: print, time.sleep, self.output_file.exists, data.get, agent_decisions.items, open, json.load, agent_data.get, self.decisions_seen.add, decision.get

## Coverage Notes

- Nested helper functions documented in this file: 0
