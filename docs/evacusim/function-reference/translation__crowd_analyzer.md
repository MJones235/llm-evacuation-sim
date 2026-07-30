# Function Reference: evacusim/translation/crowd_analyzer.py

- Source file: [evacusim/translation/crowd_analyzer.py](../../../evacusim/translation/crowd_analyzer.py)
- Top-level classes: 1
- Top-level functions: 0

## Module Summary

Crowd behavior analysis for observation generation.

## Classes and Methods

### Class CrowdAnalyzer

- Location: [evacusim/translation/crowd_analyzer.py#L14](../../../evacusim/translation/crowd_analyzer.py#L14)
- Summary: Analyzes crowd behavior and movement patterns.
- Method count: 7

#### _classify_destination

- Signature: def _classify_destination(target_exit) -> str | None
- Location: [evacusim/translation/crowd_analyzer.py#L29](../../../evacusim/translation/crowd_analyzer.py#L29)
- Summary: Classify a raw destination key into a human-readable category.
- Key calls: target_exit.lower, tl.startswith, title, tl.split, target_exit.replace

#### __init__

- Signature: def __init__(self, exits)
- Location: [evacusim/translation/crowd_analyzer.py#L51](../../../evacusim/translation/crowd_analyzer.py#L51)
- Summary: Initialize crowd analyzer.

#### summarize_behaviors

- Signature: def summarize_behaviors(self, nearby_agents, agent_injured) -> str
- Location: [evacusim/translation/crowd_analyzer.py#L60](../../../evacusim/translation/crowd_analyzer.py#L60)
- Summary: Summarize what nearby agents are doing using three-dimensional model.
- Key calls: len, join, agent.get, CrowdAnalyzer._classify_destination, max, a.get, parts.append, injured_nearby.append, dest_counts.get, sorted

#### count_agents_per_exit

- Signature: def count_agents_per_exit(self, nearby_agents) -> dict[str, int]
- Location: [evacusim/translation/crowd_analyzer.py#L136](../../../evacusim/translation/crowd_analyzer.py#L136)
- Summary: Count how many nearby agents appear to be heading toward each exit.
- Key calls: agent.get, exit_counts.get

#### categorize_density

- Signature: def categorize_density(num_nearby) -> str
- Location: [evacusim/translation/crowd_analyzer.py#L156](../../../evacusim/translation/crowd_analyzer.py#L156)
- Summary: Categorize crowd density.

#### categorize_count

- Signature: def categorize_count(count) -> str
- Location: [evacusim/translation/crowd_analyzer.py#L176](../../../evacusim/translation/crowd_analyzer.py#L176)
- Summary: Categorize people count to prevent minor changes from triggering LLM.

#### analyze_movement_pattern

- Signature: def analyze_movement_pattern(nearby_agents) -> str
- Location: [evacusim/translation/crowd_analyzer.py#L196](../../../evacusim/translation/crowd_analyzer.py#L196)
- Summary: Analyze overall movement pattern of nearby agents.
- Key calls: sum, agent.get, len, CrowdAnalyzer._classify_destination, max, sorted, a.get, join, dest_counts.get

## Coverage Notes

- Nested helper functions documented in this file: 0
