# evacusim

A reusable framework for LLM-driven pedestrian evacuation simulation, combining:

- **[JuPedSim](https://www.jupedsim.org/)** — pedestrian movement physics
- **[Concordia](https://github.com/google-deepmind/concordia)** — LLM-based agent cognition and decision-making

## Design goals

- Venue-agnostic: adaptable to stations, shopping centres, concert venues, etc.
- Scenario-agnostic: experiment differences are expressed in YAML config, not code
- Cheap to run multiple experiments: LLM cost tracking built in
- Cleanly separable: framework has no knowledge of specific venue geometry

## Package structure

```
evacusim/
├── core/           # Abstract base classes: Scenario, agent roles, typed events
├── jps/            # JuPedSim wrappers (single-level and multi-level)
├── concordia/      # Concordia/LLM integration: agent builder, LLM setup
├── translation/    # Sim state ↔ natural language (observations, actions, spatial)
├── systems/        # Event manager, PA/messaging system, staff director
├── metrics/        # Zone population monitor, LLM cost reporter
├── coordination/   # HybridSimulationRunner and supporting processors
├── visualization/  # Viewer launcher, video helper, spatial display
└── utils/          # Logger, performance timer, misc helpers
```

## Installation

```bash
pip install -e .
```

Or from a study repo:

```bash
pip install git+https://github.com/MJones235/llm-evacuation-sim.git
```

## Usage

See the [monument-evacuation](https://github.com/your-org/monument-evacuation) study repo for
a worked example of how to use this framework for a real experiment series.

## Regression KPI Harness

Use the built-in comparison harness to evaluate run-to-run quality deltas after
decision, prompt, or transfer logic changes.

Run from the repository root:

```bash
python -m evacusim.metrics.regression_harness \
	/path/to/baseline/results.json \
	/path/to/candidate/results.json \
	--out-json /path/to/comparison.json
```

The harness reports deltas for:

- final simulation time
- decision volume and wait-loop rate
- action mix and pace mix
- route-change count
- per-zone maximum occupancy from population snapshots

For queue-density hotspot analysis, define queue-specific monitoring zones in
scenario YAML so they appear in `population_timeseries` snapshots.

### Pace Multiplier Config

Configure pace multipliers in scenario YAML:

```yaml
performance:
	pace_multipliers:
		hurrying: 1.25
		running: 1.8
```

The action executor validates these values (numeric and > 0). If a pace value
is used in decisions without a configured multiplier, execution fails fast with
an explicit error.

### Decision Telemetry in Outputs

Incremental and final results JSON now include a `decision_telemetry` block
with rollups such as:

- decision_records_total
- wait_decision_rate
- fallback_rate
- action_counts
- pace_counts
- reassess_mode_counts
- repair_status_counts
- cue_type_counts

## Prompt Customization

Prompt text is loaded from external files so you can iterate on wording without
editing Python code.

- Decision prompt default: `evacusim/decision/templates/decision_prompt.template.txt`
- Decision prompt override key: `prompts.decision_prompt_template_path`
- System prompt default: `evacusim/concordia/templates/system_prompt.txt`
- System prompt override keys: `prompts.system_prompt_path` (preferred) or `llm.system_prompt_path`
- System prompt env override: `AZURE_LLM_SYSTEM_PROMPT_PATH` (takes precedence)

System prompt fallback order:

1. `AZURE_LLM_SYSTEM_PROMPT_PATH` (if set)
2. `prompts.system_prompt_path` (or `llm.system_prompt_path`)
3. Packaged default file at `evacusim/concordia/templates/system_prompt.txt`

If neither the configured file nor the packaged default can be read (or the file
is empty), startup fails with an explicit error.

Example scenario config fragment:

```yaml
prompts:
	decision_prompt_template_path: /absolute/path/to/my_decision_prompt.template.txt
	system_prompt_path: /absolute/path/to/my_system_prompt.txt
```

Decision-prompt template variables injected at runtime:

- `$role_prefix`
- `$zone_target_type_line`
- `$zone_name_example`
- `$zone_constraint_line`
- `$zones_section`
- `$following_constraint_text`
- `$valid_exits_text`

## Config-Driven Exit Semantics

Goal-aware routing can be configured without code changes by adding optional
station-level semantics in scenario YAML.

- `station.exit_semantic_tags`: map of `exit_id -> [tags...]`
- `station.goal_semantic_policies`: ordered list of prompt-routing rules

Example:

```yaml
station:
	exit_semantic_tags:
		escalator_a_down: [to_platform, vertical_connector]
		escalator_b_down: [to_platform, vertical_connector]
		blackett_street: [to_street, outside_station]
		grainger_street: [to_street, outside_station]

	goal_semantic_policies:
		- when_goal_contains_any: [platform, train]
			applies_in_zones: [concourse, level_0]
			prefer_exit_tags: [to_platform]
			avoid_exit_tags: [to_street]
			instruction: "Escalators/connectors lead to platforms; street exits leave the station."
```

These rules inject guidance into the decision prompt and prioritize matching
visible exits, while remaining fully geometry-agnostic.

## Detailed technical documentation

Comprehensive implementation and design documentation is available in
[docs/evacusim/README.md](docs/evacusim/README.md), including:

- architecture and runtime lifecycle
- package/module responsibilities
- deep dives on escalators, speed, line-of-sight, and decision triggering
- full function reference generated from source AST across all package files
- design decisions, tradeoffs, and prioritized risk register
