# Run outputs

Each run writes one directory, `<output.directory>/run_<YYYYmmdd_HHMMSS>_<engine>_s<seed>/`
(with a `_2`, `_3`, … suffix if two runs start in the same second). Times are
simulated seconds: from the run's start, or seconds after midnight for runs
that start at a time of day (`simulation.start_time_s`, calibration runs).

| File | What it holds |
|---|---|
| `manifest.json` | What produced the run and how it ended. Start here. |
| `decisions.csv` | One row per decision. |
| `exit_log.csv` | One row per person who left the station or boarded a train. |
| `population_timeseries.csv` / `.json` / `.png` | People per monitored zone over time (`monitoring`). |
| `escalator_log.csv`, `escalators.json` | One row per escalator ride; escalator geometry. |
| `agent_decisions_history.jsonl` | Position frames every 0.5 s (video and frame-based analysis). |
| `agent_decisions.json` | Full results: every decision record, events, messages, telemetry. |
| `agent_decisions_positions.json` | Latest positions, refreshed during the run for the live viewer. |
| `calibration_arrivals.csv`, `calibration_report.json` | Calibration runs: realised arrivals; expected vs realised. |
| `llm_prompt_log.jsonl` | LLM runs: every prompt and response. |
| `performance_report.txt`, `financial_report.txt` | Wall-clock profile; LLM token use and cost. |
| `route_changes.txt`, `wait_behavior.txt`, `message_analytics.txt` | Readable summaries. |
| `simulation.log` | The run's log. |
| `*.mp4` | The video, unless `--no-video`. |

## `manifest.json`

Written when the run starts and completed when it ends
(`evacusim.metrics.manifest`).

| Key | Meaning |
|---|---|
| `run_id` | The directory name. |
| `status` | `running` (still running, or killed without a chance to update), `ok`, `failed`, or `interrupted` (stopped with Ctrl-C). |
| `error` | The exception, for a failed run. |
| `started`, `finished`, `wall_time_s` | Wall-clock times (UTC). |
| `config_path`, `command` | The configuration file and the full command line. |
| `engine`, `seed` | Decision engine and master random seed. |
| `code.evacusim`, `code.study` | `commit`, `branch` and `dirty` for the engine and the study repository. `dirty: true` means there were uncommitted changes, so the commit alone does not reproduce the run. |
| `versions` | Python, evacusim, JuPedSim and pydantic versions. |
| `parameters` | Every resolved parameter, defaults included (see [parameters.md](parameters.md)). |
| `results` | `steps`, `sim_time_s`, `decisions_made`, `events_triggered`. |
| `llm` | LLM runs: requests, tokens and estimated cost. |

## `decisions.csv`

One row per decision, ordered by time then agent.

| Column | Meaning |
|---|---|
| `time_s` | When the decision was made. |
| `agent_id` | Who decided. |
| `stage` | Rule-based engine: `unaware`, `aware` or `evacuating`. Empty for the LLM engine. |
| `action` | `continue_activity`, `wait`, `seek_information`, `evacuate` or `leave_by_train`. |
| `wait_reason` | For `wait`: `awaiting_information`, `awaiting_instruction` or `route_blocked`. |
| `exit_id` | For `evacuate`: the exit chosen (a street exit, an escalator, or a train). |
| `pace` | `normal_pace`, `hurrying` or `running`; empty when waiting. |
| `reassess_when` | `next_interval`, or `new_cue_only` (decide again only when something changes). |
| `zone` | The agent's zone when deciding. |
| `cues` | Changes since the agent's last decision, `|`-separated (e.g. `alarm_state_change|zone_entry`). |
| `repair_status` | LLM: `ok`, `repair_1`, `repair_2` (answered after one or two corrections), `fallback` (no valid answer), `cached` (previous decision reused). Rule engine: `rule_based`. |
| `llm_called` | Whether a language-model request was made for this decision. |
| `prompt_hash` | SHA-256 of the prompt (LLM engine); identical prompts share a hash. |
| `route_changed_to` | The new exit, when the decision changed the agent's exit. |
| `rationale` | Why: the LLM's `option_chosen_because`, or the rule that fired. |

## `exit_log.csv`

| Column | Meaning |
|---|---|
| `agent_id` | Who left. |
| `exit_name` | The exit actually used, resolved from position (or the train boarded). |
| `intended_exit` | The exit the agent was heading for. |
| `exit_distance_m` | Distance from the exit when removed. |
| `time_s`, `level`, `x`, `y` | When and where. |
| `validated` | Whether the agent was near the exit it was assigned. |
| `spawn_source`, `spawn_location`, `spawn_time_s` | Calibration runs: how, where and when the person arrived (`entrance` or `train`). |

## `population_timeseries.csv`

`sim_time_s`, `sim_time_min`, then one column per zone in `monitoring.zones`
(e.g. `left_station`, `concourse`, `escalator_queue`, `on_escalator`,
`platform`), sampled every `monitoring.interval_seconds`.

## `escalator_log.csv`

| Column | Meaning |
|---|---|
| `agent_id`, `escalator`, `direction`, `lane` | Who rode which escalator, in which lane (`stand` or `walk`). |
| `chose_s` | When the agent chose the escalator. |
| `queue_join_s`, `board_s`, `alight_s` | Joined the queue, stepped on, stepped off. |
| `ride_s` | Time on the belt. |
| `stall_wait_s` | Time the belt was paused with the agent on it. |
| `discharge_attempts` | Tries needed to step off onto a free landing. |

## `agent_decisions_history.jsonl`

One JSON object per line, every 0.5 s of simulated time: `time`, `positions`
(agent → `[x, y]`), `agent_states`, `agent_levels`, `blocked_exits`,
`active_train_exits`, `escalators` (riders and queues).
