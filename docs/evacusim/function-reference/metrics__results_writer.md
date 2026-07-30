# Function Reference: evacusim/metrics/results_writer.py

- Source file: [evacusim/metrics/results_writer.py](../../../evacusim/metrics/results_writer.py)
- Top-level classes: 1
- Top-level functions: 0

## Module Summary

Results writing for Station Concordia simulations.

## Classes and Methods

### Class ResultsWriter

- Location: [evacusim/metrics/results_writer.py#L19](../../../evacusim/metrics/results_writer.py#L19)
- Summary: Handles writing simulation results to files.
- Method count: 3

#### save_incremental

- Signature: def save_incremental(output_file, agent_decisions, agent_positions, current_sim_time, event_history, blocked_exits, message_history, decision_interval, max_steps, num_agents, agent_levels=None) -> None
- Location: [evacusim/metrics/results_writer.py#L28](../../../evacusim/metrics/results_writer.py#L28)
- Summary: Save current results incrementally for live viewing.
- Key calls: list, output_file.parent.mkdir, output_file.with_suffix, tmp_file.replace, open, json.dump, logger.warning

#### save_positions_only

- Signature: def save_positions_only(output_file, agent_positions, current_sim_time, agent_levels=None, blocked_exits=None, agent_roles=None, active_train_exits=None) -> None
- Location: [evacusim/metrics/results_writer.py#L88](../../../evacusim/metrics/results_writer.py#L88)
- Summary: Write a lightweight positions sidecar file for the live viewer.
- Key calls: output_file.with_name, list, output_file.parent.mkdir, sidecar.with_suffix, tmp_file.replace, open, json.dump, logger.warning

#### save_final_results

- Signature: def save_final_results(output_path, agent_decisions, agent_positions, final_sim_time, event_history, blocked_exits, message_history, wait_events, decision_interval, max_steps, num_agents, performance_report, llm_provider, agent_levels=None, agent_roles=None) -> None
- Location: [evacusim/metrics/results_writer.py#L141](../../../evacusim/metrics/results_writer.py#L141)
- Summary: Save final simulation results with all reports.
- Key calls: agent_decisions.items, output_path.parent.mkdir, logger.info, FinancialReporter.generate_report, AnalyticsGenerator.save_all_analytics, data.get, list, open, json.dump, f.write

## Coverage Notes

- Nested helper functions documented in this file: 0
