# Function Reference: evacusim/metrics/analytics_generator.py

- Source file: [evacusim/metrics/analytics_generator.py](../../../evacusim/metrics/analytics_generator.py)
- Top-level classes: 1
- Top-level functions: 0

## Module Summary

Analytics generation for Station Concordia simulations.

## Classes and Methods

### Class AnalyticsGenerator

- Location: [evacusim/metrics/analytics_generator.py#L18](../../../evacusim/metrics/analytics_generator.py#L18)
- Summary: Generates analytics reports from simulation data.
- Method count: 4

#### save_all_analytics

- Signature: def save_all_analytics(output_path, route_changes, wait_events, message_history) -> None
- Location: [evacusim/metrics/analytics_generator.py#L27](../../../evacusim/metrics/analytics_generator.py#L27)
- Summary: Save all analytics reports to files.
- Key calls: AnalyticsGenerator._save_route_change_analytics, AnalyticsGenerator._save_wait_behavior_analytics, AnalyticsGenerator._save_message_analytics

#### _save_route_change_analytics

- Signature: def _save_route_change_analytics(output_path, route_changes) -> None
- Location: [evacusim/metrics/analytics_generator.py#L47](../../../evacusim/metrics/analytics_generator.py#L47)
- Summary: Save route change analytics.
- Key calls: logger.info, open, f.write, len

#### _save_wait_behavior_analytics

- Signature: def _save_wait_behavior_analytics(output_path, wait_events) -> None
- Location: [evacusim/metrics/analytics_generator.py#L69](../../../evacusim/metrics/analytics_generator.py#L69)
- Summary: Save waiting behavior analytics.
- Key calls: logger.info, open, f.write, sorted, event.get, reason_counts.items, personality_counts.items, reason_counts.get, personality_counts.get, len

#### _save_message_analytics

- Signature: def _save_message_analytics(output_path, message_history) -> None
- Location: [evacusim/metrics/analytics_generator.py#L114](../../../evacusim/metrics/analytics_generator.py#L114)
- Summary: Save messaging analytics.
- Key calls: logger.info, open, f.write, sum, sorted, sender_counts.items, msg.get, type_counts.items, len, sender_counts.get

## Coverage Notes

- Nested helper functions documented in this file: 0
