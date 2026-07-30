# Function Reference: evacusim/utils/performance_monitor.py

- Source file: [evacusim/utils/performance_monitor.py](../../../evacusim/utils/performance_monitor.py)
- Top-level classes: 1
- Top-level functions: 0

## Module Summary

Performance monitoring utilities for simulation profiling.

## Classes and Methods

### Class PerformanceTimer

- Location: [evacusim/utils/performance_monitor.py#L12](../../../evacusim/utils/performance_monitor.py#L12)
- Summary: Simple performance timer for profiling simulation bottlenecks.
- Method count: 4

#### __init__

- Signature: def __init__(self)
- Location: [evacusim/utils/performance_monitor.py#L15](../../../evacusim/utils/performance_monitor.py#L15)
- Summary: No docstring summary available.
- Key calls: set

#### record

- Signature: def record(self, name, duration, is_parallel=False)
- Location: [evacusim/utils/performance_monitor.py#L23](../../../evacusim/utils/performance_monitor.py#L23)
- Summary: Record a timing measurement.
- Key calls: self.parallel_operations.add, max, self._parallel_min.get, min, self._parallel_sum.get, float

#### measure

- Signature: def measure(self, name, is_parallel=False)
- Location: [evacusim/utils/performance_monitor.py#L44](../../../evacusim/utils/performance_monitor.py#L44)
- Summary: Context manager for timing a block of code.
- Key calls: time.perf_counter, self.record

#### report

- Signature: def report(self) -> str
- Location: [evacusim/utils/performance_monitor.py#L58](../../../evacusim/utils/performance_monitor.py#L58)
- Summary: Generate performance report.
- Key calls: sum, sorted, lines.append, join, self._parallel_sum.get, self.timings.items, self._parallel_min.get
- Nested function count: 1

##### Nested helpers inside report

###### report._sort_key

- Signature: def _sort_key(item)
- Location: [evacusim/utils/performance_monitor.py#L72](../../../evacusim/utils/performance_monitor.py#L72)
- Summary: No nested-function docstring summary available.
- Key calls: self._parallel_sum.get

## Coverage Notes

- Nested helper functions documented in this file: 1
