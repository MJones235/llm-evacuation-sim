# Evacusim Technical Documentation

This directory contains comprehensive technical documentation for evacusim implementation, architecture, and design tradeoffs.

## How to Use This Documentation

1. Start with [architecture.md](architecture.md) for system structure.
2. Read [runtime-lifecycle.md](runtime-lifecycle.md) for startup and per-step execution flow.
3. Use [packages/README.md](packages/README.md) for package responsibilities and data contracts.
4. Use [function-reference/README.md](function-reference/README.md) for full function-by-function coverage.
5. Review [design-decisions.md](design-decisions.md) and [risk-register.md](risk-register.md) for review and improvement planning.
6. Review [preliminary-fix-plan.md](preliminary-fix-plan.md) for a staged remediation proposal before implementation.
7. Execute against [implementation-checklist-v1.md](implementation-checklist-v1.md) for concrete phased tasks.
8. Use deep dives for critical mechanics across simulation, orchestration, cognition, and analytics:
   - [deep-dives/README.md](deep-dives/README.md)
   - [deep-dives/bootstrap-and-dependency-wiring.md](deep-dives/bootstrap-and-dependency-wiring.md)
   - [deep-dives/event-scheduling-and-director-systems.md](deep-dives/event-scheduling-and-director-systems.md)
   - [deep-dives/messaging-and-information-propagation.md](deep-dives/messaging-and-information-propagation.md)
   - [deep-dives/prompt-caching-concurrency-and-cost-controls.md](deep-dives/prompt-caching-concurrency-and-cost-controls.md)
   - [deep-dives/outputs-analytics-and-observability.md](deep-dives/outputs-analytics-and-observability.md)
   - [deep-dives/escalators-and-level-transfer.md](deep-dives/escalators-and-level-transfer.md)
   - [deep-dives/walking-speed-and-motion.md](deep-dives/walking-speed-and-motion.md)
   - [deep-dives/line-of-sight-and-visibility.md](deep-dives/line-of-sight-and-visibility.md)
   - [deep-dives/decision-trigger-and-execution.md](deep-dives/decision-trigger-and-execution.md)

## Coverage Summary

- Source files documented: 72
- Top-level classes: 54
- Top-level functions: 23
- Methods: 368
- Nested helper functions: 10
- Total documented symbols: 455

Coverage is generated from AST in [function-reference/README.md](function-reference/README.md), then augmented with architecture and design analyses in this directory.
