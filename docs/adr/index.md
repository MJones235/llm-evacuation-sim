# Design decisions

Architecture decision records (ADRs): one page per significant choice. Each
gives the context, what was decided, the alternatives, and the consequences.
They explain why the code is shaped as it is; read one before changing what
it describes. Add a record when you make a choice like these.

| # | Decision | Date |
|---|---|---|
| [1](0001-conveyor-escalators.md) | Escalators are conveyors detached from the floor simulation | 2026-10-05 |
| [2](0002-typed-configuration.md) | Configuration is a typed, strict schema | 2026-10-08 |
| [3](0003-decision-seam.md) | One seam between perception and decision, shared by both engines | 2026-10-08 |
| [4](0004-decide-then-apply.md) | Everyone in a cycle decides, then decisions are applied in a fixed order | 2026-10-08 |
| [5](0005-progress-watchdog.md) | A watchdog frees people wedged in place by the physics | 2026-10-12 |
