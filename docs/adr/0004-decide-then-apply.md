# 4. Everyone in a cycle decides, then decisions are applied in a fixed order

*2026-10-08. Accepted.*

## Context

LLM decisions run concurrently (`performance.max_parallel_agents`). Each was
applied as soon as its network call returned, so the order of application,
and with it what later deciders in the same cycle saw, depended on network
timing. Two runs with the same seed gave different results, even with the
deterministic fake model.

## Decision

A decision cycle has two phases:

1. **Decide.** Every deciding person is perceived, framed and decided,
   concurrently for the LLM engine, from the same state of the world.
2. **Apply.** Decisions are recorded and executed one person at a time in a
   fixed order.

## Alternatives

- **Apply in completion order, but sort completions.** Later deciders would
  still see earlier decisions of the same cycle, depending on timing.
- **Serialise LLM calls.** Reproducible, but cycles would take many times as
  long.

## Consequences

- Runs are reproducible with the fake LLM and the rule engine, whatever the
  concurrency.
- Within a cycle, nobody reacts to another person's decision from that same
  cycle. They see it from the next. This changed the rule engine's behaviour
  slightly; the golden snapshots were re-recorded when it was introduced.
