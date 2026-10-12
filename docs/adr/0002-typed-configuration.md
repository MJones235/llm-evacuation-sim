# 2. Configuration is a typed, strict schema

*2026-10-08. Accepted.*

## Context

Configuration was raw dictionaries. Defaults were scattered across
`config.get(key, default)` calls, sometimes different for the same key in
different places. The engine choice was parsed in three places. Several keys
in the study's YAML were read by nothing: a misspelt key silently did nothing,
and so did a stale one. In one case a scenario's fire brigade used
`trigger_after_seconds` where the engine read `after_seconds`, so it never
arrived.

## Decision

`evacusim.config.schema` declares every key in pydantic v2 models, with its
type, default and description, as one `RunConfig`:

- Models are strict and forbid extra keys. An unknown key, or a value of the
  wrong type, fails when the config is loaded.
- `decision` is a discriminated union of `LLMDecisionConfig` and
  `RuleBasedDecisionConfig`, so the engine is chosen once.
- `extends:` deep-merges files, and overlays are merged after them.
- [Parameters](../parameters.md) is generated from the schema, and a test
  fails if it is stale.

## Alternatives

- **Validate the dictionaries with a JSON schema.** This catches typos, but
  defaults would still live in the code that reads them.
- **Dataclasses.** No validation, and no discriminated union.

## Consequences

- The schema is the single definition of what a key means. Its descriptions
  are the reference documentation.
- Stale keys were found and deleted. Configs that relied on them failed
  loudly and were fixed.
- Some older components still take plain dicts, through `schema.as_dict`.
- One master `seed` replaced per-component seeds
  ([Reproducibility](../concepts/reproducibility.md)).
