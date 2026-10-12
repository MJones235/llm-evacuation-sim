# 3. One seam between perception and decision, shared by both engines

*2026-10-08. Accepted.*

## Context

The question the study asks is how an LLM compares with rules *as the
decision-maker*. That comparison is only fair if both engines see exactly the
same situation and their decisions are carried out identically.

The code did not allow it:

- `decision_processor.py` was nearly 2,000 lines mixing perception, prompt
  building, station-specific routing, LLM retry and caching, and execution.
- The LLM engine called back into the processor's private methods.
- Decisions travelled as dict, then JSON, then were re-parsed.
- Station names (Monument's levels and zones) were hard-coded in the engine.

## Decision

The decision cycle is split into steps with one interface between them
(`evacusim.core.decision_engine`):

```
SituationAssembler.perceive → frame ──DecisionContext──▶ DecisionEngine.decide
──DecisionResult (payload)──▶ translator → ActionExecutor
```

- **Perception and framing** (`decision/situation.py`) build one
  `DecisionContext`: the goal, the observation, and the offered actions and
  exits with routing signals.
- **Engines** subclass `DecisionEngine`. The LLM engine owns its prompt,
  cache, repair and fallback. The rule engine owns its stages.
- **The payload** (`decision/payload.py`) is the one shape every engine
  returns. It is validated against the offered set, with a safe fallback.
- **Execution** is engine-agnostic.
- Station-specific names became configuration (`station.*`) or documented
  geometry conventions (`evacusim.conventions`).

## Consequences

- An engine is swapped by one config key, and everything else is shared.
  Adding an engine means implementing `decide` and adding a config model.
- About 1,400 lines of dead code (agent speech, a legacy response schema,
  group decisions) were removed during the split.
- The rule engine reads cue *annotations*, not message text. Labelling them
  is authoring work the LLM engine does not need, and the comparison counts
  it ([Events, cues and staff](../concepts/events-and-staff.md)).
