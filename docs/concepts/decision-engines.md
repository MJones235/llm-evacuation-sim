# Decision engines

A decision engine answers one question per person per decision: *given what
they know, what do they do now?* Choose one with `decision.engine` (usually
through a study's engine overlay).

| | `rule_based` | `llm` |
|---|---|---|
| Input | the `DecisionContext`, plus **cue annotations** in the scenario | the `DecisionContext`, rendered as a prompt; reads the messages' **text** |
| How it decides | a 3-stage state machine with lognormal response delays, then weighted exit scoring | a language model, with validation and repair |
| Randomness | per-person streams derived from the run seed | the model's sampling (`llm.temperature`) |
| Cost per run | none | tokens (reported in `manifest.json` and `financial_report.txt`) |
| Speed | about a minute for a 15-minute drill | network-bound; decisions run concurrently |

Both return the same **payload** (`evacusim.decision.payload`):

```json
{"action": "evacuate", "exit_id": "escalator_b_up", "pace": "hurrying",
 "wait_reason": null, "reassess_when": "next_interval", "assessment": {...}}
```

The payload is checked against what was *offered* to that person at that
moment (an exit may have closed, a train may have left). An invalid payload is
never applied.

## Rule-based

Each person passes through three stages, after the Protective Action Decision
Model[^padm] and Kuligowski's model of behaviour in fires[^kul]:

| Stage | Behaviour |
|---|---|
| **unaware** | No warning yet. Pursues their journey (catch a train, leave the station) by the routing below. |
| **aware** | A warning cue has been perceived. Stops their journey and investigates: walks to staff or another information source if one is reachable, otherwise waits. |
| **evacuating** | Acts: follows an instruction (`board_train`), or leaves by the best route. |

**Cues** are the alarm, PA announcements and staff directives the person has
perceived. Each is annotated in the scenario with a `strength` (`weak`,
`medium`, `strong`) and an `instruction` ([Events, cues and staff](events-and-staff.md)).
A **social cue** fires when at least `decision.social.threshold` of the
people within `social.radius_m` are already evacuating.

Each cue schedules evacuation at *cue time + delay*. The delay is drawn from a
lognormal distribution, a common fit to observed pre-movement times[^purser],
with median `decision.response_median_s.<strength>` and shape
`decision.response_sigma`. The earliest scheduled time wins: a stronger later
cue can bring evacuation forward but never put it back. The delay starts no
earlier than the person's arrival, so someone who arrives after the alarm
still takes time to respond.

**Routing** scores each offered exit with signals normalised across the
options (min–max):

```
score = w_proximity·proximity + w_busyness·quietness
      + w_familiarity·familiar + w_visibility·visible
```

Here proximity is `1 / (1 + route length)`, quietness is minus the crowd
within `crowd_radius_m`, and familiar and visible are 0 or 1. The weights are
`decision.rule_weights`; ties go to the first option, so the choice is
deterministic. Exit tags (`station.exit_semantic_tags`) narrow the options
first: for example, someone leaving prefers exits towards the street.

The `rationale` column of `decisions.csv` says which rule fired and lists the
candidates' scores.

## LLM

For each person, `LLMDecisionEngine.decide`:

1. **Renders** the context into a prompt from a template with six blocks: role,
   journey, what is new, surroundings, the offered actions, and the JSON
   response format. The template can be overridden with
   `prompts.decision_prompt_template_path`.
2. **Reuses** the previous decision without calling the model when nothing
   significant has changed (prompt cache), or when the person decided very
   recently and nothing is new.
3. **Asks** the person's Concordia entity, up to three times. A response that
   is not valid JSON, or does not fit what was offered, is asked again with
   the errors appended (`repair_status`: `repair_1`, `repair_2`).
4. **Falls back** to a safe payload if all attempts fail (`fallback`:
   continue if possible, otherwise wait).

Up to `performance.max_parallel_agents` requests run at once, each limited to
`performance.decision_timeout_seconds`. Tokens and estimated cost are
recorded per run. `evacusim.testing.fake_llm` stands in for the model in
tests.

## Adding an engine

Subclass `evacusim.core.decision_engine.DecisionEngine`, implement `decide`
(and `reset_agent` / `on_agent_exit` if it keeps per-person state), add a
config model to the `decision` union in `evacusim.config.schema`, and
construct it in `SimulationRunnerFactory`. Everything else (perception,
execution, outputs) is shared.

[^padm]: Lindell, M. K. and Perry, R. W. (2012). The Protective Action Decision
    Model: theoretical modifications and additional evidence. *Risk Analysis*
    32(4), 616–632.
[^kul]: Kuligowski, E. D. (2013). Predicting human behavior during fires.
    *Fire Technology* 49(1), 101–120.
[^purser]: Purser, D. A. and Bensilum, M. (2001). Quantification of behaviour
    for engineering design standards and escape time calculations. *Safety
    Science* 38(2), 157–182.
