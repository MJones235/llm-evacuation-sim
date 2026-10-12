# Events, cues and staff

## Events

A scenario's `events` list holds timed events; `time` is the simulated time
in seconds. The four types:

| `type` | What happens |
|---|---|
| `message` | Everyone perceives it, e.g. the fire alarm. Can repeat. |
| `pa_announcement` | A PA announcement, optionally different per zone (`zone_messages`). Can repeat. |
| `train_arrival` | A train dwells at the listed platforms for `dwell_seconds`, opening `train_platform_N` as a way out. Its departure is announced. |
| `block_exit` | Exits close, e.g. the escalators near the fire. |

The full field list is in [Parameters](../parameters.md).

## Cues: what a warning means to the rule engine

The LLM engine reads a message's text and judges it. The rule-based engine
cannot, so each warning carries a **cue** annotation:

```yaml
- time: 20
  type: pa_announcement
  message: "There is a suspected fire on the North/South escalators..."
  cue: {strength: strong}                 # what the general message conveys
  zone_cues:                              # per-zone messages, per-zone cues
    platform_1: {strength: strong, instruction: board_train}
    concourse:  {strength: strong, instruction: leave_station}
```

- `strength`: `weak`, `medium` or `strong`. It selects the median response
  delay (`decision.response_median_s`).
- `instruction`: `none`, `leave_station` or `board_train`. It is what an
  evacuating person does.
- `route`: exits the message names. Hearing it makes them known, so they are
  offered.

Labelling cues is part of authoring a scenario for the rule engine. A study
should record how it chose each label. The LLM engine ignores cue
annotations.

## Staff

`systems` defines groups of staff agents, such as station staff or
firefighters. They are pedestrians: they walk and take up space, but they
never use a decision engine. Each runs a list of **phases**:

| Phase field | Meaning |
|---|---|
| `trigger` | What starts the phase: `immediate`, `on_event` (the first event with a cue, e.g. the alarm), `on_reach_zone`, or `after_seconds`. |
| `movement` | `hold` (stand at `hold_zone`) or `zone_patrol` (visit `patrol_zones`, pausing `patrol_dwell_time` at each). |
| `message`, `messages_by_zone` | What they say to people within `directive_radius`, every `directive_interval` seconds. |
| `cue`, `cues_by_zone` | What that means to the rule engine. |

A person who decides to `seek_information` walks to the nearest reachable
staff member. Staff waypoints are snapped to walkable space, and a leg that
takes more than 90 s is abandoned, so staff never wait forever for a point
they cannot reach.
