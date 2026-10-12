# Conventions and glossary

## Glossary

| Term | Meaning |
|---|---|
| **agent** | A simulated person: a passenger or a staff member. |
| **passenger** | An agent whose actions come from the decision engine. |
| **staff** | An agent driven by a `systems` entry (phases), never by the decision engine. |
| **scenario** | A run configuration describing one situation (e.g. a Proulx drill), usually `extends:` a base config. |
| **engine** | The decision engine: `llm` or `rule_based`. |
| **overlay** | A config file merged over the scenario, e.g. an engine's settings or a fit trial's parameters. |
| **run** | One scenario × engine × seed; one output directory. |
| **decision cycle** | A round of decisions, for one group or for everyone after a critical event. |
| **offered** | The actions and exits available to one agent at one decision. |
| **cue** | A warning an agent perceived: alarm, PA, staff directive, or seeing others leave. |
| **stage** | The rule engine's `unaware`, `aware` or `evacuating`. |
| **level** | One floor, with its own JuPedSim simulation (`level_<id>.xml`). |
| **zone** | A named area on a level (`concourse`, `platform_3`), used for messages, routing and monitoring. |
| **comb** | Where an escalator's belt meets the landing. |

## Geometry names

A station's geometry (`geometry/<station>/network/level_*.xml`) must follow
these names:

| Name | Meaning |
|---|---|
| `train_platform_N` | An exit that boards the train at platform `N`; open only while a train dwells there. |
| `platform_N` | The walkable area and zone of platform `N`. |
| `escalator_<x>_<up\|down>` | Escalator `x` and the direction it runs. |

Which level is the street and which the platforms is configured
(`station.street_level`, `station.platform_level`).

## Code

- **Docstrings** are Google style (`Args:`, `Returns:`, `Raises:`). Module
  docstrings say what the module is for and how it fits; the API pages are
  generated from them.
- **Configuration** keys are declared, with type, default and meaning, only
  in `evacusim.config.schema`. Reading a key anywhere else with a default is a
  bug.
- **Lint and format**: `ruff check` and `ruff format`, run by pre-commit and
  CI.
- **Behaviour changes** come with a test that fails without them. If golden
  snapshots change, the commit message says what moved and why.
