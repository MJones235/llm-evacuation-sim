# evacusim

Pedestrian evacuation simulation in which each person's decisions come from a
language model (`llm`) or from deterministic rules (`rule_based`), with
movement simulated by [JuPedSim](https://www.jupedsim.org/). Both engines see
the same situation and return the same kind of decision, so a scenario can be
run with either and compared.

```
config YAML ─▶ RunConfig ─▶ each step: physics → exits → events & staff → decisions ─▶ results/<run>/
(scenario +    (typed,                                  (perceive → frame →             manifest.json,
 engine)        validated)                                decide → apply)                decisions.csv, ...
```

The engine is venue-agnostic. A study repository supplies geometry, scenarios
and analysis; the reference study is
[monument-evacuation](https://github.com/MJones235/monument-evac-sim)
(Proulx's 1991 drills at Monument station, Newcastle).

## Documentation

Build and read the site locally:

```bash
pip install -e ".[docs]"
mkdocs serve
```

It covers how a run works, the two decision engines, escalators, events and
staff, reproducibility, every configuration key (generated from the schema),
every output file, the design decisions behind the code, and the API.

## Install and test

```bash
pip install -e ".[dev]"     # Python 3.12+
pytest -q
```

Tests that need the Monument geometry expect a `monument-evacuation` checkout
next to this one (or `EVACUSIM_MONUMENT_NETWORK`) and are skipped without it.

## Layout

```
evacusim/
├── config/         typed schema (every key and default), loader, reference generator
├── core/           the decision-engine interface
├── coordination/   HybridSimulationRunner (the main loop), progress watchdog
├── decision/       perception, both engines, payload, execution
├── escalators/     two-lane conveyor escalators
├── jps/            JuPedSim wrappers, single- and multi-level
├── systems/        scheduled events, PA and directives, staff agents
├── calibration/    normal-operation runs from real usage data
├── concordia/      LLM agents (Azure) for the LLM engine
├── translation/    simulation state ↔ natural-language observations
├── metrics/        manifest, outputs, population monitor, LLM cost
├── setup/          building a run from its config
├── visualization/  live viewers and video
├── testing/        a deterministic fake LLM
└── utils/          logging, seeding, timing
```

See [CHANGELOG.md](CHANGELOG.md) for what changed and when.
