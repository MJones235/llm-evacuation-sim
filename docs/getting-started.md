# Getting started

## Install

evacusim needs Python 3.12 or newer.

```bash
git clone git@github.com:MJones235/llm-evacuation-sim.git evacusim
cd evacusim
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

The `dev` extra adds pytest, ruff, mypy and pre-commit. Add `docs` to build
this site (`pip install -e ".[dev,docs]"`, then `mkdocs serve`).

## Run a simulation

evacusim is a library. A study repository provides the geometry and the
scenario configs and drives runs. With
[monument-evacuation](https://github.com/MJones235/monument-evac-sim) cloned
next to this repository:

```bash
cd ../monument-evacuation
bash setup.sh            # installs evacusim (editable) and the study
monument run E1 --engine rule_based
```

A run writes `results/E1/run_<time>_rule_based_s0/`. Start with its
`manifest.json` ([Outputs](outputs.md)).

To drive a run from your own code, follow `monument_evacuation/run.py`, which
is a short example:

1. `ConfigLoader.load_run_config(path, overlays=[...])` returns a validated
   `RunConfig`.
2. Build the pedestrian simulation, the station layout and the agents
   (`JuPedSimSetup`, `StationLayoutBuilder`, `AgentManager`).
3. `SimulationRunnerFactory.create_runner(...)`, then `runner.run()`.
4. `ResultsWriter.save_final_results(...)` and the manifest
   (`evacusim.metrics.manifest`).

## The LLM engine

The LLM engine needs an Azure-hosted model. Set these in the environment (or
in the study's `.env`):

```bash
AZURE_LLM_ENDPOINT=...
AZURE_LLM_API_KEY=...
AZURE_LLM_MODEL=...
```

To test the LLM pipeline without a model or any cost, use
`evacusim.testing.fake_llm` (`monument run ... --fake-llm`). It answers
deterministically, so the decisions it makes mean nothing.

## Tests

```bash
pytest -q
```

Tests that need the Monument geometry look for a `monument-evacuation`
checkout next to this one (or `EVACUSIM_MONUMENT_NETWORK`) and are skipped if
it is missing. The study repository also has golden-run tests: short seeded
simulations whose outputs must not change unless a behaviour change is
intended.
