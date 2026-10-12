# Changelog

Changes that affect what a run does or produces. Pull requests are on
[GitHub](https://github.com/MJones235/llm-evacuation-sim/pulls).

## Unreleased

- Documentation site (MkDocs Material, API generated from docstrings),
  replacing the outdated `docs/evacusim/` pages; architecture decision records.
- `video.title`: the video's title is configurable (it said "Monument
  Station Evacuation" for every venue).

## 2026-10-12: Nobody lost or wedged; decisions drive all movement (#16)

- Passengers reaching a closed escalator return to the landing to re-decide
  (they were lost).
- Seeking information clears the old exit, so choosing it again takes effect.
- Nobody spawns within 1 m of a street exit.
- `ProgressWatchdog` steps aside anyone stuck 45 s without progress (ADR 5).
- Spawned passengers hold until their first decision; waiting passengers board
  only their own platform's train (`AgentRole.target_from_zone`).

## 2026-10-12: Config overlays; `frames.jsonl` (#15)

- `ConfigLoader.load_run_config(..., overlays=[...])`.
- Position frames are written to `frames.jsonl`. The old name is still read.

## 2026-10-09: Scenario fidelity (#14)

- Offered exits are those an agent knows, sees, or is told about (`cue.route`).
- Staff `on_event` phases start on a warning; staff patrols work again.
- Sampled populations get the requested size and never overlap.

## 2026-10-09: Self-describing runs (#13)

- `manifest.json` (parameters, code versions, status), `decisions.csv`,
  run directories named `run_<time>_<engine>_s<seed>`. Interrupted runs exit
  130.

## 2026-10-09: Rule engine stages (#12)

- The rule-based engine is a cue-driven state machine (unaware, aware,
  evacuating) with lognormal response delays and a social cue.

## 2026-10-08: The simulation loop (#11)

- Failed simulations raise `SimulationError`; staff move and speak again.
- Decide all, then apply in a fixed order (ADR 4); outputs on simulated time.

## 2026-10-08: The decision seam (#10)

- Perception, engines, payload and execution split apart (ADR 3); a
  deterministic fake LLM for tests; about 1,400 lines of dead code removed.

## 2026-10-08: Typed configuration (#9)

- Every key declared in `evacusim.config.schema` (ADR 2); one master seed;
  runs reproducible without `PYTHONHASHSEED`.

## 2026-10-08: Safety net (#8)

- Ruff, pre-commit and CI; Python 3.12.

## Earlier

- Conveyor escalators (ADR 1), calibration runs from real usage data, and the
  first LLM and rule-based engines.
