# Known limitations

Limits of the current code worth knowing before relying on a result or
adapting evacusim to a new venue.

- **The video and the live spatial viewer draw two levels**, `0` and `-1`
  (`evacusim.visualization`). The simulation itself handles any number of
  levels (`simulation.levels`), but a station with more needs the renderers
  generalised.
- **The LLM engine uses Azure-hosted models** (`evacusim.concordia.azure_llm_concordia`,
  configured by `AZURE_LLM_*`). Another provider needs a Concordia language
  model wrapper with the same usage statistics.
- **The progress watchdog's thresholds are constants**: 45 s without 0.25 m of
  progress (ADR 5). Check the `[WATCHDOG]` lines in `simulation.log` when you
  calibrate a new station.
- **Exits are logged from the first step.** Anyone removed before then (for
  example, spawned inside an exit, which spawning now prevents) would not
  appear in `exit_log.csv`. Nothing flags this if it happens another way.
- **Decisions are only as fine as the cadence.** A person re-decides every
  `simulation.decision_interval` (30 s in the Monument study) unless something
  new happens, so pre-movement times move in steps of that size.
- **Cue strengths are three levels.** The rule engine cannot, for example,
  give one crowd a PA as a medium cue and another the same PA as a weak cue,
  unless the scenario labels each zone separately (`zone_cues`).
