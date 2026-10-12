# 5. A watchdog frees people wedged in place by the physics

*2026-10-12. Accepted.*

## Context

Fitting the rule engine to Proulx's drills needs the time at which the
station clears. Many runs never cleared because one or two people stayed
inside for the rest of the run. Three of the causes were bugs, fixed
separately:

- people reaching a closed escalator were lost;
- a stale destination after seeking information meant later decisions were
  ignored;
- people spawned inside an exit were removed before the exit log started.

The fourth cause is the physics. JuPedSim's collision-free speed model stops
anyone with someone directly in their path, so two people heading opposite
ways against a wall can stand pressed together indefinitely. On Monument's
platform level this happens at a corner around (−19.5, −4). On a normal
morning it also happens to people crossing the crowd at the foot of an
escalator. Real people would sidestep.

## Decision

`ProgressWatchdog` (`evacusim.coordination.progress_watchdog`) checks once
per simulated second for anyone who:

- is moving towards a target more than 1.5 m away (or an unknown one);
- has not moved 0.25 m in 45 s;
- is not queueing for an escalator and is not waiting for a train.

The runner sends them to a free spot 1–2 m to the side, preferring sideways
to the way they were heading. Their route is forgotten, and they decide again
3 s later. Every intervention is logged as `[WATCHDOG]` in `simulation.log`.

## Alternatives

- **Change the geometry at the known corner.** That fixes one place but not
  the general case, such as people crossing a queue.
- **Tune the speed model** (time gap, range). This affects everyone's
  movement and the calibration, to fix a rare event.
- **Leave it, and measure clearance as, say, the 95th percentile.** This
  hides a model artefact in the metric, and the drills report the last
  person out.

## Consequences

- With the fitted rule-based parameters, every validation run of E2, E4 and
  E5 cleared. Before, up to all five runs of a drill did not.
- On a normal morning (the 5 October replay) the watchdog intervenes about 35
  times. The calibrated exit counts were unchanged: Blackett Street averaged
  358 with and without it, against 354 counted.
- The thresholds are constants in the module. Congestion that holds someone
  still for over 45 s at an unremarkable spot would also trigger it. That has
  not been seen, but check the `[WATCHDOG]` lines when you calibrate a new
  station.
