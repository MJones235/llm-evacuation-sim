# 1. Escalators are conveyors detached from the floor simulation

*2026-10-05. Accepted.*

## Context

Escalators used to be strips of walkable floor. People walked up them, and
were moved to the other level when they reached a small exit square halfway
along. Under load, JuPedSim's collision-free speed model gridlocked at that
0.24 m square at the end of a dead-end corridor. In a replay of a real
morning (5 October 2026), escalator F stopped at 09:12 and E at 09:42, for
good, leaving 324 people stranded on the platforms. Every transfer also reset
the person's walking speed to 1.34 m/s.

## Decision

An escalator is a two-lane **conveyor** with its own kinematics
(`evacusim.escalators.conveyor`). Riders are not on any floor.
`EscalatorSystem` couples the conveyors to the floors:

- Each waiting person gets their **own queue slot**: a short line per lane in
  front of the comb, then overflow slots across the landing.
- Only the head of a lane's line is admitted onto a full-width boarding strip.
- At the far end, riders are placed on the landing with their own speed. If
  the landing is full, the belt pauses.

## Alternatives

- **JuPedSim's `NotifiableQueueStage`.** Tried in a spike on the real
  geometry. Waiting agents milled in front of the released one at the
  wall-side queue box; 9 of 40 boarded, then it stalled for good.
- **Tuning the old floor model** (admission gates, bounce-back, deferring
  decisions while stalled). This treated symptoms and kept the dead end that
  caused the gridlock.

## Consequences

- The conveyor cannot gridlock; the spike cleared 250 people with no stall.
  Congestion can form only on the floor at the boarding comb, which is where
  it forms in reality.
- Throughput is about 0.7–0.85 people per second per escalator. That is set
  more by queue dynamics on the floor than by the belt's capacity.
- Riders vanish from the floor while riding. Anything that counts people on a
  level must also count riders (`MultiLevelJuPedSimulation.is_agent_in_transit`).
- Each escalator's combs must be given in the geometry
  (`jupedsim.escalator_comb`). The study generates them with a script.
