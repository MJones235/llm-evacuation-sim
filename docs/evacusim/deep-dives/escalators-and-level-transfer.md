# Deep Dive: Escalators and Level Transfer

Escalators are **not floor**. Each one is a two-lane conveyor, detached from the
per-level JuPedSim simulations: passengers queue on the landing, step on at the
boarding comb, leave the floor simulation while they ride, and are placed on
the other level's landing at the far comb. Code: `evacusim/escalators/`.

Why: modelling the incline as walkable floor let the JuPedSim
collision-free speed model gridlock permanently at the level transfer point
(see "History" below). A real escalator cannot gridlock: the steps carry people
regardless of crowding, at most one person per step per lane, so congestion only
ever forms at the boarding comb on the open landing.

## Geometry

Each level file (`level_{N}.xml`) has one `jupedsim.escalator_comb` record per
escalator end on that level:

| attribute | meaning |
|---|---|
| `escalator`, `exit_name`, `direction` | letter, decision-facing exit name (`escalator_f_up`), `up`/`down` |
| `role` | `entry` (boarding comb) or `exit` (alighting comb) |
| `shape` | the comb as a 2-point segment |
| `floor_nx`, `floor_ny` | unit normal pointing from the comb onto the landing floor |
| `length_m` | incline length (default: sum of the two plan strips) |
| `egress_x`, `egress_y` | (exit side) legacy egress point at the comb |

They are generated from the corridor strips by the project script
`scripts/generate_escalator_combs.py` (monument-evacuation): each comb is the
strip edge farthest from that level's old transfer zone. The corridor strips
(`jupedsim.escalator`) and `esc.*` transfer zones are **removed from the
walkable floor** at load (`GeometryManager._detach_escalators`) and kept only
for drawing.

## Behaviour (`conveyor.py`, pure Python)

Positions `s` run along the incline from 0 (boarding comb) to `length_m`.

- **Stand lane (right of travel).** Carried at belt speed; ride time
  `length_m / belt_speed`. A stander boards when the previous one is at least one
  step up (two steps with probability `stander_step_gap_prob`). Ceiling
  `belt_speed / step_depth` people/s.
- **Walk lane (left).** Speed = belt + own pace (`walk_speed_factor` × the
  agent's floor speed), never slower than the belt, never overtaking, always
  at least one free step (`2 × step_depth`) behind the walker ahead.
- **Belt stopped** (`stop_belt`): a staircase; everyone walks at own pace.
- **Paused**: nobody moves or boards (the far landing is full).
- **Closed** (`close`): no boarding; riders finish.

## Coupling to the floor (`system.py`, `EscalatorSystem`)

1. **Assign.** `MultiLevelJuPedSimulation.set_agent_destination_exit` routes
   escalator exits to `EscalatorSystem.assign`. The agent gets a lane (walk with
   probability `walk_share[up|down]`, seeded per agent) and a queue slot.
   Only escalators boarding on the agent's level are accepted.
2. **Queue.** Every waiting agent targets its **own** waypoint slot, never a
   shared point. People on the landing per lane take that lane's short line in
   front of the comb in physical order (a holder near their slot keeps it; anyone
   displaced is ranked by where they actually are); the rest take overflow slots spread over the landing,
   nearest-first by route distance, outside a clear apron and every discharge
   landing. Line agents get JuPedSim `time_gap` = `queue_time_gap` (queues close up).
3. **Admit.** When a lane can admit, the agent at that lane's head slot is
   switched to the boarding strip (an `ExitStage` across the full comb width);
   one admitted agent per lane at a time. Anyone pushed onto the strip boards
   when either lane has room; an admission not boarded within 20 s is requeued.
   If people queue but nobody boards for 60 s, the queue state is logged as a
   warning (`no boarding for`).
4. **Board.** JuPedSim removes the agent at the strip; `board()` puts them on the
   conveyor. They are absent from `agent_levels` and floor positions while
   riding; `is_agent_in_transit()` is true (ExitTracker relies on this).
5. **Discharge.** At `s = length_m` the rider is placed at their lane's landing
   spot, else the nearest free spot on a strip across the comb
   (`landing_search_depth` deep). They keep their **own** walking speed and walk
   to one of a spread of egress points 3–8 m out (round robin), recorded in
   `transfer_escape_waypoints`. If no spot is free the belt pauses until one is.
   Nobody is ever sent back.

A spike on the Monument geometry showed JuPedSim's `NotifiableQueueStage`
gridlocks at a comb (waiting agents mill in front of the released one), which is
why queueing is controller-managed.

## Decisions, events, staff

- Riders get no decisions (no floor position). Agents who have joined a queue
  or been admitted are deferred (`EscalatorSystem.is_committed`); agents still
  walking towards an escalator may re-decide.
- The busyness of an escalator option is `EscalatorSystem.load()` (queueing +
  boarding + riding).
- `block_exit` on an escalator calls `MultiLevelJuPedSimulation.block_escalator`:
  it closes it, releases its queue for re-decision, and records the landing in
  `blocked_exit_positions`. Agents choosing a closed escalator walk to its
  landing, see it is closed, and are flagged to re-decide. Escalators blocked
  at t ≤ 0 are built closed and never offered as exits.
- Director agents route via the nearest open escalator and ride like anyone.

## Configuration

`simulation.escalators`: `defaults` (`belt_speed`, `step_depth`, `walk_share`,
`stander_step_gap_prob`, `walk_speed_factor`, `landing_search_depth`, queue
slot settings) and `overrides` keyed by exit name.

## Outputs

- `escalator_log.csv`: one row per ride (`chose_s` = chose the escalator,
  `queue_join_s` = reached its landing, `board_s`, `alight_s`,
  `ride_s`, `lane`, `stall_wait_s`, `discharge_attempts`).
- `escalators.json`: comb geometry, lengths, plan strip lengths.
- History frames carry `escalators`: per escalator `queue`, `riders`
  (`[agent_id, lane, s]`), `stalled`, `closed`.
- Population monitor zone types `escalator_queue` and `on_escalator`.
- Video: an escalator strip panel plus riders projected onto the plan
  (`escalators.drawing.rider_floor_position`).

## History

Until 2026-10, escalators were walkable corridor strips with an instant
teleport between levels at a 0.24 m exit square halfway up, plus a routing
admission gate, transfer cooldowns and bounce-back on a full landing. Under load
the corridor gridlocked permanently (Validation_20261005: escalator F stopped at
09:12) and every transfer reset walking speed to 1.34 m/s. All of that was
replaced by the conveyor model.
