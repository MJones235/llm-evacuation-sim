# Deep Dive: Escalators and Level Transfer

## Implementation Model

Each escalator is represented by named polygons in the station XML:

- terminal transfer zones: L{level}_esc_{letter}_{direction}
- corridor polygons: L{level}_esc_corridor_{letter}

These are parsed in:
- [LevelTransferManager._load_escalator_zones](../function-reference/coordination__level_transfer_manager.md)

Transfer pairings are built in:
- [LevelTransferManager._build_transfer_mappings](../function-reference/coordination__level_transfer_manager.md)

## Transfer Execution Path

1. Per-step transfer detection
- [MultiLevelJuPedSimulation.step](../function-reference/jps__multi_level_simulation.md)
- [MultiLevelJuPedSimulation._check_level_transfers](../function-reference/jps__multi_level_simulation.md)

2. Teleport-style transfer
- [MultiLevelJuPedSimulation._do_level_transfer](../function-reference/jps__multi_level_simulation.md)
- Agent is removed from source level sim, re-added in target level transfer zone.

3. Immediate redecision handoff
- Recently transferred agents are queued for immediate decision updates in runner loop.

## Escalator Constraints

Enforced in:
- [MultiLevelJuPedSimulation._enforce_escalator_constraints](../function-reference/jps__multi_level_simulation.md)

Behavior includes:
- Minimum escalator belt speed floor while inside escalator corridors/zones.
- Directionality correction if agent enters arrival-only zone incorrectly.
- Cached corridor routing to avoid redundant path updates.

## Exit Blocking Interaction

Initial block:
- [GeometryManager._apply_initial_blockages](../function-reference/jps__geometry_manager.md)
- [ExitManager._create_escalator_exits](../function-reference/jps__exit_manager.md)

Runtime block:
- [EventManager.block_exit](../function-reference/systems__event_manager.md)
- [ConcordiaJuPedSimulation.add_geometry_obstacle_for_exit](../function-reference/jps__jupedsim_integration.md)
- [GeometryManager.add_obstacle_polygon](../function-reference/jps__geometry_manager.md)

## Design Implications

- Multi-level movement is discrete transfer rather than continuous vertical simulation.
- This design simplifies implementation but can reduce realism around escalator travel continuity.
- Transfer-cooldown and spawn collision handling are important for stability at high transfer rates.
