# Deep Dive: Walking Speed and Motion

## Initial Speed Assignment

Primary logic:
- [sample_walking_speed](../function-reference/utils__walking_speed.md)

Characteristics:
- Gaussian sampling centered on typical pedestrian walking speed.
- Hard clipping to a bounded range for stability and realism.
- Injured agents get explicit speed override in agent setup path.

Setup application:
- [AgentManager._add_agents_to_jupedsim](../function-reference/setup__agent_manager.md)

## Action-Driven Speed Updates

Decision-time speed conversion:
- [convert_speed_to_ms](../function-reference/utils__speed_utils.md)

Application path:
- [ActionExecutor.execute_action](../function-reference/decision__action_executor.md)

Behavior:
- Action labels map to stochastic speed distributions.
- Following behavior can sync speed with target agent when close.
- Wait-related behaviors can enforce low-speed defaults.

## Escalator Overrides

Per-step enforcement:
- [MultiLevelJuPedSimulation._enforce_escalator_constraints](../function-reference/jps__multi_level_simulation.md)

Behavior:
- Escalator zone occupancy can override low desired speed to belt-floor value.
- Protects against unrealistic stalling while on escalators.

## Related Motion Control APIs

- [ConcordiaJuPedSimulation.set_agent_target](../function-reference/jps__jupedsim_integration.md)
- [ConcordiaJuPedSimulation.set_agent_speed](../function-reference/jps__jupedsim_integration.md)
- [ActionExecutor._safe_follow_target](../function-reference/decision__action_executor.md)

## Design Implications

- Speed is intentionally stochastic to avoid deterministic crowd artifacts.
- Speed control appears at multiple layers (setup, action, escalator), so precedence and interactions matter for debugging.
