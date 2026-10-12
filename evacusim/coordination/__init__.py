"""Running a simulation: the main loop and its helpers.

- :class:`HybridSimulationRunner` steps physics, events, staff and decision
  cycles and writes the outputs.
- :class:`ObservationCoordinator` builds each agent's observation.
- :class:`SimulationStateQueries` answers spatial questions about the state.
- :class:`ProgressWatchdog` frees people wedged in place by the physics.
"""
