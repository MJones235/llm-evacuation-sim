"""
Simulation coordination: the main HybridSimulationRunner and its supporting processors.

Provides:
- HybridSimulationRunner: orchestrates JuPedSim + Concordia at each timestep
- DecisionProcessor: batches and processes LLM agent decisions
- ObservationCoordinator: generates and distributes observations to agents
- SimulationStateQueries: stateless spatial queries over simulation state
- LevelTransferManager: manages agent transitions between levels (e.g. escalators)
"""
