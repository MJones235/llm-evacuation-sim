# evacusim

A reusable framework for LLM-driven pedestrian evacuation simulation, combining:

- **[JuPedSim](https://www.jupedsim.org/)** — pedestrian movement physics
- **[Concordia](https://github.com/google-deepmind/concordia)** — LLM-based agent cognition and decision-making

## Design goals

- Venue-agnostic: adaptable to stations, shopping centres, concert venues, etc.
- Scenario-agnostic: experiment differences are expressed in YAML config, not code
- Cheap to run multiple experiments: LLM cost tracking built in
- Cleanly separable: framework has no knowledge of specific venue geometry

## Package structure

```
evacusim/
├── core/           # Abstract base classes: Scenario, agent roles, typed events
├── jps/            # JuPedSim wrappers (single-level and multi-level)
├── concordia/      # Concordia/LLM integration: agent builder, LLM setup
├── translation/    # Sim state ↔ natural language (observations, actions, spatial)
├── systems/        # Event manager, PA/messaging system, staff director
├── metrics/        # Zone population monitor, LLM cost reporter
├── coordination/   # HybridSimulationRunner and supporting processors
├── visualization/  # Viewer launcher, video helper, spatial display
└── utils/          # Logger, performance timer, misc helpers
```

## Installation

```bash
pip install -e .
```

Or from a study repo:

```bash
pip install git+https://github.com/your-org/evacusim.git
```

## Usage

See the [monument-evacuation](https://github.com/your-org/monument-evacuation) study repo for
a worked example of how to use this framework for a real experiment series.
