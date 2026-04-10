"""
Translation layer between simulation state and natural language.

Provides:
- ObservationGenerator: sim state → natural language agent observations
- ActionTranslator: NL agent decision → JuPedSim waypoint
- SpatialAnalyzer: geometric proximity and zone queries
- CrowdAnalyzer: crowd density and flow descriptions
- ExitNameRegistry: canonical exit names for agent observations
"""
