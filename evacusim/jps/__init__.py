"""
JuPedSim integration layer.

Provides:
- PedestrianSimulation protocol (the interface all backends must satisfy)
- Single-level JuPedSim wrapper
- Multi-level JuPedSim wrapper (e.g. underground stations)
- SUMO network → JuPedSim geometry loader
- Agent position tracker
- Exit manager
"""
