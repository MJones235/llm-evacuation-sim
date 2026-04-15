"""
Simulation systems: event scheduling, public address messaging, director agents.

Provides:
- EventManager: timed event scheduler (alarms, PA announcements, exit blocking)
- MessageSystem: proximity-based agent-to-agent and broadcast messaging
- DirectorSystem: rule-based director agents (staff, firefighters, etc.) that
  direct passengers to exits without requiring LLM calls
"""
