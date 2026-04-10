"""
Decision-making: prompt caching, action execution, and batch decision processing.

Provides:
- DecisionProcessor: batches LLM agent decisions each decision interval
- ActionExecutor: translates resolved actions into JuPedSim waypoint changes
- PromptCache: deduplicates identical LLM prompts within a decision cycle
- ActionUtils: shared helpers for action resolution and validation
"""
